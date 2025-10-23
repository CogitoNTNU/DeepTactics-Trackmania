import torch
import torch.nn as nn
import numpy as np
import math
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage, PrioritizedReplayBuffer
from torchrl.modules import NoisyLinear

class Network(nn.Module):
    def __init__(self, img_channels=4, img_height=64, img_width=64, float_inputs_dim=9,
                 num_actions=45, iqn_embedding_dim=64, float_hidden_dim=256, dense_hidden_dim=1024,
                 noisy_std=0.5):
        """
        IQN Network for discrete action Q-value estimation.

        Architecture:
        - Image head: Conv layers for processing screenshots (4x64x64)
        - Float feature extractor: Dense layers for speed/gear/rpm/prev_actions
        - IQN quantile embedding: Cosine basis functions
        - Dueling architecture: Separate value and advantage streams with NoisyLinear
        - Output: Q-value distributions for each discrete action

        Args:
            img_channels: number of image channels (4 for frame history)
            img_height: image height (64)
            img_width: image width (64)
            float_inputs_dim: number of float features (speed, gear, rpm, 2 prev actions = 9)
            num_actions: number of discrete actions (default: 45 = 5 gas × 9 steering)
            iqn_embedding_dim: quantile embedding dimension (64)
            float_hidden_dim: hidden dim for float features (256)
            dense_hidden_dim: hidden dim for final layers (1024)
            noisy_std: standard deviation for NoisyLinear layers (default: 0.5)
        """
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.iqn_embedding_dim = iqn_embedding_dim
        self.num_actions = num_actions

        # Image head (convolutional layers)
        self.img_head = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=16, kernel_size=(4, 4), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
        ).to(self.device)

        # Calculate conv output size
        with torch.no_grad():
            dummy_img = torch.zeros(1, img_channels, img_height, img_width).to(self.device)
            conv_output_size = self.img_head(dummy_img).shape[1]

        # Float feature extractor
        self.float_feature_extractor = nn.Sequential(
            nn.Linear(float_inputs_dim, float_hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(float_hidden_dim, float_hidden_dim),
            nn.LeakyReLU(inplace=True),
        ).to(self.device)

        # Combined feature dimension
        dense_input_dim = conv_output_size + float_hidden_dim

        # IQN embedding layer for quantile values
        self.iqn_fc = nn.Sequential(
            nn.Linear(iqn_embedding_dim, dense_input_dim),
            nn.LeakyReLU(inplace=True)
        ).to(self.device)

        # Dueling architecture for Q-values with NoisyLinear for exploration
        # Advantage head outputs per-action advantages
        self.A_fc1 = NoisyLinear(dense_input_dim, dense_hidden_dim, std_init=noisy_std, device=self.device)
        self.A_fc2 = NoisyLinear(dense_hidden_dim, num_actions, std_init=noisy_std, device=self.device)

        # Value head outputs state value
        self.V_fc1 = NoisyLinear(dense_input_dim, dense_hidden_dim, std_init=noisy_std, device=self.device)
        self.V_fc2 = NoisyLinear(dense_hidden_dim, 1, std_init=noisy_std, device=self.device)

    def forward(self, img: torch.Tensor, float_inputs: torch.Tensor, n_tau: int = 8):
        """
        Forward pass for IQN Q-value distribution estimation.

        Args:
            img: (batch, img_channels, img_height, img_width) - screenshot history
            float_inputs: (batch, float_inputs_dim) - speed, gear, rpm, prev_actions
            n_tau: number of quantile samples

        Returns:
            quantile_values: (batch, n_tau, num_actions) - quantile Q-values for each action
            taus: (batch * n_tau, 1) - quantile samples used
        """
        batch_size = img.shape[0]

        # Extract features from image head
        img_features = self.img_head(img)  # (batch, conv_output_size)

        # Extract features from float inputs
        float_features = self.float_feature_extractor(float_inputs)  # (batch, float_hidden_dim)

        # Concatenate features
        concat = torch.cat((img_features, float_features), dim=1)  # (batch, dense_input_dim)

        # Sample quantiles (tau) uniformly from [0, 1]
        tau = torch.rand(size=(batch_size * n_tau, 1), device=self.device, dtype=torch.float32)

        # IQN quantile embedding using cosine basis functions
        # tau: (batch * n_tau, 1), expand for cosine basis
        i_pi = torch.arange(1, self.iqn_embedding_dim + 1, 1, device=self.device, dtype=torch.float32) * math.pi
        quantile_net = torch.cos(tau * i_pi)  # (batch * n_tau, iqn_embedding_dim)
        quantile_net = self.iqn_fc(quantile_net)  # (batch * n_tau, dense_input_dim)

        # Hadamard product: repeat state features and element-wise multiply with quantile embedding
        concat = concat.repeat_interleave(n_tau, dim=0)  # (batch * n_tau, dense_input_dim)
        concat = concat * quantile_net

        # Dueling architecture with NoisyLinear layers
        # Advantage stream
        A = self.A_fc1(concat)
        A = nn.functional.leaky_relu(A, inplace=True)
        A = self.A_fc2(A)  # (batch * n_tau, num_actions)

        # Value stream
        V = self.V_fc1(concat)
        V = nn.functional.leaky_relu(V, inplace=True)
        V = self.V_fc2(V)  # (batch * n_tau, 1)

        # Combine: Q = V + (A - mean(A))
        Q = V + A - A.mean(dim=-1, keepdim=True)  # (batch * n_tau, num_actions)

        # Reshape to (batch, n_tau, num_actions)
        quantile_values = Q.view(batch_size, n_tau, self.num_actions)

        return quantile_values, tau


class IQN:
    def __init__(self,
                 num_actions=45,
                 n_tau_train=64,  # Increased from 32 for better distribution approximation
                 n_tau_action=8,
                 cosine_dim=64,
                 learning_rate=0.00025,
                 batch_size=32,
                 discount_factor=0.99,
                 use_prioritized_replay=True,
                 alpha=0.6,
                 beta=0.4,
                 beta_increment=0.001,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 kappa=1.0,  # Huber loss threshold
                 ):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Store configuration for W&B logging
        self.config = {
            'num_actions': num_actions,
            'n_tau_train': n_tau_train,
            'n_tau_action': n_tau_action,
            'cosine_dim': cosine_dim,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'discount_factor': discount_factor,
            'use_prioritized_replay': use_prioritized_replay,
            'alpha': alpha,
            'beta': beta,
            'beta_increment': beta_increment,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'kappa': kappa,
        }

        self.num_actions = num_actions
        self.n_tau_train = n_tau_train
        self.n_tau_action = n_tau_action
        self.kappa = kappa

        # Epsilon-greedy exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Initialize networks with proper architecture
        # Note: These will be overridden in env.py with proper dimensions
        self.policy_network = Network(num_actions=num_actions, iqn_embedding_dim=cosine_dim).to(self.device)
        self.target_network = Network(num_actions=num_actions, iqn_embedding_dim=cosine_dim).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.use_prioritized_replay = use_prioritized_replay
        self.beta_increment = beta_increment
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(alpha=alpha, beta=beta, storage=LazyTensorStorage(max_size=50000), batch_size=batch_size)
        else:
            self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=50000), batch_size=batch_size)

        self.batch_size = batch_size
        self.discount_factor = discount_factor
        # Use AdamW for better generalization (weight decay built-in)
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=learning_rate)

    def store_transition(self, transition: TensorDict):
        self.replay_buffer.add(transition)

    def get_experience(self):
        if self.use_prioritized_replay:
            sample, info = self.replay_buffer.sample(return_info=True)
            return sample, info['index'], info['_weight']
        else:
            sample = self.replay_buffer.sample()
            return sample, None, None

    def get_action(self, img: torch.Tensor, float_inputs: torch.Tensor, explore=True):
        """
        Get discrete action index using epsilon-greedy strategy.

        Args:
            img: (1, img_channels, img_height, img_width) - screenshot history
            float_inputs: (1, float_inputs_dim) - speed, gear, rpm, prev_actions
            explore: whether to use epsilon-greedy exploration

        Returns:
            action_idx: integer index of the discrete action to take
        """
        # Epsilon-greedy exploration
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)

        with torch.no_grad():
            quantile_values, _ = self.policy_network.forward(
                img.to(device=self.device),
                float_inputs.to(device=self.device),
                n_tau=self.n_tau_action
            )
            # quantile_values: (1, n_tau, num_actions)
            # Average over quantiles to get Q-values
            q_values = quantile_values.mean(dim=1)  # (1, num_actions)
            # Select action with highest Q-value
            action_idx = q_values.argmax(dim=1).item()
            return action_idx

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def quantile_huber_loss(self, quantile_values, target_quantiles, taus):
        """
        Compute quantile Huber loss for IQN.

        Args:
            quantile_values: (batch, n_tau, 1) - predicted quantile values for taken actions
            target_quantiles: (batch, n_tau', 1) - target quantile values
            taus: (batch * n_tau, 1) - quantile fractions

        Returns:
            loss: scalar loss value
            td_errors: (batch,) - TD errors for prioritized replay
        """
        batch_size = quantile_values.shape[0]
        n_tau = quantile_values.shape[1]
        n_tau_prime = target_quantiles.shape[1]

        # Reshape for pairwise difference computation
        # quantile_values: (batch, n_tau, 1) -> (batch, n_tau, 1, 1)
        # target_quantiles: (batch, 1, n_tau', 1) -> (batch, 1, n_tau', 1)
        quantile_values = quantile_values.unsqueeze(2)  # (batch, n_tau, 1, 1)
        target_quantiles = target_quantiles.unsqueeze(1)  # (batch, 1, n_tau', 1)

        # TD error for each pair: δ = target - prediction
        td_errors_pairwise = target_quantiles - quantile_values  # (batch, n_tau, n_tau', 1)

        # Huber loss: smooth L1 loss
        huber_loss = torch.where(
            td_errors_pairwise.abs() <= self.kappa,
            0.5 * td_errors_pairwise.pow(2),
            self.kappa * (td_errors_pairwise.abs() - 0.5 * self.kappa)
        )  # (batch, n_tau, n_tau', 1)

        # Quantile regression loss: weight by quantile asymmetry
        # Reshape taus to (batch, n_tau, 1, 1)
        taus = taus.view(batch_size, n_tau, 1, 1)
        quantile_weight = torch.abs(taus - (td_errors_pairwise < 0).float())  # (batch, n_tau, n_tau', 1)

        # Combine Huber loss with quantile weights
        quantile_huber = quantile_weight * huber_loss  # (batch, n_tau, n_tau', 1)

        # Average over target quantiles (n_tau') and sum over quantiles (n_tau)
        loss = quantile_huber.sum(dim=1).mean(dim=1).squeeze(-1)  # (batch,)

        # TD error for prioritized replay: mean absolute TD error
        td_error_for_priority = td_errors_pairwise.abs().mean(dim=(1, 2)).squeeze(-1)  # (batch,)

        return loss, td_error_for_priority

    def get_loss(self, experiences):
        """
        Compute IQN quantile regression loss.

        Returns:
            loss: scalar loss
            td_errors: (batch,) - TD errors for prioritized replay
        """
        # Unpack observations
        imgs = experiences["img"].to(self.device)
        float_inputs = experiences["float_inputs"].to(self.device)
        next_imgs = experiences["next_img"].to(self.device)
        next_float_inputs = experiences["next_float_inputs"].to(self.device)

        actions = experiences["action"].to(self.device, dtype=torch.long)  # Discrete action indices
        rewards = experiences["reward"].to(self.device, dtype=torch.float32).unsqueeze(-1)  # (batch, 1)
        dones = experiences["done"].to(self.device, dtype=torch.bool).unsqueeze(-1)  # (batch, 1)

        batch_size = imgs.shape[0]

        # Get current quantile values for taken actions
        quantile_values, taus = self.policy_network.forward(imgs, float_inputs, n_tau=self.n_tau_train)
        # quantile_values: (batch, n_tau, num_actions)
        # Select Q-values for taken actions
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(batch_size, self.n_tau_train, 1)
        current_quantiles = quantile_values.gather(2, actions_expanded)  # (batch, n_tau, 1)

        # Compute target quantiles
        with torch.no_grad():
            # Get next state quantile values
            next_quantile_values, _ = self.target_network.forward(next_imgs, next_float_inputs, n_tau=self.n_tau_train)
            # next_quantile_values: (batch, n_tau', num_actions)

            # Select best action using policy network (double DQN)
            policy_next_quantiles, _ = self.policy_network.forward(next_imgs, next_float_inputs, n_tau=self.n_tau_action)
            next_q_values = policy_next_quantiles.mean(dim=1)  # (batch, num_actions)
            next_actions = next_q_values.argmax(dim=1)  # (batch,)

            # Get target quantiles for best action
            next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(2).expand(batch_size, self.n_tau_train, 1)
            target_quantiles = next_quantile_values.gather(2, next_actions_expanded)  # (batch, n_tau', 1)

            # Bellman update: r + γ * Q_target(s', a*)
            target_quantiles = rewards.unsqueeze(1) + self.discount_factor * target_quantiles * (~dones).unsqueeze(1).float()

        # Compute quantile Huber loss
        loss, td_errors = self.quantile_huber_loss(current_quantiles, target_quantiles, taus)

        return loss, td_errors

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        if self.use_prioritized_replay:
            experiences, idxs, weights = self.get_experience()
            weights = weights.to(self.device)
        else:
            experiences, _, _ = self.get_experience()
            weights = torch.ones(self.batch_size, device=self.device)

        per_sample_losses, td_errors = self.get_loss(experiences)

        if self.use_prioritized_replay:
            # Update priorities in replay buffer
            self.replay_buffer.update_priority(idxs, td_errors.detach())
            # Weight loss by importance sampling weights
            loss = (per_sample_losses * weights).mean()

            # Anneal beta
            current_beta = self.replay_buffer._sampler.beta
            self.replay_buffer._sampler.beta = min(1.0, current_beta + self.beta_increment)
        else:
            loss = per_sample_losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()