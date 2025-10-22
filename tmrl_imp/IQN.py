import torch
import torch.nn as nn
import math
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage, PrioritizedReplayBuffer

class Network(nn.Module):
    def __init__(self, img_channels=4, img_height=64, img_width=64, float_inputs_dim=9,
                 action_dim=3, iqn_embedding_dim=64, float_hidden_dim=256, dense_hidden_dim=1024):
        """
        IQN Network matching Linesight's architecture for continuous actions.

        Architecture:
        - Image head: Conv layers for processing screenshots (4x64x64)
        - Float feature extractor: Dense layers for speed/gear/rpm/prev_actions
        - IQN quantile embedding: Cosine basis functions
        - Dueling architecture: Separate value and advantage streams
        - Output: Continuous actions [gas, brake, steer] in [-1, 1]

        Args:
            img_channels: number of image channels (4 for frame history)
            img_height: image height (64)
            img_width: image width (64)
            float_inputs_dim: number of float features (speed, gear, rpm, 2 prev actions = 9)
            action_dim: action dimension (3 for TrackMania)
            iqn_embedding_dim: quantile embedding dimension (64)
            float_hidden_dim: hidden dim for float features (256)
            dense_hidden_dim: hidden dim for final layers (1024)
        """
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.iqn_embedding_dim = iqn_embedding_dim
        self.action_dim = action_dim

        # Image head (convolutional layers) - matches Linesight
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

        # Float feature extractor - matches Linesight
        self.float_feature_extractor = nn.Sequential(
            nn.Linear(float_inputs_dim, float_hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(float_hidden_dim, float_hidden_dim),
            nn.LeakyReLU(inplace=True),
        ).to(self.device)

        # Combined feature dimension
        dense_input_dim = conv_output_size + float_hidden_dim

        # IQN embedding layer
        self.iqn_fc = nn.Sequential(
            nn.Linear(iqn_embedding_dim, dense_input_dim),
            nn.LeakyReLU(inplace=True)
        ).to(self.device)

        # Dueling architecture for continuous actions
        # Advantage head outputs action values
        self.A_head = nn.Sequential(
            nn.Linear(dense_input_dim, dense_hidden_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dense_hidden_dim // 2, action_dim),
        ).to(self.device)

        # Value head outputs state value
        self.V_head = nn.Sequential(
            nn.Linear(dense_input_dim, dense_hidden_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dense_hidden_dim // 2, 1),
        ).to(self.device)

    def forward(self, img: torch.Tensor, float_inputs: torch.Tensor, n_tau: int = 8):
        """
        Forward pass matching Linesight's IQN architecture for continuous actions.

        Args:
            img: (batch, img_channels, img_height, img_width) - screenshot history
            float_inputs: (batch, float_inputs_dim) - speed, gear, rpm, prev_actions
            n_tau: number of quantile samples

        Returns:
            actions: (batch, action_dim) - continuous actions in [-1, 1]
            taus: (batch * n_tau, 1) - quantile samples used
        """
        batch_size = img.shape[0]

        # Extract features from image head
        img_features = self.img_head(img)  # (batch, conv_output_size)

        # Extract features from float inputs
        float_features = self.float_feature_extractor(float_inputs)  # (batch, float_hidden_dim)

        # Concatenate features
        concat = torch.cat((img_features, float_features), dim=1)  # (batch, dense_input_dim)

        # Sample quantiles (tau) - symmetric sampling like Linesight
        tau = (
            torch.arange(n_tau // 2, device=self.device, dtype=torch.float32)
            .repeat_interleave(batch_size).unsqueeze(1)
            + torch.rand(size=(batch_size * n_tau // 2, 1), device=self.device, dtype=torch.float32)
        ) / n_tau
        tau = torch.cat((tau, 1 - tau), dim=0)  # Symmetric around 0.5

        # IQN quantile embedding using cosine basis functions
        quantile_net = torch.cos(
            torch.arange(1, self.iqn_embedding_dim + 1, 1, device=self.device) * math.pi * tau
        )
        quantile_net = quantile_net.expand([-1, self.iqn_embedding_dim])
        quantile_net = self.iqn_fc(quantile_net)  # (batch * n_tau, dense_input_dim)

        # Hadamard product: repeat state features and element-wise multiply with quantile embedding
        concat = concat.repeat(n_tau, 1)  # (batch * n_tau, dense_input_dim)
        concat = concat * quantile_net

        # Dueling architecture
        A = self.A_head(concat)  # (batch * n_tau, action_dim) - advantage
        V = self.V_head(concat)  # (batch * n_tau, 1) - value

        # Combine: Q = V + (A - mean(A))
        Q = V + A - A.mean(dim=-1, keepdim=True)  # (batch * n_tau, action_dim)

        # Reshape to (batch, n_tau, action_dim) and average over quantiles
        Q = Q.view(batch_size, n_tau, self.action_dim)
        actions = Q.mean(dim=1)  # (batch, action_dim)

        # Apply tanh to bound to [-1, 1]
        actions = torch.tanh(actions)

        return actions, tau


class IQN:
    def __init__(self,
                 n_tau_train=64,
                 n_tau_action=64,
                 cosine_dim=32,
                 learning_rate=0.00025,
                 batch_size=64,
                 discount_factor=0.99,
                 use_prioritized_replay=True,
                 alpha=0.6,
                 beta=0.4,
                 beta_increment=0.001,
                 ):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Store configuration for W&B logging
        self.config = {
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
        }

        self.n_tau_train = n_tau_train
        self.n_tau_action = n_tau_action

        # Initialize networks with proper architecture
        # Note: These will be overridden in env.py with proper dimensions
        self.policy_network = Network().to(self.device)
        self.target_network = Network().to(self.device)

        self.use_prioritized_replay = use_prioritized_replay
        self.beta_increment = beta_increment
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(alpha=alpha, beta=beta, storage=LazyTensorStorage(max_size=10000), batch_size=batch_size)
        else:
            self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000), batch_size=batch_size)

        self.batch_size = batch_size
        self.discount_factor = discount_factor
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

    def get_action(self, img: torch.Tensor, float_inputs: torch.Tensor, n_tau=None):
        """
        Get continuous action for given observation.

        Args:
            img: (1, img_channels, img_height, img_width) - screenshot history
            float_inputs: (1, float_inputs_dim) - speed, gear, rpm, prev_actions

        Returns:
            action: numpy array of shape (action_dim,) with values in [-1, 1]
                    For TrackMania: [gas, brake, steer]
        """
        if n_tau is None:
            n_tau = self.n_tau_action

        with torch.no_grad():
            actions, _ = self.policy_network.forward(
                img.to(device=self.device),
                float_inputs.to(device=self.device),
                n_tau
            )
            # actions is (batch, action_dim), we want (action_dim,)
            return actions.squeeze(0).cpu().numpy()

    def get_loss(self, experiences):
        """
        Compute loss for continuous action training using MSE.

        For continuous actions, we use simple regression loss between
        predicted actions and taken actions, weighted by rewards.
        """
        # Unpack observations (img and float_inputs are separate)
        imgs = experiences["img"].to(self.device)
        float_inputs = experiences["float_inputs"].to(self.device)
        next_imgs = experiences["next_img"].to(self.device)
        next_float_inputs = experiences["next_float_inputs"].to(self.device)

        actions = experiences["action"].to(self.device, dtype=torch.float32)  # Continuous actions
        rewards = experiences["reward"].to(self.device, dtype=torch.float32)
        dones = experiences["done"].to(self.device, dtype=torch.bool)

        # Get predicted actions from policy network
        predicted_actions, _ = self.policy_network.forward(imgs, float_inputs, n_tau=self.n_tau_train)

        # Get next actions from target network
        with torch.no_grad():
            next_actions, _ = self.target_network.forward(next_imgs, next_float_inputs, n_tau=self.n_tau_train)

        # Compute TD targets: r + γ * value_of_next_action (simplified for continuous)
        # For continuous actions, we use the reward as direct supervision
        # and penalize deviation from successful actions
        with torch.no_grad():
            # Actions that led to positive rewards should be reinforced
            target_actions = actions.clone()

        # MSE loss between predicted and target actions
        # Weight by rewards to reinforce good actions
        action_diff = (predicted_actions - target_actions).pow(2).sum(dim=1)  # (batch,)

        # Reward weighting: positive rewards reinforce the action, negative penalize
        reward_weights = torch.abs(rewards)  # Use absolute value for weighting
        per_sample_losses = action_diff * (1.0 + reward_weights)  # Amplify loss based on reward magnitude

        # For PER, we use action difference as TD error
        per_sample_td_errors = action_diff.sqrt()  # RMSE per sample

        return per_sample_losses, per_sample_td_errors

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
            weights = None

        per_sample_losses, td_errors = self.get_loss(experiences)

        if self.use_prioritized_replay:
            self.replay_buffer.update_priority(idxs, td_errors.detach())
            loss = (per_sample_losses * weights).mean()

            current_beta = self.replay_buffer._sampler.beta
            self.replay_buffer._sampler.beta = min(1.0, current_beta + self.beta_increment)
        else:
            loss = per_sample_losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()