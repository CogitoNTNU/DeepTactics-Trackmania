from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import NoisyLinear, reset_noise
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage, PrioritizedReplayBuffer
from config_files.tm_config import Config

config = Config()

class Network(nn.Module):
    def __init__(self, config = Config()):
        super().__init__()
        self.img_x = config.img_x
        self.img_y = config.img_y
        cosine_dim = config.cosine_dim
        use_dueling = config.use_dueling
        cosine_dim = config.cosine_dim
        use_dueling = config.use_dueling
        hidden_dim = config.hidden_dim
        output_dim = config.output_dim
        cosine_dim = config.cosine_dim
        noisy_std= config.noisy_std
        input_car_dim = config.input_car_dim
        # self.input_x = config.input_x hvar her tidligere usikker om brukt
        # self.input_y = config.input_y
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        hidden_dim1 = config.hidden_dim1
        hidden_dim2 = config.hidden_dim2

        car_feature_hidden_dim = config.car_feature_hidden_dim
        conv_hidden_size = int(hidden_dim2 * 6 * 6)
        dense_input_size = conv_hidden_size +  car_feature_hidden_dim # image features + car features

        self.tau_embedding_fc1 = nn.Linear(cosine_dim, dense_input_size, device=self.device)

        self.conv = nn.Sequential(
            nn.Conv2d(
                4, hidden_dim1, stride=2, kernel_size=3, padding=1
            ),  # floor((96-3+2*2)/2)+1 = 48
            nn.BatchNorm2d(hidden_dim1),
            nn.Conv2d(
                hidden_dim1, hidden_dim2, stride=2, kernel_size=3, padding=1
            ),  # floor((48-3+2*2)/2)+1 = 24
            nn.BatchNorm2d(hidden_dim2),
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=3, stride=2, padding=1),  # -> 12x12
            nn.BatchNorm2d(hidden_dim2),
            nn.Conv2d(
                hidden_dim2, hidden_dim2, kernel_size=3, stride=2, padding=1
            ),  # -> 6 "pixels" x 6 "pixels" x 64 planes
        ).to(self.device)

        self.car_feature_fc = nn.Sequential(
            nn.Linear(input_car_dim, car_feature_hidden_dim, device=self.device),
            nn.LeakyReLU(),
            nn.Linear(car_feature_hidden_dim, car_feature_hidden_dim, device=self.device),
            nn.LeakyReLU(),
        ).to(self.device)


        if use_dueling:
            self.value_fc1 = NoisyLinear(dense_input_size, hidden_dim, std_init=noisy_std, device=self.device)
            self.value_fc2 = NoisyLinear(hidden_dim, 1, std_init=noisy_std, device=self.device)

            self.advantage_fc1 = NoisyLinear(dense_input_size, hidden_dim, std_init=noisy_std, device=self.device)
            self.advantage_fc2 = NoisyLinear(hidden_dim, output_dim, std_init=noisy_std, device=self.device)
        else:
            self.fc4 = NoisyLinear(dense_input_size, hidden_dim, std_init=noisy_std, device=self.device)
            self.fc5 = NoisyLinear(hidden_dim, output_dim, std_init=noisy_std, device=self.device)

    def tau_forward(self, batch_size, n_tau):
        taus = torch.rand((batch_size, n_tau, 1), device = self.device)
        cosine_values = torch.arange(self.cosine_dim, device = self.device) * torch.pi
        cosine_values = cosine_values.unsqueeze(0).unsqueeze(0)

        embedded_taus = torch.cos(taus * cosine_values)
        embedded_taus = embedded_taus.to(self.device)

        tau_x = self.tau_embedding_fc1.forward(embedded_taus)
        tau_x = F.relu(tau_x)
        return tau_x, taus

    def forward(self, image: torch.Tensor, features: torch.Tensor , n_tau: int = 8):
        
        batch_size = image.shape[0]
        
        # Shared encoder
        activation_maps = self.conv(image)
        activation_maps = torch.flatten(activation_maps, start_dim=1)

        # Process car features
        car_features = self.car_feature_fc(features)
      
        car_features = torch.flatten(car_features, start_dim=1)
       
       
        activation_maps = torch.cat([activation_maps, car_features], dim=1)
        
        
        # Quantile embedding
        tau_x, taus = self.tau_forward(batch_size, n_tau)

        # Merge state and quantile embeddings
        activation_maps = activation_maps.unsqueeze(dim=1)
        activation_maps = activation_maps * tau_x

        if self.use_dueling:
            # Value stream: V(s,τ) for each quantile
            v = self.value_fc1(activation_maps)
            v = F.relu(v)
            v = self.value_fc2(v)  # (batch, n_tau, 1)

            # Advantage stream: A(s,a,τ) for each action and quantile
            a = self.advantage_fc1(activation_maps)
            a = F.relu(a)
            a = self.advantage_fc2(a)  # (batch, n_tau, n_actions)

            # Combine: Q(s,a,τ) = V(s,τ) + (A(s,a,τ) - mean_a A(s,a,τ))
            a_mean = a.mean(dim=2, keepdim=True)
            q = v + (a - a_mean)
        else:
            q = self.fc4(activation_maps)
            q = F.relu(q)
            q = self.fc5(q)

        return q, taus


class Rainbow:
    def __init__(self, config = Config()):
        #kanskje mulig å fjerne en del av self. ene, men gadd ikke å se på det nå
        self.n_tau_train = config.n_tau_train
        self.n_tau_action= config.n_tau_action
        self.output_dim = config.output_dim
        self.cosine_dim= config.cosine_dim
        self.learning_rate= config.learning_rate
        self.batch_size= config.batch_size
        self.discount_factor= config.discount_factor
        self.use_prioritized_replay= config.use_prioritized_replay
        self.alpha= config.alpha
        self.beta= config.beta
        self.beta_increment= config.beta_increment
        self.max_buffer_size = config.max_buffer_size
        self.epsilon = config.epsilon_start
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Store configuration for W&B logging
        self.config = {
            'n_tau_train': self.n_tau_train,
            'n_tau_action': self.n_tau_action,
            'cosine_dim': self.cosine_dim,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'discount_factor': self.discount_factor,
            'use_prioritized_replay': self.use_prioritized_replay,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
        }

       

        self.policy_network = Network(config).to(self.device)
        self.target_network = Network(config).to(self.device)
        reset_noise(self.policy_network)
        reset_noise(self.target_network)
        self.use_prioritized_replay = self.use_prioritized_replay
        self.beta_increment = self.beta_increment
        
        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(alpha=self.alpha, beta=self.beta, storage=LazyTensorStorage(self.max_buffer_size), batch_size=self.batch_size)
        else:
            self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(self.max_buffer_size), batch_size=self.batch_size)

        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=self.learning_rate)

    def store_transition(self, transition: TensorDict):
        self.replay_buffer.add(transition)

    def get_experience(self):
        if self.use_prioritized_replay:
            sample, info = self.replay_buffer.sample(return_info=True)
            return sample, info['index'], info['_weight']
        else:
            sample = self.replay_buffer.sample()
            return sample, None, None

    def get_best_action(self, network: nn.Module, image: torch.Tensor, car_features: torch.Tensor , n_tau=None):
        """Get the best action for a given observation using the provided network."""
        if n_tau is None:
            n_tau = self.n_tau_train
        with torch.no_grad():
            action_quantiles, _ = network.forward(image.to(self.device),car_features.to(self.device), n_tau)
            q_values = action_quantiles.mean(dim=1)
            return q_values.argmax(dim=1)

    def get_action(self, img: torch.Tensor, car_features: torch.Tensor, n_tau=None, use_epsilon=True) -> tuple[int, Optional[float]]:
        if n_tau is None:
            n_tau = self.n_tau_action
        reset_noise(self.policy_network)
        # Epsilon-greedy exploration
        if use_epsilon and torch.rand(1).item() < self.epsilon:
            # Random action
            action = torch.randint(0, self.output_dim, (1,)).item()
            return int(action), None
        else:
            # Greedy action based on Q-values
            with torch.no_grad():
                actions_quantiles, _ = self.policy_network.forward(
                        img.to(device=self.device), car_features.to(device=self.device) , n_tau
                    )
                q_values = actions_quantiles.mean(dim=1)
                best_action = torch.argmax(q_values, dim=1)
                return int(best_action.item()), float(q_values.max().item())

    def get_loss(self, experiences):
        image = experiences["image"].to(self.device)
        car_features = experiences["car_features"].to(self.device)
        next_image = experiences["next_image"].to(self.device)
        next_car_features = experiences["next_car_features"].to(self.device)
        actions = experiences["action"].to(self.device)
        rewards = experiences["reward"].to(self.device, dtype=torch.float32)
        dones = experiences["done"].to(self.device, dtype=torch.bool)

        reset_noise(self.policy_network)
        policy_predictions, policy_quantiles = self.policy_network.forward(image, car_features, n_tau=self.n_tau_train)

        reset_noise(self.target_network)
        # DDQN: policy network selects actions, target network evaluates them
        with torch.no_grad():
            next_actions = self.get_best_action(self.policy_network, next_image, next_car_features, n_tau=self.n_tau_train)
            next_target_q, _ = self.target_network.forward(next_image, next_car_features, n_tau=self.n_tau_train)

        n_policy_tau = policy_predictions.shape[1]
        n_target_tau = next_target_q.shape[1]

        policy_q_index = actions.unsqueeze(1).unsqueeze(2).expand(-1, n_policy_tau, -1)
        policy_q_selected = policy_predictions.gather(2, policy_q_index).squeeze(2)

        with torch.no_grad():
            target_q_selected = next_target_q.gather(
                2, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, n_target_tau, -1)
            ).squeeze(2)

            target_values = (
                rewards.unsqueeze(1)
                + self.discount_factor
                * target_q_selected
                * (~dones).unsqueeze(1).float()
            )

        # Vectorized pairwise quantile regression: broadcast to (batch, n_policy_tau, n_target_tau)
        policy_q_expanded = policy_q_selected.unsqueeze(2)
        target_values_expanded = target_values.unsqueeze(1)
        td_errors = target_values_expanded - policy_q_expanded

        policy_taus = policy_quantiles.squeeze(2)
        tau_expanded = policy_taus.unsqueeze(2).to(self.device)

        # Quantile regression: |tau - I(td_error < 0)| * huber_loss(td_error)
        indicator = (td_errors < 0).float().to(self.device)
        quantile_weights = torch.abs(tau_expanded - indicator)

        loss_func = nn.SmoothL1Loss(reduction="none")
        huber_loss = loss_func(td_errors, torch.zeros_like(td_errors))
        quantile_loss = quantile_weights * huber_loss

        per_sample_losses = quantile_loss.mean(dim=2).sum(dim=1)
        per_sample_td_errors = td_errors.abs().mean(dim=(1, 2))

        return per_sample_losses, per_sample_td_errors

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        reset_noise(self.target_network)

    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

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

        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 100)
        self.optimizer.step()
        reset_noise(self.policy_network)
        reset_noise(self.target_network)

        return loss.item()

    def save_checkpoint(self, filepath: str, episode: int, step: int, additional_info: dict = None):
        """Save a complete checkpoint of the agent state."""
        checkpoint = {
            'episode': episode,
            'step': step,
            'epsilon': self.epsilon,
            'policy_network_state_dict': self.policy_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }

        # Add replay buffer state if using prioritized replay
        if self.use_prioritized_replay:
            checkpoint['beta'] = self.replay_buffer._sampler.beta

        # Add any additional info (e.g., total reward, wandb run id)
        if additional_info:
            checkpoint['additional_info'] = additional_info

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> dict:
        """Load a checkpoint and restore agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

        # Restore beta if using prioritized replay
        if self.use_prioritized_replay and 'beta' in checkpoint:
            self.replay_buffer._sampler.beta = checkpoint['beta']

        reset_noise(self.policy_network)
        reset_noise(self.target_network)

        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from episode {checkpoint['episode']}, step {checkpoint['step']}")

        return checkpoint