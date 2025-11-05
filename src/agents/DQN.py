import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from config_files.config import Config
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage, PrioritizedReplayBuffer


class Network(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.use_dueling = config.use_dueling
        input_dim = config.input_dim
        hidden_dim = config.hidden_dim
        output_dim = config.output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if self.use_dueling:
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )

            self.advantage = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        if not self.use_dueling:
            return x
        else:
            x = F.relu(x)
            v = self.value(x)
            a = self.advantage(x)

            a_mean = a.mean(dim=1, keepdim=True)
            q = v + (a - a_mean)
            return q


class DQN:
    def __init__(self, config=Config()):
        self.batch_size = config.batch_size
        self.discount_factor = config.discount_factor
        self.use_prioritized_replay = config.use_prioritized_replay
        self.use_doubleDQN = config.use_doubleDQN
        self.use_dueling = config.use_dueling
        self.alpha = config.alpha
        self.beta = config.beta
        self.beta_increment = config.beta_increment
        self.max_buffer_size = config.max_buffer_size
        self.epsilon = config.epsilon_start
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.output_dim = config.output_dim
        self.learning_rate_start = config.learning_rate_start
        self.learning_rate_end = config.learning_rate_end
        self.cosine_annealing_decay_episodes = config.cosine_annealing_decay_episodes
        self.tau = config.tau
        self.config = config

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Store configuration for W&B logging
        self.config = {
            'agent_type': 'DQN',
            'use_dueling': self.use_dueling,
            'use_prioritized_replay': self.use_prioritized_replay,
            'use_doubleDQN': self.use_doubleDQN,
            'batch_size': self.batch_size,
            'discount_factor': self.discount_factor,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
        }

        self.policy_network = Network(config).to(self.device)
        self.target_network = Network(config).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                alpha=self.alpha,
                beta=self.beta,
                storage=LazyTensorStorage(self.max_buffer_size),
                batch_size=self.batch_size
            )
        else:
            self.replay_buffer = ReplayBuffer(
                storage=LazyTensorStorage(self.max_buffer_size),
                batch_size=self.batch_size
            )

        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=self.learning_rate_start)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cosine_annealing_decay_episodes,
            eta_min=self.learning_rate_end
        )

    def store_transition(self, transition: TensorDict):
        self.replay_buffer.add(transition)

    def get_experience(self):
        if self.use_prioritized_replay:
            sample, info = self.replay_buffer.sample(return_info=True)
            return sample, info['index'], info['_weight']
        else:
            sample = self.replay_buffer.sample()
            return sample, None, None

    def get_action(self, obs) -> tuple[int, float | None]:
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1), None
        else:
            with torch.no_grad():
                q_values = self.policy_network(obs.to(device=self.device))
                best_action = torch.argmax(q_values)
                return int(best_action.item()), float(q_values.max().item())

    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Soft update of target network parameters: θ_target = τ*θ_policy + (1-τ)*θ_target"""
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def train(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        if self.use_prioritized_replay:
            experiences, idxs, weights = self.get_experience()
            weights = weights.to(self.device)
        else:
            experiences, _, _ = self.get_experience()
            weights = None

        # Extract from TensorDict (torchrl ReplayBuffer format)
        states = experiences["observation"].to(self.device)
        next_states = experiences["next_observation"].to(self.device)
        actions = experiences["action"].to(self.device)
        rewards = experiences["reward"].to(self.device, dtype=torch.float32)
        dones = experiences["done"].to(self.device, dtype=torch.bool)

        q_values = self.policy_network(states)
        policy_predictions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN: policy network selects actions, target network evaluates them
        with torch.no_grad():
            if self.use_doubleDQN:
                next_policy_q = self.policy_network(next_states)
                next_actions = next_policy_q.argmax(dim=1)
                next_target_q = self.target_network(next_states)
                next_q_values = next_target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Regular DQN
                next_target_q = self.target_network(next_states)
                next_q_values = next_target_q.max(dim=1)[0]

        targets = torch.where(
            dones, rewards, rewards + self.discount_factor * next_q_values
        )

        # Calculate TD errors (Absolutt-verdi og stabiliserer?)
        td_errors = policy_predictions - targets

        if self.use_prioritized_replay:
            # Update priorities in replay buffer (use absolute TD errors)
            self.replay_buffer.update_priority(idxs, td_errors.abs().detach())

            # Apply importance sampling weights to loss
            loss_func = nn.SmoothL1Loss(reduction="none")
            losses = loss_func(policy_predictions, targets)
            weighted_losses = losses * weights
            loss = weighted_losses.mean()

            # Anneal beta towards 1.0 to reduce bias over time
            current_beta = self.replay_buffer._sampler.beta
            self.replay_buffer._sampler.beta = min(1.0, current_beta + self.beta_increment)
        else:
            loss_func = nn.SmoothL1Loss()
            loss = loss_func(policy_predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

