import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from config_files import tm_config
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage, PrioritizedReplayBuffer


class Network(nn.Module):
    def __init__(
        self,
        input_dim=8,
        hidden_dim=128,
        output_dim=4,
        use_dueling=tm_config.use_dueling,
    ):
        super().__init__()
        self.use_dueling = use_dueling
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if use_dueling:
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
    def __init__(
        self,
        e_start=1.0,
        e_end=0.01,
        e_decay_rate=0.996,
        batch_size=64,
        discount_factor=0.99,
        use_prioritized_replay=True,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001, # reach 1 in about 600 steps
    ):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.policy_network = Network().to(self.device)
        self.target_network = Network().to(self.device)

        self.use_prioritized_replay = use_prioritized_replay
        self.beta_increment = beta_increment
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(alpha=alpha, beta=beta, storage=LazyTensorStorage(max_size=10000), batch_size=batch_size)
        else:
            self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000), batch_size=batch_size)

        self.eps = e_start
        self.e_end = e_end
        self.e_decay_rate = e_decay_rate
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=0.001)

    def store_transition(self, transition: TensorDict):
        self.replay_buffer.add(transition)

    def get_experience(self):
        if self.use_prioritized_replay:
            sample, info = self.replay_buffer.sample(return_info=True)
            return sample, info['index'], info['_weight']
        else:
            sample = self.replay_buffer.sample()
            return sample, None, None

    def get_action(self, obs, n_tau=None) -> int:
        if self.eps > self.e_end:
            self.eps *= self.e_decay_rate

        if random.random() < self.eps:
            return random.randint(0, 3), None
        else:
            with torch.no_grad():
                actions = self.policy_network(obs.to(device=self.device))
                n = torch.argmax(actions)
                return int(n.item()), int(actions.max().item())

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

        # Extract from TensorDict (torchrl ReplayBuffer format)
        states = experiences["observation"].to(self.device)
        next_states = experiences["next_observation"].to(self.device)
        actions = experiences["action"].to(self.device)
        rewards = experiences["reward"].to(self.device, dtype=torch.float32)
        dones = experiences["done"].to(self.device, dtype=torch.bool)

        q_values = self.policy_network(states)
        policy_predictions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN som bruker policy-network sin next-action for at target-network kan predikere med den
        with torch.no_grad():
            if self.config.use_doubleDQN:
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

