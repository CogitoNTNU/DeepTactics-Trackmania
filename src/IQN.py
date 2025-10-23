from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import NoisyLinear
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage, PrioritizedReplayBuffer
from tmrl import get_environment
from time import sleep
import numpy as np

class Network(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=4, cosine_dim=32, noisy_std=0.5, use_dueling=True):
        super().__init__()
        self.cosine_dim = cosine_dim
        self.use_dueling = use_dueling
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.tau_embedding_fc1 = nn.Linear(cosine_dim, hidden_dim, device=self.device)

        self.fc1 = nn.Linear(input_dim, hidden_dim, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, device=self.device)

        if use_dueling:
            self.value_fc1 = NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std, device=self.device)
            self.value_fc2 = NoisyLinear(hidden_dim, 1, std_init=noisy_std, device=self.device)

            self.advantage_fc1 = NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std, device=self.device)
            self.advantage_fc2 = NoisyLinear(hidden_dim, output_dim, std_init=noisy_std, device=self.device)
        else:
            self.fc4 = NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std, device=self.device)
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

    def forward(self, x: torch.Tensor, n_tau: int = 8):
        batch_size = x.shape[0]

        # Shared encoder
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        # Quantile embedding
        tau_x, taus = self.tau_forward(batch_size, n_tau)

        # Merge state and quantile embeddings
        x = x.unsqueeze(dim=1)
        x = x * tau_x

        if self.use_dueling:
            # Value stream: V(s,τ) for each quantile
            v = self.value_fc1(x)
            v = F.relu(v)
            v = self.value_fc2(v)  # (batch, n_tau, 1)

            # Advantage stream: A(s,a,τ) for each action and quantile
            a = self.advantage_fc1(x)
            a = F.relu(a)
            a = self.advantage_fc2(a)  # (batch, n_tau, n_actions)

            # Combine: Q(s,a,τ) = V(s,τ) + (A(s,a,τ) - mean_a A(s,a,τ))
            a_mean = a.mean(dim=2, keepdim=True)
            q = v + (a - a_mean)
        else:
            q = self.fc4(x)
            q = F.relu(q)
            q = self.fc5(q)

        return q, taus


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

        self.policy_network = Network(cosine_dim=cosine_dim).to(self.device)
        self.target_network = Network(cosine_dim=cosine_dim).to(self.device)

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

    def get_best_action(self, network: nn.Module, obs: torch.Tensor, n_tau=None):
        """Get the best action for a given observation using the provided network."""
        if n_tau is None:
            n_tau = self.n_tau_train
        with torch.no_grad():
            action_quantiles, _ = network.forward(obs.to(self.device), n_tau)
            q_values = action_quantiles.mean(dim=1)
            return q_values.argmax(dim=1)

    def get_action(self, obs: torch.Tensor, n_tau=None) -> tuple[int, Optional[float]]:
        if n_tau is None:
            n_tau = self.n_tau_action

        with torch.no_grad():
            actions_quantiles, _ = self.policy_network.forward(
                    obs.to(device=self.device), n_tau
                )
            q_values = actions_quantiles.mean(dim=1)
            best_action = torch.argmax(q_values, dim=1)
            return int(best_action.item()), float(q_values.max().item())

    def get_loss(self, experiences):
        states = experiences["observation"].to(self.device)
        next_states = experiences["next_observation"].to(self.device)
        actions = experiences["action"].to(self.device)
        rewards = experiences["reward"].to(self.device, dtype=torch.float32)
        dones = experiences["done"].to(self.device, dtype=torch.bool)

        policy_predictions, policy_quantiles = self.policy_network.forward(states, n_tau=self.n_tau_train)

        # DDQN: policy network selects actions, target network evaluates them
        with torch.no_grad():
            next_actions = self.get_best_action(self.policy_network, next_states, n_tau=self.n_tau_train)
            next_target_q, target_quantiles = self.target_network.forward(next_states, n_tau=self.n_tau_train)

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