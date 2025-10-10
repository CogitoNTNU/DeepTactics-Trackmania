from typing import Optional
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from config_files import tm_config
from torchrl.modules import NoisyLinear
from src.experience import Experience
from src.replay_buffer import PrioritizedReplayBuffer

#change cuda to cpu or mac alternative
class Network(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=4, cosine_dim=32):
        super().__init__()
        self.cosine_dim = cosine_dim
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.tau_embedding_fc1 = NoisyLinear(cosine_dim, hidden_dim, device=self.device)

        self.fc1 = NoisyLinear(input_dim, hidden_dim, device=self.device)
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim, device=self.device)
        self.fc3 = NoisyLinear(hidden_dim, hidden_dim, device=self.device)
        self.fc4 = NoisyLinear(hidden_dim, hidden_dim, device=self.device)
        self.fc5 = NoisyLinear(hidden_dim, output_dim, device=self.device)

    def tau_forward(self, batch_size, n_tau):
        taus = torch.rand((batch_size, n_tau, 1), device = self.device)
        cosine_values = torch.arange(self.cosine_dim, device = self.device) * torch.pi
        # Cosine. (cosine_dim)
        cosine_values = cosine_values.unsqueeze(0).unsqueeze(0)

        # taus: (batch_size, n_tau, 1)
        # Cosine: (1, 1, cosine_dim)
        embedded_taus = torch.cos(
            taus * cosine_values
        )  # dim: (batch_size, n_tau, cosine_dim)
        #embedded_taus = embedded_taus.to(self.device) #sender tensoren til riktig enhet
        # for hver [cosine_dim] tau - send gjennom et linear layer (tau_embedding_fc1) - og kjør relu på output.
        tau_x = self.tau_embedding_fc1.forward(embedded_taus)
        tau_x = F.relu(
            tau_x
        )  # tensor med shape (n_tau, hidden_dim - dette er vektor med 512 verdier, da må output fra å sende state x inn i netteverket også ha 512 verdier.)
        return tau_x, taus

    def forward(self, x: torch.Tensor, n_tau: int = 8):
        # x: dim (batch_size, action_dim)
        batch_size = x.shape[0]

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        tau_x, taus = self.tau_forward(batch_size, n_tau)

        # x: dim (batch_size, hidden_dim)
        x = x.unsqueeze(dim=1)
        # tau_x: dim (batch_size, n_tau, hidden_dim)
        # x: dim (batch_size, 1, hidden_dim)
        x = x * tau_x
        # x: dim (batch_size, n_tau, hidden_dim)

        # dim: (n_tau, action_size)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)

        return x, taus  # x dim: (batch_size, n_tau, output_dim)


class IQN:
    def __init__(self, e_start=0.9, e_end=0.05, e_decay_rate=0.9999, batch_size=256, 
                 discount_factor=0.99, use_prioritized_replay=True, 
                 alpha=0.6, beta=0.4, beta_increment=0.001):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    
        self.policy_network = Network().to(self.device)
        self.target_network = Network().to(self.device)

        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=10000, alpha=alpha, beta=beta, beta_increment=beta_increment
            )
        else:
            self.replay_buffer = deque(maxlen=10000)

        self.eps = e_start
        self.e_end = e_end
        self.e_decay_rate = e_decay_rate
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=0.002)

    def store_transition(self, transition: Experience):
        if self.use_prioritized_replay:
            self.replay_buffer.add(transition)
        else:
            self.replay_buffer.append(transition)

    def get_experience(self):
        if self.use_prioritized_replay:
            experiences, idxs, weights = self.replay_buffer.sample(self.batch_size)
            return experiences, idxs, weights
        else:
            experiences = random.sample(self.replay_buffer, self.batch_size)
            return experiences, None, None

    def get_best_action(self, network: nn.Module, obs: torch.Tensor, n_tau=8):
        """Get the best action for a given observation using the provided network."""

        action_quantiles, _ = network.forward(obs.to(self.device), n_tau)
        q_values = action_quantiles.mean(dim=1)
        return q_values.argmax(dim=1)

    def get_action(self, obs: torch.Tensor, n_tau=8) -> tuple[int, Optional[float]]:
        if self.eps > self.e_end:
            self.eps *= self.e_decay_rate

        if random.random() < self.eps:
            return random.randint(0, 3), None
        else:
            actions_quantiles, quantiles = self.policy_network.forward(
                obs.to(device=self.device), n_tau
            )
            # (batch_size, n_tau, action_size)
            q_values = actions_quantiles.mean(dim=1)
            best_action = torch.argmax(q_values, dim=1)
            return int(best_action.item()), float(q_values.max().item())

    # room for a lot of improvement, O(n^3)-no tensors
    # for batch_idx in range(policy_quantiles.shape[0]):
    #     curr_loss = 0
    #     for i in range(policy_quantiles.shape[1]):
    #         for j in range(target_quantiles.shape[1]):
    #             action = actions[batch_idx]
    #             next_action = next_actions[batch_idx]
    #             done = dones[batch_idx]
    #             if done:
    #                 td_error = rewards[batch_idx] - policy_predictions[batch_idx][i][action]
    #             else:
    #                 td_error = rewards[batch_idx] + 0.99 * next_target_q[batch_idx][j][next_action] - policy_predictions[batch_idx][i][action]
    #             loss = torch.abs(policy_quantiles[batch_idx][i] - (td_error < 0).float()) * loss_func(td_error, torch.tensor(0.)) / 1
    #             curr_loss += loss

    #     curr_loss /= target_quantiles.shape[1]
    #     tot_loss += curr_loss

    def get_loss(self, experiences):
        states = torch.stack([e.state for e in experiences]).to(self.device)
        next_states = torch.stack([e.next_state for e in experiences]).to(self.device)

        actions = torch.tensor([e.action for e in experiences], device=self.device)
        rewards = torch.tensor(
            [e.reward for e in experiences], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [e.done for e in experiences], dtype=torch.bool, device=self.device
        )

        # Use policy network to get current Q-value estimates
        policy_predictions, policy_quantiles = self.policy_network.forward(
            states
        )  # (batch_size, n_tau, action_size)

        # DDQN: Use policy network to select next actions, target network for Q-values
        with torch.no_grad():

            # Use policy network to get next actions
            next_actions = self.get_best_action(self.policy_network, next_states)

            # Use target network to get next Q-values
            next_target_q, target_quantiles = self.target_network.forward(next_states)

        # Vectorized IQN loss computation
        n_policy_tau = policy_predictions.shape[1]
        n_target_tau = next_target_q.shape[1]

        # Get indexes for gathering Q-values
        policy_q_index = actions.unsqueeze(1).unsqueeze(2).expand(-1, n_policy_tau, -1)

        # Get Q-values for selected actions: (batch_size, n_policy_tau)
        policy_q_selected = policy_predictions.gather(2, policy_q_index).squeeze(2)

        # Get target Q-values for next actions: (batch_size, n_target_tau)
        with torch.no_grad():
            target_q_selected = next_target_q.gather(
                2, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, n_target_tau, -1)
            ).squeeze(2)

            # Compute target values: (batch_size, n_target_tau)
            target_values = (
                rewards.unsqueeze(1)
                + self.discount_factor
                * target_q_selected
                * (~dones).unsqueeze(1).float()
            )

        # Expand dimensions for broadcasting
        # policy_q_selected: (batch_size, n_policy_tau) -> (batch_size, n_policy_tau, 1)
        # target_values: (batch_size, n_target_tau) -> (batch_size, 1, n_target_tau)
        policy_q_expanded = policy_q_selected.unsqueeze(2)
        target_values_expanded = target_values.unsqueeze(1)

        # Compute TD errors for all combinations: (batch_size, n_policy_tau, n_target_tau)
        td_errors = target_values_expanded - policy_q_expanded

        # Fix quantile shape - policy_quantiles is (batch_size, n_tau, 1), squeeze to (batch_size, n_tau)
        policy_taus = policy_quantiles.squeeze(2)  # Remove the last dimension
        # Now expand to (batch_size, n_policy_tau, 1) for broadcasting
        tau_expanded = policy_taus.unsqueeze(2)
        tau_expanded = tau_expanded.to(self.device)

        # Quantile regression loss: |tau - I(td_error < 0)| * huber_loss(td_error)
        indicator = (
            td_errors < 0
        ).float()  # Shape: (batch_size, n_policy_tau, n_target_tau)
        indicator = indicator.to(self.device)
        quantile_weights = torch.abs(tau_expanded - indicator)

        # Huber loss for all TD errors
        loss_func = nn.SmoothL1Loss(reduction="none")
        huber_loss = loss_func(td_errors, torch.zeros_like(td_errors))

        # Combine quantile weights with huber loss
        quantile_loss = quantile_weights * huber_loss

        # Compute per-sample losses: average over target quantiles, sum over policy quantiles
        per_sample_losses = quantile_loss.mean(dim=2).sum(dim=1)  # Shape: (batch_size,)

        # Compute per-sample TD errors for prioritized replay
        per_sample_td_errors = td_errors.abs().mean(dim=(1, 2))  # Shape: (batch_size,)

        return per_sample_losses, per_sample_td_errors

    def update_target_network(self):
        self.target_network = copy.deepcopy(self.policy_network)

    def train(self) -> float | None:
        buffer_len = len(self.replay_buffer)
        if buffer_len < self.batch_size:
            return None

        if self.use_prioritized_replay:
            experiences, idxs, weights = self.get_experience()
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            experiences, _, _ = self.get_experience()
            weights = None

        per_sample_losses, td_errors = self.get_loss(experiences)

        if self.use_prioritized_replay:
            # Update priorities with TD errors
            self.replay_buffer.update_priorities(idxs, td_errors.detach().cpu().numpy())
            # Apply importance sampling weights
            loss = (per_sample_losses * weights).mean()
        else:
            loss = per_sample_losses.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    network = Network(hidden_dim=16)
    print(network.forward(torch.rand((2, 8)), 6))

    # iqn = IQN(e_start=0., e_end=0.)
    # print(iqn.get_action(torch.rand((8)), 6))
