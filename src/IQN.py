from git import Optional
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from config_files import tm_config

from src.experience import Experience
from src.replay_buffer import PrioritizedReplayBuffer



class Network(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=4):
        super().__init__()

        self.tau_embedding_fc1 = nn.Linear(64, hidden_dim)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        

    def tau_forward(self, batch_size, n_tau):
        taus = torch.rand((batch_size, n_tau, 1))
        print("Tau values: ", taus)
        cosine_values = torch.arange(64) * torch.pi
        cosine_values = cosine_values.unsqueeze(0).unsqueeze(0)

        # taus: (batch_size, n_tau, 1)
        # Cosine: (1, 1, 64)
        embedded_taus = torch.cos(taus * cosine_values) # dim: (n_tau, 64)
        
        # for hver [64] tau - send gjennom et linear layer (tau_embedding_fc1) - og kjør relu på output. 
        tau_x = self.tau_embedding_fc1(embedded_taus)
        tau_x = F.relu(tau_x) # tensor med shape (n_tau, hidden_dim - dette er vektor med 512 verdier, da må output fra å sende state x inn i netteverket også ha 512 verdier.)
        return tau_x

    def forward(self, x: torch.Tensor, n_tau: int):
        batch_size = x.shape[0]

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        tau_x = self.tau_forward(batch_size, n_tau)
        x = x.unsqueeze(dim=1)
        # tau_x: dim (1, n_tau, 64)
        # x: dim (batch_size, 1, hidden_dim)
        x = x * tau_x

        # dim: (n_tau, action_size)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)

        return x, tau_x

class IQN:
    def __init__(self, e_start=0.9, e_end=0.05, e_decay_rate=0.9999, batch_size=32, 
                 discount_factor=0.99, use_prioritized_replay=True, 
                 alpha=0.6, beta=0.4, beta_increment=0.001):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")
        self.device = "cpu"
        self.policy_network = Network().to(self.device)
        self.target_network = Network().to(self.device)
        
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=10000, 
                alpha=alpha, 
                beta=beta, 
                beta_increment=beta_increment
            )
        else:
            self.replay_buffer = deque(maxlen=10000)
        
        self.eps = e_start
        self.e_end = e_end
        self.e_decay_rate = e_decay_rate
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=0.003)
    
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
    
    def get_action(self, obs, n_tau) -> tuple[int, Optional[float]]:
        if self.eps > self.e_end:
            self.eps *= self.e_decay_rate
        
        if random.random() < self.eps:
            return random.randint(0, 3), None
        else:
            actions_quantiles, quantiles = self.policy_network.forward(obs.to(device=self.device), n_tau)
            # (batch_size, n_tau, action_size)
            q_values = actions_quantiles.mean(dim=1)
            print("q_values: ", q_values)
            best_action = torch.argmax(q_values, dim=1)
            return int(best_action.item()), float(q_values.max().item())
    
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
        
        states = torch.stack([e.state for e in experiences]).to(self.device)
        next_states = torch.stack([e.next_state for e in experiences]).to(self.device)
        actions = torch.tensor([e.action for e in experiences], device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.bool, device=self.device)
        
        q_values, policy_quantiles = self.policy_network(states, 8) #legg til dependency på n_tau
        policy_predictions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) # policy pred: (batch_size, n_tau, action_size)

        # DDQN som bruker policy-network sin next-action for at target-network kan predikere med den
        with torch.no_grad():
            next_policy_q, _policy_quantiles = self.policy_network(next_states, 8) #legg til dependency på n_tau
            next_actions = next_policy_q.argmax(dim=1)
            next_target_q, target_quantiles = self.target_network(next_states, 8) #legg til dependency på n_tau
            next_q_values = next_target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        targets = torch.where(dones, rewards, rewards + self.discount_factor * next_q_values)
        
        loss_func = nn.SmoothL1Loss(reduction="none")
        
        tot_loss = 0
        #room for a lot of improvement, O(n^3)-no tensors
        for batch_idx in range(policy_quantiles.shape[0]):
            curr_loss = 0
            for i in range(policy_quantiles.shape[1]):
                for j in range(target_quantiles.shape[1]):
                    action = actions[batch_idx]
                    td_error = rewards[batch_idx] + 0.99 * next_target_q[batch_idx][j][action] - policy_predictions[batch_idx][i][action]
                    loss = torch.abs(policy_quantiles[i] - (td_error < 0)) * loss_func(td_error) / 1
                    curr_loss += loss

            curr_loss /= target_quantiles.shape[1]
            tot_loss += curr_loss

        
        if self.use_prioritized_replay:
            # Update priorities in replay buffer
            self.replay_buffer.update_priorities(idxs, td_errors.detach().cpu().numpy())
            
            # Apply importance sampling weights to loss
            weighted_losses = tot_loss * weights
            loss = weighted_losses.mean()
        else:
            loss_func = nn.SmoothL1Loss()
            loss = loss_func(policy_predictions, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

if __name__ == "__main__":
    network = Network(hidden_dim=16)
    print(network.forward(torch.rand((2, 8)), 6))
    

    #iqn = IQN(e_start=0., e_end=0.)
    #print(iqn.get_action(torch.rand((8)), 6))


