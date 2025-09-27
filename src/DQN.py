import numpy as np
from collections import deque
import torch
import torch.nn as nn
import copy
import random

from src.experience import Experience
from src.replay_buffer import PrioritizedReplayBuffer



class Network(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        v = self.value(x)
        a = self.advantage(x)
        
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)
        return q

class DQN:
    def __init__(self, e_start=0.9, e_end=0.05, e_decay_rate=0.9999, batch_size=32, 
                 discount_factor=0.99, use_prioritized_replay=True, 
                 alpha=0.6, beta=0.4, beta_increment=0.001):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")
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
    
    def get_action(self, obs) -> int:
        if self.eps > self.e_end:
            self.eps *= self.e_decay_rate
        
        if random.random() < self.eps:
            return random.randint(0, 1), None
        else:
            actions = self.policy_network(obs.to(device=self.device))
            n = torch.argmax(actions)
            return int(n.item()), int(actions.max().item())
    
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
        
        q_values = self.policy_network(states)
        policy_predictions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN som bruker policy-network sin next-action for at target-network kan predikere med den
        with torch.no_grad():
            next_policy_q = self.policy_network(next_states)
            next_actions = next_policy_q.argmax(dim=1)
            next_target_q = self.target_network(next_states)
            next_q_values = next_target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        targets = torch.where(dones, rewards, rewards + self.discount_factor * next_q_values)
        
        # Calculate TD errors (Absolutt-verdi og stabiliserer?)
        td_errors = policy_predictions - targets
        
        if self.use_prioritized_replay:
            # Update priorities in replay buffer
            self.replay_buffer.update_priorities(idxs, td_errors.detach().cpu().numpy())
            
            # Apply importance sampling weights to loss
            loss_func = nn.SmoothL1Loss(reduction='none')
            losses = loss_func(policy_predictions, targets)
            weighted_losses = losses * weights
            loss = weighted_losses.mean()
        else:
            loss_func = nn.SmoothL1Loss()
            loss = loss_func(policy_predictions, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()