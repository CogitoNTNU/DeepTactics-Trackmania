import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import copy
import random

@dataclass
class Experience:
    state: torch.Tensor
    next_state: Optional[torch.Tensor]
    action: int
    done: bool
    reward: float

class SumTree:
    """Sum tree data structure for efficient prioritized sampling"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample on leaf node"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get sum of all priorities"""
        return self.tree[0]
    
    def add(self, p: float, data):
        """Add new experience with priority p"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, p: float):
        """Update priority of experience at idx"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, any]:
        """Get experience with cumulative priority s"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.epsilon = 1e-6  # Small constant to avoid zero priorities
    
    def add(self, experience: Experience):
        """Add experience to buffer with maximum priority"""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[list, np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        experiences = []
        idxs = []
        priorities = []
        
        segment = self.tree.total() / batch_size
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, experience = self.tree.get(s)
            experiences.append(experience)
            idxs.append(idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        priorities = np.array(priorities)
        sampling_probabilities = priorities / self.tree.total()
        weights = (self.tree.n_entries * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        return experiences, np.array(idxs), weights
    
    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.tree.n_entries

class Network(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x

class DQN:
    def __init__(self, e_start=0.9, e_end=0.01, e_decay_rate=0.997, batch_size=32, 
                 discount_factor=0.99, use_prioritized_replay=True, 
                 alpha=0.6, beta=0.4, beta_increment=0.001):
        self.device = torch.device("mps")
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
            return random.randint(0, 1)
        else:
            actions = self.policy_network(obs.to(device=self.device))
            n = torch.argmax(actions)
            return int(n.item())
    
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
        next_q_values = self.target_network(next_states).max(1).values
        targets = torch.where(dones, rewards, rewards + self.discount_factor * next_q_values)
        
        # Calculate TD errors
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

# Usage example:
if __name__ == "__main__":
    # Create DQN with prioritized replay
    agent = DQN(use_prioritized_replay=True)
    
    # Create DQN without prioritized replay (original behavior)
    # agent = DQN(use_prioritized_replay=False)
    
    print(f"Using prioritized replay: {agent.use_prioritized_replay}")