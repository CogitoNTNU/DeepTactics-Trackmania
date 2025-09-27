import numpy as np
from typing import Tuple
import random
from src.experience import Experience

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