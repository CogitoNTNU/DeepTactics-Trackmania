"""
Prioritized Experience Replay for TMRL.

This creates a custom memory class that adds full PER functionality to TMRL's memory system,
using the same approach as your original IQN.py implementation.
"""

import numpy as np
import torch
from tmrl.custom.custom_memories import MemoryTMFull


class SumTree:
    """
    Sum tree data structure for efficient O(log n) priority sampling.

    Based on your IQN.py implementation adapted from the original PER paper.
    """

    def __init__(self, capacity):
        """
        Initialize sum tree.

        Args:
            capacity: Maximum number of experiences
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0

    def _propagate(self, idx, change):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find sample index from cumulative priority value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return sum of all priorities."""
        return self.tree[0]

    def add(self, priority):
        """Add new experience with given priority."""
        idx = self.data_pointer + self.capacity - 1
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, idx, priority):
        """Update priority of experience at tree index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Get data index and priority from cumulative sum."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx]


class PrioritizedMemoryTMFull(MemoryTMFull):
    """
    TMRL Memory with full Prioritized Experience Replay.

    Extends MemoryTMFull with priority-based sampling, exactly like your IQN.py
    but integrated into TMRL's memory system.
    """

    def __init__(self, *args, per_alpha=0.6, per_beta=0.4, per_beta_increment=0.001, per_epsilon=1e-6, **kwargs):
        """
        Initialize prioritized memory.

        Args:
            per_alpha: Priority exponent (0=uniform, 1=full prioritization)
            per_beta: Importance sampling exponent (anneals to 1.0)
            per_beta_increment: Beta annealing rate
            per_epsilon: Small constant to ensure non-zero priorities
            *args, **kwargs: Passed to MemoryTMFull
        """
        super().__init__(*args, **kwargs)

        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment
        self.per_epsilon = per_epsilon

        # Initialize sum tree
        self.tree = SumTree(self.memory_size)
        self.max_priority = 1.0

        # Track current priorities for new experiences
        self.current_priorities = []

    def append(self, *args, **kwargs):
        """Add experience with maximum priority (updated after training)."""
        super().append(*args, **kwargs)
        # Add to sum tree with max priority
        self.tree.add(self.max_priority ** self.per_alpha)

    def sample_indices(self):
        """
        Sample batch indices using priority-based sampling.

        Overrides MemoryTMFull.sample_indices() to use PER instead of uniform sampling.

        Returns:
            indices: List of sampled indices
        """
        batch_size = self.batch_size
        indices = []
        priorities = []

        # Divide priority range into segments
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            # Sample uniformly within each segment
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            # Get index and priority from sum tree
            idx, priority = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)

        # Store for importance sampling weight computation
        self._last_priorities = np.array(priorities)
        self._last_indices = indices

        return indices

    def get_importance_weights(self):
        """
        Compute importance sampling weights for the last sampled batch.

        Returns:
            weights: Tensor of importance sampling weights
        """
        # Compute sampling probabilities
        priorities = self._last_priorities
        probs = priorities / self.tree.total()

        # Importance sampling: (N * P(i))^(-beta)
        n = len(self)
        weights = (n * probs) ** (-self.per_beta)

        # Normalize by max weight for stability
        weights = weights / weights.max()

        # Anneal beta towards 1.0
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        return torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.

        Args:
            indices: Batch indices (from sample_indices)
            td_errors: TD errors for each sample
        """
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        for idx, td_error in zip(indices, td_errors):
            # Priority = (|TD_error| + epsilon)^alpha
            priority = (abs(td_error) + self.per_epsilon) ** self.per_alpha

            # Update in sum tree (convert data index to tree index)
            tree_idx = idx + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)

            # Track max priority
            self.max_priority = max(self.max_priority, priority)

    def sample(self):
        """
        Sample a batch with priority-based sampling.

        Overrides MemoryTMFull.sample() to also return indices and weights.
        """
        # This is called by the iterator in the training loop
        indices = self.sample_indices()
        batch = [self[idx] for idx in indices]
        batch = self.collate(batch, self.device)

        # Return batch (TMRL expects just the batch from sample())
        # We'll store indices/weights as attributes for the training agent to access
        self._last_batch_indices = indices

        return batch

    def get_last_batch_info(self):
        """
        Get indices and weights for the last sampled batch.

        Returns:
            indices: Batch indices
            weights: Importance sampling weights
        """
        weights = self.get_importance_weights()
        return self._last_batch_indices, weights
