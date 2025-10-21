"""
IQN-based Critic Module for TrackMania.

This module implements the Q-function network that estimates action-values
using distributional RL.
"""

import torch.nn as nn
from tmrl.models.iqn_network import IQNCNN


class IQNCNNQFunction(nn.Module):
    """
    IQN-based critic module for distributional RL.

    This network learns a distribution over Q-values using quantile regression,
    providing richer value estimates than standard Q-learning.
    """

    def __init__(self, observation_space, action_space, n_quantiles=32):
        """
        Initialize the IQN-based critic.

        Args:
            observation_space: observation space of the Gymnasium environment
            action_space: action space of the Gymnasium environment
            n_quantiles: number of quantile samples for distributional learning
        """
        super().__init__()
        self.net = IQNCNN(q_net=True, n_quantiles=n_quantiles)
        self.n_quantiles = n_quantiles

    def forward(self, obs, act, n_tau=None):
        """
        Estimates the distribution of action-values for the (obs, act) pair.

        Unlike standard Q-learning which outputs a single value, IQN outputs
        a distribution over Q-values represented by quantiles.

        Args:
            obs: current observation
            act: tried next action
            n_tau: number of quantile samples (defaults to self.n_quantiles)

        Returns:
            quantile_values: (batch, n_tau, 1) - distribution over Q-values
            taus: (batch, n_tau, 1) - quantile locations
        """
        # Append action to observation
        x = (*obs, act)
        # Get distributional Q-values
        q_quantiles, taus = self.net(x, n_tau=n_tau if n_tau is not None else self.n_quantiles)
        return q_quantiles, taus
