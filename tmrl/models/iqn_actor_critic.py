"""
IQN-based Actor-Critic Module for TrackMania.

Combines the actor and critic networks for distributional RL training.
"""

import sys
from pathlib import Path

# Ensure parent directory is in path for local imports
models_dir = Path(__file__).parent.parent
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

import torch.nn as nn
from models.iqn_actor import MyActorModule
from models.iqn_critic import IQNCNNQFunction


class IQNCNNActorCritic(nn.Module):
    """
    IQN-based actor-critic module for distributional RL.

    Uses two parallel critics (double Q-learning) with IQN to learn
    distributional value estimates.
    """

    def __init__(self, observation_space, action_space, n_quantiles_actor=8, n_quantiles_critic=32):
        """
        Initialize the actor-critic module.

        Args:
            observation_space: observation space of the Gymnasium environment
            action_space: action space of the Gymnasium environment
            n_quantiles_actor: number of quantiles for the actor (fewer for efficiency)
            n_quantiles_critic: number of quantiles for the critics (more for accuracy)
        """
        super().__init__()

        # Policy network (actor) with fewer quantiles for efficiency
        self.actor = MyActorModule(observation_space, action_space, n_quantiles=n_quantiles_actor)

        # Two value networks (critics) with more quantiles for accuracy
        self.q1 = IQNCNNQFunction(observation_space, action_space, n_quantiles=n_quantiles_critic)
        self.q2 = IQNCNNQFunction(observation_space, action_space, n_quantiles=n_quantiles_critic)
