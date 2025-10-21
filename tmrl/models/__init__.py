"""
IQN Models package for TrackMania RL.
"""

from tmrl.models.iqn_network import IQNCNN
from tmrl.models.iqn_actor import MyActorModule
from tmrl.models.iqn_critic import IQNCNNQFunction
from tmrl.models.iqn_actor_critic import IQNCNNActorCritic

__all__ = [
    'IQNCNN',
    'MyActorModule',
    'IQNCNNQFunction',
    'IQNCNNActorCritic',
]
