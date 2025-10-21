"""
IQN Models package for TrackMania RL.
"""

import sys
from pathlib import Path

# Ensure parent directory is in path for local imports
models_dir = Path(__file__).parent.parent
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

from models.iqn_network import IQNCNN
from models.iqn_actor import MyActorModule
from models.iqn_critic import IQNCNNQFunction
from models.iqn_actor_critic import IQNCNNActorCritic

__all__ = [
    'IQNCNN',
    'MyActorModule',
    'IQNCNNQFunction',
    'IQNCNNActorCritic',
]
