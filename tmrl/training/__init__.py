"""
IQN Training package for TrackMania RL.
"""

import sys
from pathlib import Path

# Ensure parent directory is in path for local imports
training_dir = Path(__file__).parent.parent
if str(training_dir) not in sys.path:
    sys.path.insert(0, str(training_dir))

from training.iqn_training_agent import IQNTrainingAgent

__all__ = [
    'IQNTrainingAgent',
]
