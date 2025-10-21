"""
Utilities package for TrackMania RL.
"""

import sys
from pathlib import Path

# Ensure parent directory is in path for local imports
utils_dir = Path(__file__).parent.parent
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

from utils.nn_utils import mlp, num_flat_features, conv2d_out_dims
from utils.serialization import TorchJSONEncoder, TorchJSONDecoder

__all__ = [
    'mlp',
    'num_flat_features',
    'conv2d_out_dims',
    'TorchJSONEncoder',
    'TorchJSONDecoder',
]
