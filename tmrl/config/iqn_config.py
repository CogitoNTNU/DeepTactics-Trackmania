"""
IQN Configuration for TrackMania RL Training.

This module contains all configuration parameters and setup for the IQN training pipeline.
"""

import os
import sys
from pathlib import Path

# Ensure parent directory is in path for local imports
config_dir = Path(__file__).parent.parent
if str(config_dir) not in sys.path:
    sys.path.insert(0, str(config_dir))

from tmrl.util import partial

import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj

from models.iqn_actor_critic import IQNCNNActorCritic
from training.iqn_training_agent import IQNTrainingAgent
from training.prioritized_memory import PrioritizedMemoryTMFull
from training.per_training_offline import PERTrainingOffline


# =====================================================================
# USEFUL PARAMETERS
# =====================================================================
# These are loaded from config.json

# Training epochs and steps
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# Device configuration
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'
device_worker = 'cpu'

# Memory configuration
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

# Wandb configuration
wandb_run_id = cfg.WANDB_RUN_ID
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]
os.environ['WANDB_API_KEY'] = wandb_key

# Worker configuration
max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

# Networking parameters
server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER
server_port = cfg.PORT
password = cfg.PASSWORD
security = cfg.SECURITY

# Environment configuration
window_width = cfg.WINDOW_WIDTH
window_height = cfg.WINDOW_HEIGHT
img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT
img_grayscale = cfg.GRAYSCALE
imgs_buf_len = cfg.IMG_HIST_LEN
act_buf_len = cfg.ACT_BUF_LEN

# =====================================================================
# KEYBOARD INPUT OVERRIDE
# =====================================================================
# Override the VIRTUAL_GAMEPAD setting to use keyboard input instead
# must also change it ibn the tmrl config.json

use_keyboard_input = False  # Set to False to use gamepad


# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================

memory_base_cls = cfg_obj.MEM
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR
sample_preprocessor = None
dataset_path = cfg.DATASET_PATH
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR


# =====================================================================
# COMPETITION FIXED PARAMETERS
# =====================================================================

env_cls = cfg_obj.ENV_CLS


# =====================================================================
# MEMORY CLASS WITH PRIORITIZED EXPERIENCE REPLAY
# =====================================================================

# Use PrioritizedMemoryTMFull instead of the base class for full PER
memory_cls = partial(PrioritizedMemoryTMFull,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False,
                     # PER parameters (same as training agent)
                     per_alpha=0.6,
                     per_beta=0.4,
                     per_beta_increment=0.001,
                     per_epsilon=1e-6)


# =====================================================================
# IQN TRAINING AGENT
# =====================================================================
# Configure the IQN training agent with hyperparameters

training_agent_cls = partial(IQNTrainingAgent,
                             model_cls=IQNCNNActorCritic,
                             gamma=0.99,  # Match original IQN.py
                             polyak=0.995,
                             alpha=0.01,  # SAC entropy coefficient (different from PER alpha)
                             lr_actor=0.00025,  # Match original IQN.py learning_rate
                             lr_critic=0.00025,  # Match original IQN.py learning_rate
                             n_quantiles_policy=64,  # Match original n_tau_action
                             n_quantiles_target=64,  # Match original n_tau_train
                             kappa=1.0,
                             # PER parameters (matching your original IQN.py)
                             per_alpha=0.6,  # Priority exponent
                             per_beta=0.4,  # IS exponent (starts low, anneals to 1.0)
                             per_beta_increment=0.0000006,  # Reach 1.0 after ~1M steps
                             per_epsilon=1e-6)


# =====================================================================
# TMRL TRAINER WITH PER
# =====================================================================

training_cls = partial(
    PERTrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device_trainer)
