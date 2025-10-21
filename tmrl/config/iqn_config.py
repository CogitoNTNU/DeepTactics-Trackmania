"""
IQN Configuration for TrackMania RL Training.

This module contains all configuration parameters and setup for the IQN training pipeline.
"""

import os
from tmrl.util import partial
from tmrl.training_offline import TrainingOffline

import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj

from tmrl.models.iqn_actor_critic import IQNCNNActorCritic
from tmrl.training.iqn_training_agent import IQNTrainingAgent


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
# MEMORY CLASS
# =====================================================================

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False)


# =====================================================================
# IQN TRAINING AGENT
# =====================================================================
# Configure the IQN training agent with hyperparameters

training_agent_cls = partial(IQNTrainingAgent,
                             model_cls=IQNCNNActorCritic,
                             gamma=0.995,
                             polyak=0.995,
                             alpha=0.01,
                             lr_actor=0.00001,
                             lr_critic=0.00005,
                             n_quantiles_policy=8,
                             n_quantiles_target=32,
                             kappa=1.0)


# =====================================================================
# TMRL TRAINER
# =====================================================================

training_cls = partial(
    TrainingOffline,
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
