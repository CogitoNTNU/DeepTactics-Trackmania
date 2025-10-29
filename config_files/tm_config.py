from sys import platform
import numpy as np
import tmrl.config as cfg
from src.helper_functions.tm_actions import number_of_actions


class Config:
    def __init__(self):
        self.training_steps = 1_000_000
        self.target_network_update_frequency = 1000
        self.use_dueling = True
        self.record_video = False  # Set to True to record episode videos (slows training)

        # Choose between "CarRacing-v3", "LunarLander-v3", "CartPole-v1", "TM20"
        # carracing-v3 might need its own env file
        self.env_name = "CarRacing-v3"
        match self.env_name:
            case "CarRacing-v3":
                self.input_dim = 3
                self.img_x = 96
                self.img_y = 96
                self.output_dim = 5 #3 if discrete, 5 if discrete
                self.conv_input = 3 
            case "LunarLander-v3":
                self.input_dim = 8
                self.output_dim = 4
            case "CartPole-v1":
                self.input_dim = 4
                self.output_dim = 2
            case "TM20":
                self.img_x = cfg.IMG_HEIGHT
                self.img_y = cfg.IMG_WIDTH
                self.output_dim = number_of_actions # number of actions that can be taken, kanskje bare skrive antallet her?
                self.conv_input = cfg.IMG_HIST_LEN
                self.input_car_dim = 3
                self.car_feature_hidden_dim = 256
                
        self.run_name = "Run_name_for_wandb" # Run name for wandb

        # Checkpoint settings
        self.checkpoint = True # Disable model saving
        self.checkpoint_dir = "checkpoints"  # Directory to save checkpoints
        self.checkpoint_frequency = 10  # Save checkpoint every N episodes
        self.keep_last_n_checkpoints = 3  # Keep only the last N periodic checkpoints (to save disk space)
        self.resume_from_checkpoint = True  # If True, will try to resume from latest checkpoint
        self.n_zone_centers_extrapolate_before_start_of_map = 20
        self.n_zone_centers_extrapolate_after_end_of_map = 1_000

        # Config parameters for DQN:
        # Config parameters for rainbow/iqn
        #Hyper parameters
        self.n_tau_train=64
        self.n_tau_action=64
        self.cosine_dim=32
        self.learning_rate=0.00025
        self.batch_size=64
        self.discount_factor=0.99
        #buffer settings
        self.max_buffer_size = 10000
        self.use_prioritized_replay=True
        self.alpha=0.6
        self.beta=0.4
        self.beta_increment=0.001
        # Epsilon-greedy exploration parameters
        self.epsilon_start=1.0
        self.epsilon_end=0.01
        self.epsilon_decay=0.997
        #Connv dimensions
        self.hidden_dim = 128
        self.cosine_dim = 32
        self.noisy_std = 0.5
        self.hidden_dim1 = 8
        self.hidden_dim2 = 16
        