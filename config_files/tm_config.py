from sys import platform
import numpy as np

# Conditional import for tmrl (not available on macOS)
if platform != "darwin":
    import tmrl.config.config_constants as cfg
    from src.helper_functions.tm_actions import number_of_actions


class Config:
    def __init__(self):
        # =============================================================================
        # GENERAL SETTINGS
        # =============================================================================
        self.training_steps = 1_000_000
        self.target_network_update_frequency = 500
        self.record_video = True  # Set to True to record episode videos (slows training)
        self.record_frequency = 50
        
        # Choose environment: "CarRacing-v3", "LunarLander-v3", "CartPole-v1", "TM20"
        self.env_name = "Acrobot-v1"

        # =============================================================================
        # ALGORITHM SELECTION
        # =============================================================================
        # Choose which agent to use (only one should be True)
        self.use_DQN = True    # Basic DQN agent
        self.use_IQN = False     # IQN agent (Implicit Quantile Networks)

        # =============================================================================
        # ALGORITHM FEATURES (apply to both DQN and IQN)
        # =============================================================================
        self.use_dueling = True           # Use Dueling architecture
        self.use_prioritized_replay = True   # Use Prioritized Experience Replay (PER)
        self.use_doubleDQN = True          # Use Double DQN (reduces overestimation)

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
            case "Acrobot-v1":
                self.input_dim = 6
                self.output_dim = 3
            case "MountainCar-v0":
                self.input_dim = 2
                self.output_dim = 3
            case "TM20":
                self.img_x = cfg.IMG_HEIGHT
                self.img_y = cfg.IMG_WIDTH
                self.output_dim = number_of_actions # number of actions that can be taken, kanskje bare skrive antallet her?
                self.conv_input = cfg.IMG_HIST_LEN
                self.input_car_dim = 3
                self.car_feature_hidden_dim = 256
                
                
        # Checkpoint settings
        self.checkpoint = True # Disable model saving
        self.checkpoint_dir = "checkpoints"  # Directory to save checkpoints
        self.checkpoint_frequency = 10  # Save checkpoint every N episodes
        self.keep_last_n_checkpoints = 3  # Keep only the last N periodic checkpoints (to save disk space)
        self.resume_from_checkpoint = True  # If True, will try to resume from latest checkpoint
        self.n_zone_centers_extrapolate_before_start_of_map = 20
        self.n_zone_centers_extrapolate_after_end_of_map = 1_000

        # =============================================================================
        # HYPERPARAMETERS (DQN/IQN)
        # =============================================================================
        # IQN-specific parameters
        self.n_tau_train = 8        # Number of quantiles for training
        self.n_tau_action = 8         # Number of quantiles for action selection
        self.cosine_dim = 32          # Dimension of cosine embedding for quantiles

        # Learning parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.discount_factor = 0.99

        # Replay buffer settings
        self.max_buffer_size = 100000
        self.alpha = 0.6              # PER: how much prioritization (0=uniform, 1=full prioritization)
        self.beta = 0.4               # PER: importance sampling weight (increases to 1)
        self.beta_increment = 0.001   # PER: beta increase per training step

        # Epsilon-greedy exploration parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.997

        # Network architecture
        self.hidden_dim = 128
        self.noisy_std = 0.5                # Standard deviation for NoisyLinear layers (0 = no noise/no exploration!)
        self.conv_channels_1 = 8            # First convolutional layer output channels
        self.conv_channels_2 = 16           # Second convolutional layer output channels
        