

class Config:
    def __init__(self):
        # =============================================================================
        # GENERAL SETTINGS
        # =============================================================================
        self.training_steps = 10_000_000 
        self.target_network_update_frequency = 32_000 # Use 1 with soft update of the target network
        self.tau = 1.0 # Soft update the target network. tau = 1 means hard update.
        self.record_video = True  # Set to True to record episode videos (slows training; requires display)
        self.record_frequency = 20
        self.video_folder = None

        # Choose environment: "CarRacing-v3", "LunarLander-v3", "CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Ant-v5"
        self.env_name = "LunarLander-v3"
        self.run_name = "yoo" 
        
        # =============================================================================
        # ALGORITHM SELECTION
        # =============================================================================
        # Choose which agent to use (only one should be True)
        self.use_DQN = False    # Basic DQN agent
        self.use_IQN = True     # IQN agent (Implicit Quantile Networks)

        # =============================================================================
        # ALGORITHM FEATURES (apply to both DQN and IQN)
        # =============================================================================
        self.use_dueling = False           # Use Dueling architecture
        self.use_prioritized_replay = True   # Use Prioritized Experience Replay (PER)
        self.use_doubleDQN = True          # Use Double DQN (reduces overestimation)

        
        match self.env_name:
            case "CarRacing-v3":
                self.input_dim = 3
                self.img_x = 96
                self.img_y = 96
                self.output_dim = 5
                self.conv_input = 3
                self.input_car_dim = 0
                self.car_feature_hidden_dim = 0
                self.conv_hidden_image_variable = 6  # For 96x96 images, 4 for 64x64 images
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
            case "Ant-v5":
                self.input_dim = 105  # Ant observation space with contact forces
                self.output_dim = 17  # 1 zero action + 8 joints × 2 directions
            
                
        # Checkpoint settings
        self.checkpoint = True # Disable model saving
        self.checkpoint_dir = "checkpoints"  # Directory to save checkpoints
        self.checkpoint_frequency = 10  # Save checkpoint every N episodes
        self.keep_last_n_checkpoints = 3  # Keep only the last N periodic checkpoints (to save disk space)
        self.resume_from_checkpoint = True  # If True, will try to resume from latest checkpoint
        self.n_zone_centers_extrapolate_before_start_of_map = 20
        self.n_zone_centers_extrapolate_after_end_of_map = 1_000
        self.load_checkpoint = False
        self.load_checkpoint_name = "__insert_name_here__"
        # =============================================================================
        # HYPERPARAMETERS (DQN/IQN)
        # =============================================================================
        # IQN-specific parameters
        self.n_tau_train = 8        # Number of quantiles for training
        self.n_tau_action = 8         # Number of quantiles for action selection
        self.cosine_dim = 64          # Dimension of cosine embedding for quantiles #currently cant be changed

        # Learning parameters
        self.learning_rate = 0.0001
        self.cosine_annealing_decay_episodes = 800 # Number of episodes before it uses constant learning rate
        self.batch_size = 32
        self.discount_factor = 0.997

        # Replay buffer settings
        self.max_buffer_size = 100000
        self.alpha = 0.6              # PER: how much prioritization (0=uniform, 1=full prioritization)
        self.beta = 0.4               # PER: importance sampling weight (increases to 1)
        self.beta_increment = 0.001   # PER: beta increase per training step

        # Epsilon-greedy exploration parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.997
        self.epsilon_decay_to = 2_500_000
        self.epsilon_cutoff = 25_000_000

        # Network architecture
        self.hidden_dim = 128
        self.noisy_std = 0.5                # Standard deviation for NoisyLinear layers (0 = no noise/no exploration!)
        self.conv_channels_1 = 8            # First convolutional layer output channels
        self.conv_channels_2 = 16           # Second convolutional layer output channels
        
    
    def to_dict(self):
        """Convert all config attributes to a dictionary, excluding methods."""
        return {key: value for key, value in self.__dict__.items() if not callable(value)}
