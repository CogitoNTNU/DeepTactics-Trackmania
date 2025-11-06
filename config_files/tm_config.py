from sys import platform
import numpy as np

# Conditional import for tmrl (not available on macOS)
if platform != "darwin" and platform != "linux":
    import tmrl.config.config_constants as cfg
    from src.helper_functions.tm_actions import number_of_actions


class Config_tm:
    def __init__(self):
        # =============================================================================
        # GENERAL SETTINGS
        # =============================================================================
        self.training_steps = 10_000_000 
        self.target_network_update_frequency = 1
        self.tau = 0.002
        self.record_video = True
        self.record_frequency = 20
        self.video_folder = None
        self.wang_distribution = False
        self.wang_distortion: float = -0.3
        self.run_name = "Simple_Train_action_history_4" 
        self.crash_detection = False 
        self.crash_threshold = 10.0 
        self.crash_penalty = 0.5

        # =============================================================================
        # ALGORITHM FEATURES
        # =============================================================================
        self.use_dueling = True
        self.use_prioritized_replay = True
        self.use_doubleDQN = True

        # Import obs dimensions from cfg
        self.img_x = cfg.IMG_HEIGHT
        self.img_y = cfg.IMG_WIDTH
        self.output_dim = number_of_actions
        self.conv_input = cfg.IMG_HIST_LEN
        self.input_car_dim = 3
        self.car_feature_hidden_dim = 256
        self.conv_hidden_image_variable = 4 #4 fo 64x64 images
        self.action_history_hidden_dim = 256
                
        # Checkpoint settings
        self.checkpoint = True
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_frequency = 10
        self.keep_last_n_checkpoints = 1000
        self.resume_from_checkpoint = True
        self.n_zone_centers_extrapolate_before_start_of_map = 20
        self.n_zone_centers_extrapolate_after_end_of_map = 1_000
        self.load_checkpoint = False
        self.load_checkpoint_name = "__insert_name_here__"
        
        # =============================================================================
        # HYPERPARAMETERS
        # =============================================================================
        self.n_tau_train = 8
        self.n_tau_action = 8
        self.cosine_dim = 64

        self.learning_rate_start = 0.001
        self.learning_rate_end = 0.00005
        self.cosine_annealing_decay_episodes = 5000
        self.batch_size = 32
        self.discount_factor = 0.997

        self.max_buffer_size = 100000
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001

        self.epsilon_start = 0.9
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.997
        self.epsilon_decay_episodes = 1000
        self.epsilon_cutoff_episodes = 5000

        self.hidden_dim = 128
        self.noisy_std = 0.5
        self.conv_channels_1 = 8
        self.conv_channels_2 = 16

        # Note: VIRTUAL_GAMEPAD and other top-level config settings are loaded from tmrl's config
        self.virtual_gamepad = True
        # TMRL-specific settings - accessing constants directly from cfg
        self.rtgym_interface = cfg.RTGYM_INTERFACE
        self.time_step_duration = cfg.ENV_CONFIG["RTGYM_CONFIG"]["time_step_duration"]
        self.start_obs_capture = cfg.ENV_CONFIG["RTGYM_CONFIG"]["start_obs_capture"]
        self.time_step_timeout_factor = cfg.ENV_CONFIG["RTGYM_CONFIG"]["time_step_timeout_factor"]
        self.act_buf_len = cfg.ACT_BUF_LEN
        self.benchmark = cfg.ENV_CONFIG["RTGYM_CONFIG"]["benchmark"]
        self.wait_on_done = cfg.ENV_CONFIG["RTGYM_CONFIG"]["wait_on_done"]
        self.ep_max_length = cfg.ENV_CONFIG["RTGYM_CONFIG"]["ep_max_length"]
        self.interface_kwargs = cfg.ENV_CONFIG["RTGYM_CONFIG"]["interface_kwargs"]
        
        self.window_width = cfg.WINDOW_WIDTH
        self.window_height = cfg.WINDOW_HEIGHT
        self.img_width = cfg.IMG_WIDTH
        self.img_height = cfg.IMG_HEIGHT
        self.img_grayscale = cfg.GRAYSCALE
        self.sleep_time_at_reset = cfg.SLEEP_TIME_AT_RESET
        
        # Reward configuration
        self.reward_end_of_track = cfg.REWARD_CONFIG["END_OF_TRACK"]
        self.reward_constant_penalty = cfg.REWARD_CONFIG["CONSTANT_PENALTY"]
        self.reward_check_forward = cfg.REWARD_CONFIG["CHECK_FORWARD"]
        self.reward_check_backward = cfg.REWARD_CONFIG["CHECK_BACKWARD"]
        self.reward_failure_countdown = cfg.REWARD_CONFIG["FAILURE_COUNTDOWN"]
        self.reward_min_steps = cfg.REWARD_CONFIG["MIN_STEPS"]
        self.reward_max_stray = cfg.REWARD_CONFIG["MAX_STRAY"]

    def to_dict(self):
        """Convert all config attributes to a dictionary, excluding methods."""
        return {key: value for key, value in self.__dict__.items() if not callable(value)}
