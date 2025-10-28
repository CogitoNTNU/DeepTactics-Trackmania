from sys import platform
import numpy as np

class Config:
    def __init__(self):
        self.training_steps = 1_000_000
        self.target_network_update_frequency = 1000
        self.use_dueling = False
        self.record_video = False  # Set to True to record episode videos (slows training)

        # Choose between "CarRacing-v3",   
        self.env_name = "CarRacing-v3"
        match self.env_name:
            case "CarRacing-v3":
                self.input_dim = 3
                self.output_dim = 13
            case "LunarLander-v3":
                self.input_dim = 8
                self.output_dim = 4
            case "CartPole-v1":
                self.input_dim = 4
                self.output_dim = 2



            case "TM20":
                
        self.run_name = "Run_name_for_wandb" # Run name for wandb

        # Checkpoint settings
        self.checkpoint = False # Disable model saving
        self.checkpoint_dir = "checkpoints"  # Directory to save checkpoints
        self.checkpoint_frequency = 10  # Save checkpoint every N episodes
        self.keep_last_n_checkpoints = 3  # Keep only the last N periodic checkpoints (to save disk space)
        self.resume_from_checkpoint = True  # If True, will try to resume from latest checkpoint
        self.n_zone_centers_extrapolate_before_start_of_map = 20
        self.n_zone_centers_extrapolate_after_end_of_map = 1_000

        # Config parameters for DQN:
