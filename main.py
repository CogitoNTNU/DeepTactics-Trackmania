from config_files.tm_config import Config

config = Config()

if config.env_name == "CarRacing-v3" or config.env_name == "TM20":
    from src.env_tm import run_training 
else:
    from src.env import run_training

if __name__ == "__main__":
    run_training()