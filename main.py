from config_files.tm_config import Config

config = Config()

match config.env_name:
    case "CarRacing-v3":
        from src.env_tm import run_training 
    case "LunarLander-v3":
        from src.env import run_training
    case "CartPole-v1":
        from src.env import run_training
    case "TM20":
        from src.env_tm import run_training



if __name__ == "__main__":
    run_training()