import argparse
from src.env import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training with specified parameters.")
    parser.add_argument("--agent_type", type=str, default="IQN",required=False, help="Type of RL agent (DQN/IQN)")
    parser.add_argument("--environment_type", type=str, default="cart_pole",required=False, help="Type of environment (cart_pole/lunar_lander-v3)")
    parser.add_argument("--record_video", type=bool, default=False, required=False, help="Whether to record video")

    args = parser.parse_args()
    run_training(args.agent_type, args.environment_type, args.record_video)