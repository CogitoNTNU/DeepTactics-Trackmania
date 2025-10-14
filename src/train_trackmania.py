"""
Simple single-process training script for Trackmania with IQN.
This is a minimal example - you'll need to adapt your IQN to work with image+float inputs.
"""

import torch
import wandb
from pathlib import Path

from src.IQN import IQN
from src.trackmania_env import TrackmaniaEnv
from tensordict import TensorDict
from config_files import tm_config


def train_trackmania():
    """
    Train IQN on Trackmania.

    NOTE: The current IQN in src/IQN.py is designed for simple vector inputs (LunarLander).
    You'll need to modify the Network class to:
    1. Add convolutional layers for processing images
    2. Add dense layers for processing float features (speed, etc.)
    3. Merge both streams before the quantile embedding

    See linesight's IQN_Network in trackmania_rl/agents/iqn.py for reference.
    """

    # Initialize wandb
    wandb_api_key = input("Enter your W&B API key (or press Enter to skip logging): ")
    use_wandb = len(wandb_api_key) > 0

    if use_wandb:
        wandb.login(key=wandb_api_key)

    # Setup paths
    base_dir = Path(__file__).resolve().parents[1]
    maps_dir = base_dir / "maps"

    # TODO: Configure these for your map
    map_path = '"ESL-Hockolicious.Challenge.Gbx"'  # Adjust this
    zone_centers_file = "ESL-Hockolicious_0.5m_cl2.npy"  # Adjust this

    # Create environment
    print("="*50)
    print("Initializing Trackmania environment...")
    print("="*50)

    env = TrackmaniaEnv(
        port=tm_config.base_tmi_port,
        map_path=map_path,
        zone_centers_file=zone_centers_file,
        maps_dir=maps_dir
    )

    # TODO: Replace this with your modified IQN that handles image inputs
    # For now, this will fail because IQN expects vector inputs, not images
    print("\n" + "="*50)
    print("WARNING: Current IQN doesn't support image inputs!")
    print("You need to modify Network class in src/IQN.py to add:")
    print("  1. Conv layers for images")
    print("  2. Dense layers for float features")
    print("  3. Merge both streams")
    print("="*50 + "\n")

    # Uncomment when you have a proper network:
    # agent = IQN()

    if use_wandb:
        run_name = f"Trackmania_IQN"
        run = wandb.init(project="DeepTactics-Trackmania", name=run_name)
        # run.watch(agent.policy_network)

    # Training loop
    try:
        print("Connecting to Trackmania...")
        obs = env.reset()
        print("Connected! Starting training...\n")

        for episode in range(100):  # Adjust number of episodes
            obs = env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done:
                # TODO: Replace with actual action selection from your network
                # For now, random actions
                action = torch.randint(0, len(tm_config.inputs), (1,)).item()

                # Step environment
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                steps += 1

                # TODO: Store transition and train
                # experience = TensorDict({
                #     "observation": ...,  # Need to handle image+float properly
                #     "action": torch.tensor(action),
                #     "reward": torch.tensor(reward),
                #     "next_observation": ...,
                #     "done": torch.tensor(done)
                # }, batch_size=torch.Size([]))
                # agent.store_transition(experience)
                # loss = agent.train()

                obs = next_obs

                if steps % 10 == 0:
                    print(f"Episode {episode}, Step {steps}, Reward: {episode_reward:.2f}")

            print(f"\nEpisode {episode} finished!")
            print(f"Total reward: {episode_reward:.2f}, Steps: {steps}")
            print(f"Final zone: {info['current_zone']}\n")

            if use_wandb:
                wandb.log({
                    "episode_reward": episode_reward,
                    "episode_steps": steps,
                    "final_zone": info['current_zone']
                }, step=episode)

    finally:
        env.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  DeepTactics Trackmania Training")
    print("="*70)
    print("\nBefore running this, make sure:")
    print("  1. TMInterface 2.x is installed")
    print("  2. Python_Link.as is in your TMInterface Plugins folder")
    print("  3. Trackmania is running with TMInterface loaded")
    print("  4. You've configured paths in config_files/tm_config.py")
    print("  5. You have map zone centers (.npy files) in the maps/ folder")
    print("\n" + "="*70 + "\n")

    input("Press Enter to continue...")

    train_trackmania()
