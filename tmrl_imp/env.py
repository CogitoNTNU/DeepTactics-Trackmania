import os
import torch
import wandb
import numpy as np
from src.IQN import IQN
from tensordict import TensorDict
from config_files import tm_config
from tmrl import get_environment

def run_training():
    WANDB_API_KEY=os.getenv("WANDB_API_KEY")

    # Create IQN agent with optimal parameters for TrackMania
    # TrackMania IMAGE observations: (4, 64, 64) grayscale images + float features
    # Float features: speed(1) + gear(1) + rpm(1) + prev_act1(3) + prev_act2(3) = 9
    # Actions: [gas, brake, steer] - 3 continuous values in [-1, 1]
    dqn_agent = IQN(
        n_tau_train=8,  # Match Linesight (must be even for symmetric sampling)
        n_tau_action=32,  # Match Linesight (must be even)
        cosine_dim=64,  # Match Linesight IQN embedding
        learning_rate=0.00025,
        batch_size=64,
        discount_factor=0.99,
        use_prioritized_replay=True,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
    )

    # Update network for TrackMania image-based observations (matching Linesight architecture)
    from src.IQN import Network
    dqn_agent.policy_network = Network(
        img_channels=4,  # 4 frame history
        img_height=64,
        img_width=64,
        float_inputs_dim=9,  # speed, gear, rpm, 2 prev actions (3 each)
        action_dim=3,
        iqn_embedding_dim=64,
        float_hidden_dim=256,
        dense_hidden_dim=1024
    ).to(dqn_agent.device)

    dqn_agent.target_network = Network(
        img_channels=4,
        img_height=64,
        img_width=64,
        float_inputs_dim=9,
        action_dim=3,
        iqn_embedding_dim=64,
        float_hidden_dim=256,
        dense_hidden_dim=1024
    ).to(dqn_agent.device)

    dqn_agent.optimizer = torch.optim.AdamW(dqn_agent.policy_network.parameters(), lr=0.00025)

    # Print device information
    print("="*50)
    print(f"Training on device: {dqn_agent.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*50)

    env_name = "TrackMania"
    wandb.login(key=WANDB_API_KEY)

    # Get TrackMania environment from TMRL
    # Configuration comes from config.json
    env = get_environment()
    print(f"TrackMania environment loaded successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Helper function to process TrackMania image observations
    def process_obs(obs):
        """
        Process TrackMania IMAGE observation.
        obs format: (speed, gear, rpm, images, act1, act2)

        Returns:
            img: (img_channels, img_height, img_width) - image tensor
            float_inputs: (float_inputs_dim,) - concatenated float features
        """
        speed, gear, rpm, images, act1, act2 = obs

        # Images: (4, 64, 64) already in correct format
        img = np.array(images, dtype=np.float32)

        # Float inputs: concatenate speed, gear, rpm, act1, act2
        float_inputs = np.concatenate([
            np.array(speed).flatten(),  # (1,)
            np.array(gear).flatten(),   # (1,)
            np.array(rpm).flatten(),    # (1,)
            np.array(act1).flatten(),   # (3,)
            np.array(act2).flatten(),   # (3,)
        ], dtype=np.float32)  # Total: (9,)

        return img, float_inputs

    # Create descriptive run name
    run_name = f"IQN_TrackMania_continuous"

    with wandb.init(project="Trackmania", name=run_name, config=dqn_agent.config) as run:
        run.watch(dqn_agent.policy_network, log="all", log_freq=100)
        run.watch(dqn_agent.target_network, log="all", log_freq=100)

        tot_reward = 0
        episode = 0
        step_count = 0
        exploration_episodes = 50  # Use random exploration for first 50 episodes

        observation, _ = env.reset()
        for i in range(tm_config.training_steps):
            # Process observation into img and float_inputs
            img, float_inputs = process_obs(observation)
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 4, 64, 64)
            float_tensor = torch.tensor(float_inputs, dtype=torch.float32).unsqueeze(0)  # (1, 9)

            # Get continuous action [gas, brake, steer]
            if episode < exploration_episodes:
                # Random exploration: mostly forward with some steering
                action = np.array([
                    np.random.uniform(0.5, 1.0),   # Gas: 50-100%
                    0.0,                            # No brake
                    np.random.uniform(-0.5, 0.5)   # Steer: random
                ], dtype=np.float32)
            else:
                action = dqn_agent.get_action(img_tensor, float_tensor)

            # Debug: Print actions occasionally to see what network outputs
            if i % 20 == 0:
                mode = "RANDOM" if episode < exploration_episodes else "NETWORK"
                print(f"Step {i} [{mode}]: Action = {action} (gas={action[0]:.3f}, brake={action[1]:.3f}, steer={action[2]:.3f})")

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Process next observation
            next_img, next_float_inputs = process_obs(next_obs)

            experience = TensorDict({
                "img": torch.tensor(img, dtype=torch.float32),
                "float_inputs": torch.tensor(float_inputs, dtype=torch.float32),
                "action": torch.tensor(action, dtype=torch.float32),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_img": torch.tensor(next_img, dtype=torch.float32),
                "next_float_inputs": torch.tensor(next_float_inputs, dtype=torch.float32),
                "done": torch.tensor(done)
            }, batch_size=torch.Size([]))

            dqn_agent.store_transition(experience)
            tot_reward += float(reward)
            step_count += 1

            loss = dqn_agent.train()

            if done:
                log_metrics = {
                    "episode_reward": tot_reward,
                    "episode_length": step_count,
                    "loss": loss if loss is not None else 0.0,
                    "learning_rate": dqn_agent.optimizer.param_groups[0]['lr'],
                }

                run.log(log_metrics, step=episode)

                loss_str = f"{loss:.4f}" if loss is not None else "0.0000"
                print(f"Episode {episode}: Reward={tot_reward:.2f}, Steps={step_count}, Loss={loss_str}")

                episode += 1
                tot_reward = 0
                step_count = 0

                observation, info = env.reset()
            else:
                observation = next_obs

            if i % tm_config.target_network_update_frequency == 0:
                dqn_agent.update_target_network()

    env.unwrapped.wait()  # TMRL-specific method to pause environment

if __name__ == "__main__":
    run_training()