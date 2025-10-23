import os
import torch
import wandb
import numpy as np
from tensordict import TensorDict
from config_files import tm_config

# Import from installed tmrl package
from tmrl import get_environment

# Import IQN from local tmrl_custom folder
from tmrl_custom.IQN import IQN

# Define discrete action space
# Gas levels: [0.0, 0.25, 0.5, 0.75, 1.0] (5 levels)
# Steering angles: [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0] (9 levels)
# Total: 5 × 9 = 45 discrete actions
GAS_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
STEERING_ANGLES = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
NUM_ACTIONS = len(GAS_LEVELS) * len(STEERING_ANGLES)

def discrete_to_continuous(action_idx):
    """
    Convert discrete action index to continuous action [gas, brake, steer].

    Args:
        action_idx: integer in range [0, NUM_ACTIONS)

    Returns:
        action: numpy array [gas, brake, steer] with values in appropriate ranges
    """
    gas_idx = action_idx // len(STEERING_ANGLES)
    steer_idx = action_idx % len(STEERING_ANGLES)

    gas = GAS_LEVELS[gas_idx]
    steer = STEERING_ANGLES[steer_idx]
    brake = 0.0  # We don't use brake in this simple discretization

    return np.array([gas, brake, steer], dtype=np.float32)

def run_training():
    WANDB_API_KEY=os.getenv("WANDB_API_KEY")

    # Initialize IQN agent with discrete action space
    # Using improved hyperparameters aligned with src/IQN.py best practices
    dqn_agent = IQN(
        num_actions=NUM_ACTIONS,
        n_tau_train=64,  # Increased for better distribution approximation
        n_tau_action=8,
        learning_rate=0.00025,  # Standard IQN learning rate
        batch_size=tm_config.batch_size,  # Use config batch size (512)
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,  # Lower final epsilon
        epsilon_decay=0.995,
    )

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

        # Images: (4, 64, 64) - normalize to [0, 1]
        img = np.array(images, dtype=np.float32) / 255.0

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

        observation, _ = env.reset()
        for i in range(tm_config.training_steps):
            # Process observation into img and float_inputs
            img, float_inputs = process_obs(observation)
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 4, 64, 64)
            float_tensor = torch.tensor(float_inputs, dtype=torch.float32).unsqueeze(0)  # (1, 9)

            # Get discrete action using epsilon-greedy
            action_idx = dqn_agent.get_action(img_tensor, float_tensor, explore=True)

            # Convert discrete action to continuous for environment
            action_continuous = discrete_to_continuous(action_idx)

            # Debug: Print actions occasionally to see what network outputs
            if i % 100 == 0:
                print(f"Step {i} [ε={dqn_agent.epsilon:.3f}]: Action {action_idx} -> gas={action_continuous[0]:.2f}, steer={action_continuous[2]:.2f}")

            next_obs, reward, terminated, truncated, _ = env.step(action_continuous)
            done = terminated or truncated

            # Process next observation
            next_img, next_float_inputs = process_obs(next_obs)

            # Store discrete action index in replay buffer
            experience = TensorDict({
                "img": torch.tensor(img, dtype=torch.float32),
                "float_inputs": torch.tensor(float_inputs, dtype=torch.float32),
                "action": torch.tensor(action_idx, dtype=torch.long),  # Store discrete action
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
                # Decay epsilon after each episode
                dqn_agent.decay_epsilon()

                log_metrics = {
                    "episode_reward": tot_reward,
                    "episode_length": step_count,
                    "loss": loss if loss is not None else 0.0,
                    "epsilon": dqn_agent.epsilon,
                    "learning_rate": dqn_agent.optimizer.param_groups[0]['lr'],
                }

                run.log(log_metrics, step=episode)

                loss_str = f"{loss:.4f}" if loss is not None else "0.0000"
                print(f"Episode {episode}: Reward={tot_reward:.2f}, Steps={step_count}, Loss={loss_str}, ε={dqn_agent.epsilon:.3f}")

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