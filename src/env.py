"""
Training script for Gymnasium environments
Supports: CarRacing-v3, LunarLander-v3, CartPole-v1, Acrobot-v1, MountainCar-v0
"""
import os
import torch
import wandb
import glob
import time
from src.agents.IQN import IQN
from src.agents.DQN import DQN
from config_files.tm_config import Config
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from tensordict import TensorDict


def run_training():
    config = Config()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)

    # Create agent based on config
    if config.use_DQN:
        agent = DQN(config)
    else:
        agent = IQN(config)

    env_kwargs = {}
    if config.env_name == "CarRacing-v3":
        env_kwargs = {
            "lap_complete_percent": 0.95,
            "domain_randomize": False,
            "continuous": False
        }

    if config.record_video:
        env_kwargs["render_mode"] = "rgb_array"
        env = gym.make(config.env_name, **env_kwargs)

        video_folder = f"videos/{config.env_name}-training"
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix="eval",
            episode_trigger=lambda x: x % config.record_frequency == 0,
        )
    else:
        env = gym.make(config.env_name, **env_kwargs)
        video_folder = None

    # Create descriptive run name
    agent_name = "DQN" if config.use_DQN else "IQN"
    features = []
    if config.use_dueling:
        features.append("Dueling")
    if config.use_prioritized_replay:
        features.append("PER")
    if config.use_doubleDQN:
        features.append("Double")

    feature_str = "+".join(features) if features else "Basic"
    run_name = f"{agent_name}_{config.env_name}_{feature_str}"

    with wandb.init(entity="cogitod", project="Trackmania", name=run_name, config=agent.config) as run:
        run.watch(agent.policy_network, log="all", log_freq=100)
        run.watch(agent.target_network, log="all", log_freq=100)

        tot_reward = 0
        episode = 0
        tot_q_value = 0
        n_q_values = 0

        observation, _ = env.reset()

        for i in range(config.training_steps):
            # Preprocess observation based on environment type
            if config.env_name == "CarRacing-v3":
                # Image observation: normalize and permute
                obs_tensor = torch.tensor(observation, dtype=torch.float32) / 255
                obs_tensor = obs_tensor.permute(2, 0, 1)  # HWC -> CHW
            else:
                # Vector observation: just convert to tensor
                obs_tensor = torch.tensor(observation, dtype=torch.float32)

            action, q_value = agent.get_action(obs_tensor.unsqueeze(0))

            if q_value is not None:
                tot_q_value += q_value
                n_q_values += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Create experience for replay buffer
            if config.env_name == "CarRacing-v3":
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32) / 255
                next_obs_tensor = next_obs_tensor.permute(2, 0, 1)
            else:
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)

            experience = TensorDict({
                "observation": obs_tensor,
                "action": torch.tensor(action),
                "reward": torch.tensor(reward),
                "next_observation": next_obs_tensor,
                "done": torch.tensor(done)
            }, batch_size=torch.Size([]))

            agent.store_transition(experience)
            tot_reward += float(reward)

            loss = agent.train()

            if done:
                avg_q_value = tot_q_value / n_q_values if n_q_values > 0 else -1

                log_metrics = {
                    "episode_reward": tot_reward,
                    "loss": loss,
                    "learning_rate": agent.optimizer.param_groups[0]['lr'],
                    "q_values": avg_q_value,
                    "epsilon": agent.epsilon
                }

                # Only process videos if recording is enabled
                if config.record_video and video_folder:
                    video_path = None
                    pattern = os.path.join(video_folder, "*.mp4")
                    deadline = time.time() + 2
                    while time.time() < deadline:
                        candidates = glob.glob(pattern)
                        if candidates:
                            video_path = max(candidates, key=os.path.getctime)
                            break

                    if video_path:
                        log_metrics["episode_video"] = wandb.Video(
                            video_path,
                            format="mp4",
                            caption=f"Episode {episode}"
                        )

                run.log(log_metrics, step=episode)

                # Only decay epsilon for agents using epsilon-greedy (DQN)
                if hasattr(agent, 'decay_epsilon'):
                    agent.decay_epsilon()
                agent.scheduler.step()

                episode += 1
                tot_reward = 0
                tot_q_value = 0
                n_q_values = 0

                observation, _ = env.reset()
            else:
                observation = next_obs

            if i % config.target_network_update_frequency == 0:
                agent.update_target_network()

    env.close()


if __name__ == "__main__":
    run_training()
