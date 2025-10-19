import json
import gymnasium as gym
import os
import torch
import wandb
import glob
import time
from src.learners.IQN import IQN
from src.learners.DQN import DQN

from tensordict import TensorDict
from gymnasium.wrappers import RecordVideo


def run_training(agent_type: str, enviroment_type: str, record_video: bool):
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")

    with open("config_files/params/Enviroments.json") as f:
        Enviroment_params = json.load(f).get(enviroment_type)
    with open("config_files/params/Learners.json") as f:
        Learner_params = json.load(f).get(agent_type)
    with open("config_files/params/Training.json") as f:
        Training_params = json.load(f)

    Learner_params["network_params"]["input_dim"] = Enviroment_params["input_dim"]
    Learner_params["network_params"]["output_dim"] = Enviroment_params["output_dim"]

    # Create agent with optimal parameters
    if agent_type == "DQN":
        agent = DQN(**Learner_params)
    elif agent_type == "IQN":
        agent = IQN(**Learner_params)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Print device information
    print("=" * 50)
    print(f"Training on device: {agent.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    print("=" * 50)

    wandb.login(key=WANDB_API_KEY)

    # Configure video recording based on config
    if record_video:
        env = gym.make(enviroment_type, render_mode="rgb_array")
        video_folder = f"{enviroment_type}-training"
        os.makedirs(video_folder, exist_ok=True)

        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix="eval",
            episode_trigger=lambda x: x % Training_params["record_video_frequency"]
            == 0,
        )
    else:
        env = gym.make(enviroment_type)  # No rendering for faster training
        video_folder = None

    # Create descriptive run name
    run_name = f"{agent_type}_{enviroment_type}"

    with wandb.init(project="Trackmania", name=run_name, config=agent.config) as run:
        run.watch(agent.policy_network, log="all", log_freq=100)
        run.watch(agent.target_network, log="all", log_freq=100)

        tot_reward = 0
        episode = 0
        tot_q_value = 0
        n_q_values = 0
        total_steps = 0

        observation, _ = env.reset()
        # Run for max_episodes, with a safety limit on steps per episode
        while episode < Enviroment_params["max_episodes"]:
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            action, q_value = agent.get_action(obs_tensor.unsqueeze(0))
            if q_value is not None:
                tot_q_value += q_value
                n_q_values += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_steps += 1

            experience = TensorDict(
                {
                    "observation": obs_tensor,
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "next_observation": torch.tensor(
                        next_obs, dtype=torch.float32
                    ),  # Next state
                    "done": torch.tensor(done),
                },
                batch_size=torch.Size([]),
            )

            agent.store_transition(experience)
            tot_reward += float(reward)

            loss = agent.train()

            if done:
                if n_q_values > 0:
                    avg_q_value = tot_q_value / n_q_values
                else:
                    avg_q_value = -1

                log_metrics = {
                    "episode_reward": tot_reward,
                    "loss": loss,
                    "learning_rate": agent.optimizer.param_groups[0]["lr"],
                    "q_values": avg_q_value,
                }

                # Only process videos if recording is enabled
                if record_video and video_folder:
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
                            video_path, format="mp4", caption=f"Episode {episode}"
                        )

                run.log(log_metrics, step=episode)

                episode += 1
                tot_reward = 0
                tot_q_value = 0
                n_q_values = 0

                observation, info = env.reset()
            else:
                observation = next_obs

            if total_steps % Training_params["target_network_update_frequency"] == 0:
                agent.update_target_network()

    env.close()
