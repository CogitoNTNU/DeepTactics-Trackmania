import gymnasium as gym
import os
import torch
import wandb
import glob
import time
from src.IQN_cnn import IQN
from tensordict import TensorDict
from gymnasium.wrappers import RecordVideo
from config_files import tm_config

def run_training():
    WANDB_API_KEY=os.getenv("WANDB_API_KEY")

    # Create IQN agent with optimal parameters
    dqn_agent = IQN()

    # Print device information
    print("="*50)
    print(f"Training on device: {dqn_agent.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*50)

    #env_name = "LunarLander-v3"
    env_name = "CarRacing-v3"

    wandb.login(key=WANDB_API_KEY)

    # Configure video recording based on config
    if tm_config.record_video:
        #env = gym.make(env_name, render_mode="rgb_array")
        env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
        episode_record_frequency = 20
        video_folder = f"{env_name}-training"
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix="eval",
            episode_trigger=lambda x: x % episode_record_frequency == 0,
        )
    else:
        env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
        #env = gym.make(env_name)  # No rendering for faster training
        video_folder = None

    # Create descriptive run name
    run_name = f"IQN_ntau{dqn_agent.n_tau_train}-{dqn_agent.n_tau_action}_noisy"

    with wandb.init(project="Trackmania", name=run_name, config=dqn_agent.config) as run:
        run.watch(dqn_agent.policy_network, log="all", log_freq=100)
        run.watch(dqn_agent.target_network, log="all", log_freq=100)

        tot_reward = 0
        episode = 0
        tot_q_value = 0
        n_q_values = 0


        observation, _ = env.reset()
        for i in range(tm_config.training_steps):
            obs_tensor = torch.tensor(observation, dtype=torch.float32)/255
            obs_tensor = obs_tensor.permute(2, 0, 1)
            action, q_value = dqn_agent.get_action(obs_tensor.unsqueeze(0))
            print(f"Action: {action}")
            if q_value is not None:
                tot_q_value += q_value
                n_q_values += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            experience = TensorDict({
                "observation": obs_tensor,
                "action": torch.tensor(action),
                "reward": torch.tensor(reward),
                "next_observation": torch.tensor(next_obs, dtype=torch.float32).permute(2, 0, 1), # Next state
                "done": torch.tensor(done)
            }, batch_size=torch.Size([]))

            dqn_agent.store_transition(experience)
            tot_reward += float(reward)

            loss = dqn_agent.train()

            if done:
                if n_q_values > 0:
                    avg_q_value = tot_q_value / n_q_values
                else:
                    avg_q_value = -1

                log_metrics = {
                    "episode_reward": tot_reward,
                    "loss": loss,
                    "learning_rate": dqn_agent.optimizer.param_groups[0]['lr'],
                    "q_values": avg_q_value,
                    "epsilon": dqn_agent.epsilon
                }

                # Only process videos if recording is enabled
                if tm_config.record_video and video_folder:
                    video_path = None
                    pattern = os.path.join(video_folder, "*.mp4")
                    deadline = time.time() + 2
                    while time.time() < deadline:
                        candidates = glob.glob(pattern)
                        if candidates:
                            video_path = max(candidates, key=os.path.getctime)
                            break

                    if video_path:
                        log_metrics["episode_video"] = wandb.Video(video_path, format="mp4", caption=f"Episode {episode}")

                run.log(log_metrics, step=episode)

                # Decay epsilon after each episode

                episode += 1
                tot_reward = 0
                tot_q_value = 0
                n_q_values = 0

                observation, info = env.reset()
            else:
                observation = next_obs
            
            dqn_agent.decay_epsilon()

            if i % tm_config.target_network_update_frequency == 0:
                dqn_agent.update_target_network()

    env.close()

if __name__ == "__main__":
    run_training()