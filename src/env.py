import gymnasium as gym
import os
import torch
import wandb
import glob
import time
from src.IQN import IQN
from src.experience import Experience
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from config_files import tm_config

def run_training():
    WANDB_API_KEY=os.getenv("WANDB_API_KEY")

    dqn_agent = IQN()
    n_tau = 8
    env_name = "LunarLander-v3"

    wandb.login(key=WANDB_API_KEY)
    env = gym.make(env_name, render_mode="rgb_array")
    
    episode_record_frequency = 4
    video_folder = f"{env_name}-training"

    env = RecordVideo(
        env,
        video_folder=video_folder, # Folder to save videos
        name_prefix="eval", # Prefix for video filenames
        episode_trigger=lambda x: x % episode_record_frequency == 0, # Record every 'x' episode
    )

    with wandb.init(project="Trackmania") as run:
        run.watch(dqn_agent.policy_network, log="all", log_freq=100)
        run.watch(dqn_agent.target_network, log="all", log_freq=100)

        tot_reward = 0
        episode = 0
        tot_q_value = 0
        n_q_values = 0


        observation, info = env.reset()
        for i in range(tm_config.training_steps):
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            action, q_value = dqn_agent.get_action(obs_tensor.unsqueeze(0), n_tau)
            if q_value is not None:
                tot_q_value += q_value
                n_q_values += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            dqn_agent.store_transition(Experience(obs_tensor, torch.tensor(next_obs, dtype=torch.float32), action, done, float(reward)))
            tot_reward += float(reward)

            loss = dqn_agent.train()

            if done:
                if n_q_values > 0:
                    avg_q_value = tot_q_value / n_q_values
                else:
                    avg_q_value = -1
                video_path = None
                pattern = os.path.join(video_folder, "*.mp4")
                # Didn't work on max, so added a timer. 
                #TODO: look for better way to get videos - Sverre
                deadline = time.time() + 2
                while time.time() < deadline:
                    candidates = glob.glob(pattern)
                    if candidates:
                        video_path = max(candidates, key=os.path.getctime) # Gets the last created time
                log_metrics = {
                    "episode_reward": tot_reward,
                    "loss": loss, 
                    "epsilon": dqn_agent.eps,
                    "learning_rate": dqn_agent.optimizer.param_groups[0]['lr'],
                    "q_values": avg_q_value
                }

                if video_path:
                    log_metrics["episode_video"] = wandb.Video(video_path, format="mp4", caption=f"Episode {episode}")

                run.log(log_metrics, step=episode)
                
                episode += 1
                tot_reward = 0
                tot_q_value = 0
                n_q_values = 0

                observation, info = env.reset()
            else:
                observation = next_obs
                
            if i % tm_config.target_network_update_frequency == 0:
                dqn_agent.update_target_network()

    env.close()

if __name__ == "__main__":
    run_training()