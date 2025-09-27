import gymnasium as gym
import os
import torch
import wandb
from DQN import DQN, Experience

env = gym.make("CartPole-v1", render_mode="rgb_array")
video_dir = "videos"
env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda e: e % 50 == 0)
dqn_agent = DQN()

wandb.login()
wandb.init(
    project="Trackmania",
    config={
        "env": "CartPole-v1",
        "batch_size": 32,
        "discount_factor": 0.99,
        "lr": 0.003,
    }
)


training_steps = 1_000_000
tot_reward = 0
episode = 0
cumulative_loss = 0.0

observation, info = env.reset()
for step in range(training_steps):
    obs_tensor = torch.tensor(observation, dtype=torch.float32)
    action = dqn_agent.get_action(obs_tensor)

    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    dqn_agent.store_transition(Experience(obs_tensor, torch.tensor(next_obs, dtype=torch.float32), action, done, float(reward)))
    tot_reward += float(reward)

    loss = dqn_agent.train()
    if loss is not None:
        cumulative_loss += loss

    if done:
        wandb.log({
            "loss": cumulative_loss/tot_reward,
            "epsilon": dqn_agent.eps,
            "learning_rate": dqn_agent.optimizer.param_groups[0]['lr']
        }, step=episode)
        wandb.log({"episode_reward": tot_reward}, step=episode)
        if episode % 50 == 0:
            video_filename = f"rl-video-episode-{episode}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            if os.path.exists(video_path):
                wandb.log({"video": wandb.Video(video_path, format="mp4")}, step=episode)

        episode += 1
        tot_reward = 0
        cumulative_loss = 0.0

        observation, info = env.reset()
        dqn_agent.update_target_network()
    else:
        observation = next_obs

env.close()