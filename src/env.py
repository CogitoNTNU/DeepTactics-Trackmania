import gymnasium as gym
import os
import torch
import wandb
from src.DQN import DQN
from src.experience import Experience
import os
from config_files import tm_config

def run_training():
    WANDB_API_KEY=os.getenv("WANDB_API_KEY")

    dqn_agent = DQN()

    wandb.login(key=WANDB_API_KEY)
    env = gym.make("CartPole-v1")


    with wandb.init(project="Trackmania") as run:
        dqn_agent = DQN()
        run.watch(dqn_agent.policy_network, log="all", log_freq=100)
        run.watch(dqn_agent.target_network, log="all", log_freq=100)

        tot_reward = 0
        episode = 0

        observation, info = env.reset()
        for i in range(tm_config.training_steps):
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            action = dqn_agent.get_action(obs_tensor)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if i % 50:
                env.render()

            dqn_agent.store_transition(Experience(obs_tensor, torch.tensor(next_obs, dtype=torch.float32), action, done, float(reward)))
            tot_reward += float(reward)

            loss = dqn_agent.train()

            if done:
                run.log({
                    "episode_reward": tot_reward,
                    "loss": loss, 
                    "epsilon": dqn_agent.eps,
                    "learning_rate": dqn_agent.optimizer.param_groups[0]['lr'] 
                }, step=episode)
                episode += 1
                tot_reward = 0

                observation, info = env.reset()
                dqn_agent.update_target_network()
            else:
                observation = next_obs

    env.close()