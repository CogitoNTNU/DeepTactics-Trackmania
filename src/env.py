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
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    with wandb.init(project="Trackmania") as run:
        dqn_agent = DQN()
        run.watch(dqn_agent.policy_network, log="all", log_freq=100)
        run.watch(dqn_agent.target_network, log="all", log_freq=100)

        tot_reward = 0
        episode = 0
        tot_q_value = 0
        n_q_values = 0


        observation, info = env.reset()
        print(observation)
        for i in range(tm_config.training_steps):
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            action, q_value = dqn_agent.get_action(obs_tensor.unsqueeze(0))
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
                run.log({
                    "episode_reward": tot_reward,
                    "loss": loss, 
                    "epsilon": dqn_agent.eps,
                    "learning_rate": dqn_agent.optimizer.param_groups[0]['lr'],
                    "q_values": avg_q_value
                }, step=episode)
                
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