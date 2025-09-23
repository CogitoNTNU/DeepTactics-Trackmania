import gymnasium as gym
from DQN import DQN, Experience
import torch
import wandb
import os

WANDB_API_KEY=os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)

env = gym.make("CartPole-v1", render_mode="human")


with wandb.init() as run:
    dqn_agent = DQN()
    run.watch(dqn_agent.policy_network, log="all", log_freq=100)
    run.watch(dqn_agent.target_network, log="all", log_freq=100)

    training_steps = 1_000_000

    observation, info = env.reset()
    best_rew = 0
    tot_reward = 0

    num_episodes = 0

    for i in range(training_steps):
        observation = torch.tensor(observation)
        action = dqn_agent.get_action(observation)

        next_obs, reward, terminated, truncated, info = env.step(action)
        if num_episodes % 10 == 0:
                env.render()

        tot_reward += reward

        done = terminated or truncated

        dqn_agent.store_transition(Experience(observation, torch.tensor(next_obs), action, done, float(reward)))

        if done:
            if num_episodes % 30 == 0:
                print("Replacing network!")
                print(f"Epsilon: {dqn_agent.eps}")
            dqn_agent.update_target_network()
            num_episodes += 1
            if num_episodes % 10 == 0:
                print(f"Got {tot_reward} on episode: {num_episodes}")
            observation, info = env.reset()
            wandb.log({"tot_reward": tot_reward,
                       "epochs": num_episodes})
            if tot_reward > best_rew:
                best_rew = tot_reward
                print(f"New best reward!: {best_rew}")
            tot_reward = 0
        
        observation = next_obs

        loss = dqn_agent.train()
        wandb.log({
                    "loss": loss,
                    "epsilon": dqn_agent.eps,
                })
        

        

    env.close()