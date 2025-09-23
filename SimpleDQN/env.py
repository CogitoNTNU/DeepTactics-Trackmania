import gymnasium as gym
from DQN import DQN, Experience
import torch

env = gym.make("CartPole-v1", render_mode="rgb_array")

dqn_agent = DQN()

training_steps = 1_000_000


observation, info = env.reset()
best_rew = 0
tot_reward = 0

for i in range(training_steps):
    observation = torch.tensor(observation)
    action = dqn_agent.get_action(observation)

    next_obs, reward, terminated, truncated, info = env.step(action)

    tot_reward += reward

    done = terminated or truncated

    dqn_agent.store_transition(Experience(observation, torch.tensor(next_obs), action, done, float(reward)))

    if done:
        observation, info = env.reset()
        if tot_reward > best_rew:
            best_rew = tot_reward
            print(best_rew)
        tot_reward = 0
    
    observation = next_obs

    dqn_agent.train()

    if i % 1_000 == 0:
        print("Replacing network!")
        print(f"Epsilon: {dqn_agent.eps}")
        dqn_agent.update_target_network()
    

env.close()