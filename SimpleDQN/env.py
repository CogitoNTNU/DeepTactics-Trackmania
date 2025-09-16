import gymnasium as gym
from DQN import DQN
import torch

env = gym.make('CartPole-v1', render_mode='rgb_array')

dqn_agent = DQN()

steps = 100


for _ in range(steps):
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        observation = torch.tensor(observation)
        action = dqn_agent.get_action(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    
    # Train policy network
    # Update target network

env.close()