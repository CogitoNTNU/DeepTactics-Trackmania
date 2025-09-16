import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import random

class Network(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim = 128, output_dim = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x

class DQN():
    def __init__(self, e_start = 0.9, e_end = 0.05, e_decay_rate = 0.99):
        self.policy_network = Network()
        self.target_network = Network()
        self.replay_buffer = []
        self.eps = e_start
        self.e_end = e_end
        self.e_decay_rate = e_decay_rate

    def get_action(self, obs) -> int:
        if random.random() > self.eps:
            return random.randint(0, 1)
        else:
            n = torch.argmax(self.policy_network(obs))
            print(n)
            print(type(n.item()))
            return n.item()

    def copy_network(self, network: Network) -> Network:
        return copy.deepcopy(network)
