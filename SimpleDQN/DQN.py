from collections import deque
from dataclasses import dataclass
import sys
from typing import Optional
import torch
import torch.nn as nn

import copy
import random

@dataclass
class Experience:
    state: torch.Tensor
    next_state: Optional[torch.Tensor]
    action: int
    done: bool
    reward: float

class Network(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=2):
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

class DQN:
    def __init__(self, e_start=0.9, e_end=0.05, e_decay_rate=0.9999, batch_size=32, discount_factor = 0.99):
        self.device = torch.device("mps")
        self.policy_network = Network().to(device=torch.device("mps"))
        self.target_network = Network().to(device=torch.device("mps"))
        self.replay_buffer = []
        self.eps = e_start
        self.e_end = e_end
        self.e_decay_rate = e_decay_rate
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        self.replay_buffer = deque(maxlen=10_000)
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=0.003)

    def store_transition(self, transition: Experience):
        self.replay_buffer.append(transition)

    def get_experience(self):
        return [random.choice(self.replay_buffer) for _ in range(self.batch_size)]

    def get_action(self, obs) -> int:
        if self.eps > self.e_end:
            self.eps *= self.e_decay_rate
            
        if random.random() > self.eps:
            return random.randint(0, 1)
        else:
            n = torch.argmax(self.policy_network(obs.to(device=self.device)))
            return int(n.item())

    def update_target_network(self):
        self.target_network = copy.deepcopy(self.policy_network)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        experience = self.get_experience()

        states = []
        next_states = []
        dones = []
        actions = []
        rewards = []

        for e in experience:
            states.append(e.state)
            next_states.append(e.next_state)
            dones.append(e.done)
            actions.append(e.action)
            rewards.append(e.reward)
        
        policy_predictions = self.policy_network.forward(torch.stack(states).to(device=self.device)).max(1).values


        all_state_tensor = torch.stack(next_states)
        next_state_predictions = self.target_network.forward(all_state_tensor.to(device=self.device))

        targets = []
        
        for i in range(self.batch_size):
            if dones[i]:
                targets.append(torch.tensor(-10.0, device=self.device))
            else:
                q_target = rewards[i] + self.discount_factor * next_state_predictions[i].max(0).values
                targets.append(q_target)

        targets = torch.stack(targets)

        loss_func = nn.SmoothL1Loss().to(device=torch.device("mps"))

        loss = loss_func(policy_predictions.to(device=torch.device("mps")), targets.to(device=torch.device("mps")))
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()