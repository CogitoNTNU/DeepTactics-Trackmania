"""
IQN-based Actor Module for TrackMania.

This module implements the policy network that selects actions based on
distributional value estimates from IQN.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from tmrl.actor import TorchActorModule
from tmrl.models.iqn_network import IQNCNN
from tmrl.utils.serialization import TorchJSONEncoder, TorchJSONDecoder


class MyActorModule(TorchActorModule):
    """
    IQN-based policy wrapped in the TMRL ActorModule class.

    This uses distributional RL (IQN) for better value estimation,
    combined with a stochastic policy for continuous action spaces.
    """

    def __init__(self, observation_space, action_space, n_quantiles=8):
        """
        Initialize the IQN-based actor.

        Args:
            observation_space: observation space of the Gymnasium environment
            action_space: action space of the Gymnasium environment
            n_quantiles: number of quantile samples for distributional learning
        """
        super().__init__(observation_space, action_space)

        dim_act = action_space.shape[0]  # dimensionality of actions (3 for TrackMania)
        act_limit = action_space.high[0]  # maximum amplitude of actions

        # IQN network for distributional value estimates
        self.net = IQNCNN(q_net=False, n_quantiles=n_quantiles)
        self.n_quantiles = n_quantiles

        # Policy head that outputs action distribution parameters
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

        # Additional layer to aggregate quantile information for action selection
        self.quantile_aggregation = nn.Linear(n_quantiles, 1)

    def save(self, path):
        """
        JSON-serialize a detached copy of the ActorModule and save it in path.

        IMPORTANT: FOR THE COMPETITION, WE ONLY ACCEPT JSON AND PYTHON FILES.
        IN PARTICULAR, WE *DO NOT* ACCEPT PICKLE FILES.

        Args:
            path: pathlib.Path: path to where the object will be stored.
        """
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)

    def load(self, path, device):
        """
        Load the parameters of your trained ActorModule from a JSON file.

        Args:
            path: pathlib.Path: full path of the JSON file
            device: str: device on which the ActorModule should live (e.g., "cpu")

        Returns:
            The loaded ActorModule instance
        """
        import os
        self.device = device

        # Check if the file exists
        if not os.path.exists(path):
            # No saved model, just return initialized model
            self.to_device(device)
            return self

        try:
            # Try loading as JSON first (our new format)
            with open(path, 'r', encoding='utf-8') as json_file:
                state_dict = json.load(json_file, cls=TorchJSONDecoder)
            self.load_state_dict(state_dict)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to PyTorch format (old saved models)
            try:
                state_dict = torch.load(path, map_location=device)
                self.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Could not load model from {path}: {e}")
                print("Starting with randomly initialized weights.")

        self.to_device(device)
        return self

    def forward(self, obs, test=False, compute_logprob=True):
        """
        Computes the output action using IQN's distributional value estimates.

        OPTIMIZED for real-time inference by avoiding redundant CNN passes.

        Args:
            obs: the observation from the Gymnasium environment
            test (bool): True for test episodes (deterministic), False for training (stochastic)
            compute_logprob (bool): whether to compute log probabilities for training

        Returns:
            the action sampled from our policy
            the log probability of this action (for training)
        """
        # Decompose observation
        speed, gear, rpm, images, act1, act2 = obs

        # Extract CNN features ONCE (this is the expensive part)
        features = F.relu(self.net.conv1(images))
        features = F.relu(self.net.conv2(features))
        features = F.relu(self.net.conv3(features))
        features = F.relu(self.net.conv4(features))
        features = features.view(features.size(0), -1)

        # Concatenate with other features
        features = torch.cat((speed, gear, rpm, features, act1, act2), -1)
        net_features = F.relu(self.net.fc_features(features))

        # Compute action distribution parameters directly from features
        mu = self.mu_layer(net_features)
        log_std = self.log_std_layer(net_features)
        log_std = torch.clamp(log_std, -20, 2)  # Clamp for stability
        std = torch.exp(log_std)

        # Sample action from the distribution
        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu  # Deterministic at test time
        else:
            pi_action = pi_distribution.rsample()  # Stochastic during training

        # Compute log probabilities if needed (only during training)
        if compute_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # Correction for tanh squashing
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # Squash action to valid range
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        """
        Computes an action from an observation.

        This method is the one all participants must implement.
        It is the policy that TMRL will use in TrackMania to evaluate your submission.

        Args:
            obs (object): the input observation
            test (bool): True at test-time, False otherwise

        Returns:
            act (numpy.array): the computed action (3 values between -1.0 and 1.0)
        """
        try:
            import time
            start_time = time.time()

            with torch.no_grad():
                a, _ = self.forward(obs=obs, test=test, compute_logprob=False)
                action = a.cpu().numpy()

            inference_time = time.time() - start_time

            # Debug: print action and timing occasionally
            if not hasattr(self, '_action_count'):
                self._action_count = 0
                self._total_time = 0.0
            self._action_count += 1
            self._total_time += inference_time

            if self._action_count % 100 == 1:  # Print every 100 actions
                avg_time = self._total_time / self._action_count
                print(f"[IQN Debug] Action {self._action_count}: {action}")
                print(f"[IQN Timing] Inference: {inference_time*1000:.2f}ms | Avg: {avg_time*1000:.2f}ms")

            return action
        except Exception as e:
            print(f"[IQN ERROR in act()]: {e}")
            import traceback
            traceback.print_exc()
            # Return safe default action
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
