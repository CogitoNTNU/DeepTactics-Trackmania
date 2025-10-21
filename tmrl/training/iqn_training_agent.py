"""
IQN Training Agent for TrackMania.

Implements the training algorithm using Implicit Quantile Networks (IQN)
for distributional reinforcement learning.
"""

import sys
from pathlib import Path

# Ensure parent directory is in path for local imports
training_dir = Path(__file__).parent.parent
if str(training_dir) not in sys.path:
    sys.path.insert(0, str(training_dir))

import torch
import itertools
from copy import deepcopy
from torch.optim import Adam

from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from models.iqn_actor_critic import IQNCNNActorCritic


class IQNTrainingAgent(TrainingAgent):
    """
    IQN-based training algorithm for distributional RL.

    This implementation uses Implicit Quantile Networks (IQN) with quantile regression
    to learn distributions over Q-values, combined with an actor-critic architecture
    for continuous control.

    Custom TrainingAgents implement two methods: train(batch) and get_actor().

    Required arguments:
    - observation_space: observation space
    - action_space: action space
    - device: training device (e.g., "cpu" or "cuda:0")
    """

    # no-grad copy of the model used to send the Actor weights in get_actor()
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=IQNCNNActorCritic,
                 gamma=0.99,
                 polyak=0.995,
                 alpha=0.2,
                 lr_actor=1e-3,
                 lr_critic=1e-3,
                 n_quantiles_policy=8,
                 n_quantiles_target=32,
                 kappa=1.0):
        """
        Initialize the IQN training agent.

        Args:
            observation_space: observation space
            action_space: action space
            device: training device
            model_cls: actor-critic model class
            gamma: discount factor
            polyak: exponential averaging factor for target network
            alpha: entropy coefficient
            lr_actor: actor learning rate
            lr_critic: critic learning rate
            n_quantiles_policy: number of quantiles for policy network
            n_quantiles_target: number of quantiles for target Q-network
            kappa: Huber loss threshold for quantile regression
        """
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        # Initialize IQN-based model
        model = model_cls(observation_space, action_space,
                         n_quantiles_actor=n_quantiles_policy,
                         n_quantiles_critic=n_quantiles_target)
        self.model = model.to(self.device)
        self.model_target = no_grad(deepcopy(self.model))

        # Hyperparameters
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.n_quantiles_policy = n_quantiles_policy
        self.n_quantiles_target = n_quantiles_target
        self.kappa = kappa

        # Optimizers
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)
        self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        """
        Returns a copy of the current ActorModule.

        We return a copy without gradients, as this is for sending to the RolloutWorkers.

        Returns:
            actor: ActorModule: updated actor module to forward to the worker(s)
        """
        return self.model_nograd.actor

    def quantile_huber_loss(self, quantiles, target, taus):
        """
        Compute the quantile Huber loss for IQN.

        This is the core loss function for IQN, which combines Huber loss with
        quantile regression to learn the full distribution of Q-values.

        Args:
            quantiles: (batch, N, 1) - predicted quantile values
            target: (batch, N', 1) - target quantile values
            taus: (batch, N, 1) - quantile fractions

        Returns:
            loss: scalar tensor - quantile Huber loss
        """
        # Expand dimensions for pairwise differences
        pairwise_delta = target[:, None, :, :] - quantiles[:, :, None, :]  # (batch, N, N', 1)
        abs_pairwise_delta = torch.abs(pairwise_delta)

        # Huber loss
        huber_loss = torch.where(abs_pairwise_delta > self.kappa,
                                 self.kappa * (abs_pairwise_delta - 0.5 * self.kappa),
                                 0.5 * pairwise_delta ** 2)

        # Quantile regression weights
        tau_hat = taus[:, :, None, :]
        indicator = (pairwise_delta.detach() < 0).float()
        quantile_weight = torch.abs(tau_hat - indicator)

        # Final loss
        loss = (quantile_weight * huber_loss).mean()
        return loss

    def train(self, batch):
        """
        Training step using IQN's quantile regression.

        This method implements distributional RL by learning the full distribution
        of Q-values rather than just their expected values.

        Args:
            batch: (o, a, r, o2, d, _) - RL transition batch
                o: initial observation
                a: action taken
                r: reward received
                o2: next observation
                d: done signal
                _: truncated signal (ignored)

        Returns:
            logs: dictionary of training metrics
        """
        # Decompose batch
        o, a, r, o2, d, _ = batch

        # Sample action from current policy
        pi, logp_pi = self.model.actor(obs=o, test=False, compute_logprob=True)

        # Get distributional Q-values for current state-action
        q1_quantiles, q1_taus = self.model.q1(o, a, n_tau=self.n_quantiles_policy)
        q2_quantiles, q2_taus = self.model.q2(o, a, n_tau=self.n_quantiles_policy)

        # Compute target Q distribution
        with torch.no_grad():
            # Sample next action from policy
            a2, logp_a2 = self.model.actor(o2)

            # Get target Q distributions
            q1_target_quantiles, q1_target_taus = self.model_target.q1(o2, a2, n_tau=self.n_quantiles_target)
            q2_target_quantiles, q2_target_taus = self.model_target.q2(o2, a2, n_tau=self.n_quantiles_target)

            # Take minimum for double Q-learning
            q_target_quantiles = torch.min(q1_target_quantiles, q2_target_quantiles)

            # Expand reward and done for broadcasting
            r = r.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
            d = d.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
            logp_a2 = logp_a2.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)

            # Compute Bellman target with entropy
            target_quantiles = r + self.gamma * (1 - d) * (q_target_quantiles - self.alpha_t * logp_a2)

        # Compute quantile regression losses
        loss_q1 = self.quantile_huber_loss(q1_quantiles, target_quantiles.detach(), q1_taus)
        loss_q2 = self.quantile_huber_loss(q2_quantiles, target_quantiles.detach(), q2_taus)
        loss_q = loss_q1 + loss_q2

        # Optimize critics
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze critics for policy update
        for p in self.q_params:
            p.requires_grad = False

        # Get Q-values for sampled actions
        q1_pi_quantiles, _ = self.model.q1(o, pi, n_tau=self.n_quantiles_policy)
        q2_pi_quantiles, _ = self.model.q2(o, pi, n_tau=self.n_quantiles_policy)

        # Use mean of quantiles as Q-value estimate
        q1_pi = q1_pi_quantiles.mean(dim=1).squeeze(-1)  # (batch,)
        q2_pi = q2_pi_quantiles.mean(dim=1).squeeze(-1)  # (batch,)
        q_pi = torch.min(q1_pi, q2_pi)  # (batch,)

        # Policy loss with entropy regularization
        q_pi_mean = q_pi.mean(dim=1) 
        loss_pi = (self.alpha_t * logp_pi - q_pi).mean()

        # Optimize actor
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze critics
        for p in self.q_params:
            p.requires_grad = True

        # Update target network with polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # Log metrics
        ret_dict = dict(
            loss_actor=loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
            loss_q1=loss_q1.detach().item(),
            loss_q2=loss_q2.detach().item(),
            q_mean=q_pi.mean().detach().item(),
        )
        return ret_dict
