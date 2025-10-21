"""
==============================================================================
COMPETITION TUTORIAL #1: Custom model and RL algorithm
==============================================================================

In this tutorial, we will customize the TrackMania training pipeline.

The tutorial works with the TrackMania FULL Gymnasium environment.
Please refer to the README on GitHub to set up this environment in config.json:
https://github.com/trackmania-rl/tmrl#full-environment

Note: This tutorial describes implementing and running a TrainingAgent along with an ActorModule.
It is relevant if you want to implement your own RL approaches in TrackMania.
If you plan to try non-RL approaches instead, this is also accepted in the competition:
just use the Gymnasium Full environment and do whatever you need,
then, wrap your trained policy in an ActorModule, and submit your entry :)

Copy and adapt this script to implement your own algorithm/model in TrackMania.
Then, use the script as follows:

To launch the Server, provided the script is named custom_actor_module.py, execute:
python custom_actor_module.py --server

In another terminal, launch the Trainer:
python custom_actor_module.py --trainer

And in yet another terminal, launch a RolloutWorker:
python custom_actor_module.py --worker

You can launch these in any order, but we recommend server, then trainer, then worker.
If you are running everything on the same machine, your trainer may consume all your resource,
resulting in your worker struggling to collect samples in a timely fashion.
If your worker crazily warns you about time-steps timing out, this is probably the issue.
The best way of using TMRL with TrackMania is to have your worker(s) and trainer on separate machines.
The server can run on either of these machines, or yet another machine that both can reach via network.
Achieving this is easy (and is also kind of the whole point of the TMRL framework).
Just adapt config.json (or this script) to your network configuration.
In particular, you will want to set the following in the TMRL config.json file of all your machines:

"LOCALHOST_WORKER": false,
"LOCALHOST_TRAINER": false,
"PUBLIC_IP_SERVER": "<ip.of.the.server>",
"PORT": <port of the server (usually requires port forwarding if accessed via the Internet)>,

If you are training over the Internet, please read the security instructions on the TMRL GitHub page.

IMPORTANT: Set a custom 'RUN_NAME' in config.json, otherwise this script will not work.
"""

# Let us start our tutorial by importing some useful stuff.

# The constants that are defined in config.json:
import tmrl.config.config_constants as cfg
# Useful classes:
import tmrl.config.config_objects as cfg_obj
# The utility that TMRL uses to partially instantiate classes:
from tmrl.util import partial
# The TMRL three main entities (i.e., the Trainer, the RolloutWorker and the central Server):
from tmrl.networking import Trainer, RolloutWorker, Server

# The training class that we will customize with our own training algorithm in this tutorial:
from tmrl.training_offline import TrainingOffline

# And a couple external libraries:
import numpy as np
import os


# Now, let us look into the content of config.json:

# =====================================================================
# USEFUL PARAMETERS
# =====================================================================
# You can change these parameters here directly (not recommended),
# or you can change them in the TMRL config.json file (recommended).

# Maximum number of training 'epochs':
# (training is checkpointed at the end of each 'epoch', this is also when training metrics can be logged to wandb)
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]

# Number of rounds per 'epoch':
# (training metrics are displayed in the terminal at the end of each round)
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]

# Number of training steps per round:
# (a training step is a call to the train() function that we will define later in this tutorial)
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]

# Minimum number of environment steps collected before training starts:
# (this is useful when you want to fill your replay buffer with samples from a baseline policy)
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

# Maximum training steps / environment steps ratio:
# (if training becomes faster than this ratio, it will be paused, waiting for new samples from the environment)
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]

# Number of training steps performed between broadcasts of policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]

# Number of training steps performed between retrievals of received samples to put them in the replay buffer:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# Training device (e.g., "cuda:0"):
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'

# Maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# Batch size for training:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

# Wandb credentials:
# (Change this with your own if you want to keep your training curves private)
# (Also, please use your own wandb account if you are going to log huge stuff :) )

wandb_run_id = cfg.WANDB_RUN_ID  # change this by a name of your choice for your run
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]  # name of the wandb project in which your run will appear
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # wandb account
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # wandb API key

os.environ['WANDB_API_KEY'] = wandb_key  # this line sets your wandb API key as the active key

# Number of time-steps after which episodes collected by the worker are truncated:
max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

# Networking parameters:
# (In TMRL, networking is managed by tlspyo. The following are tlspyo parameters.)
server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER  # IP of the machine running the Server (trainer point of view)
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER  # IP of the machine running the Server (worker point of view)
server_port = cfg.PORT  # port used to communicate with this machine
password = cfg.PASSWORD  # password that secures your communication
security = cfg.SECURITY  # when training over the Internet, it is safer to change this to "TLS"
# (please read the security instructions on GitHub)


# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================
# You may want to change the following in advanced applications;
# however, most competitors will not need to change this.
# If interested, read the full TMRL tutorial on GitHub.
# These parameters are to change here directly (if you want).
# (Note: The tutorial may stop working if you change these)

# Base class of the replay memory used by the trainer:
memory_base_cls = cfg_obj.MEM

# Sample compression scheme applied by the worker for this replay memory:
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR

# Sample preprocessor for data augmentation:
sample_preprocessor = None

# Path from where an offline dataset can be loaded to initialize the replay memory:
dataset_path = cfg.DATASET_PATH

# Preprocessor applied by the worker to the observations it collects:
# (Note: if your script defines the name "obs_preprocessor", we will use your preprocessor instead of the default)
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR


# =====================================================================
# COMPETITION FIXED PARAMETERS
# =====================================================================
# Competitors CANNOT change the following parameters.

# rtgym environment class (full TrackMania Gymnasium environment):
env_cls = cfg_obj.ENV_CLS

# Device used for inference on workers (change if you like but keep in mind that the competition evaluation is on CPU)
device_worker = 'cpu'


# =====================================================================
# ENVIRONMENT PARAMETERS
# =====================================================================
# You are allowed to customize these environment parameters.
# Do not change these here though, customize them in config.json.
# Your environment configuration must be part of your submission,
# e.g., the "ENV" entry of your config.json file.

# Dimensions of the TrackMania window:
window_width = cfg.WINDOW_WIDTH  # must be between 256 and 958
window_height = cfg.WINDOW_HEIGHT  # must be between 128 and 488

# Dimensions of the actual images in observations:
img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT

# Whether you are using grayscale (default) or color images:
# (Note: The tutorial will stop working if you use colors)
img_grayscale = cfg.GRAYSCALE

# Number of consecutive screenshots in each observation:
imgs_buf_len = cfg.IMG_HIST_LEN

# Number of actions in the action buffer (this is part of observations):
# (Note: The tutorial will stop working if you change this)
act_buf_len = cfg.ACT_BUF_LEN


# =====================================================================
# MEMORY CLASS
# =====================================================================
# Nothing to do here.
# This is the memory class passed to the Trainer.
# If you need a custom memory, change the relevant advanced parameters.
# Custom memories are described in the full TMRL tutorial.

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False)


# =====================================================================
# CUSTOM MODEL - IQN IMPLEMENTATION
# =====================================================================
# This implementation uses Implicit Quantile Networks (IQN) for TrackMania.
# IQN uses quantile regression to learn the full distribution of Q-values,
# which provides richer information than standard Q-learning approaches.

# We will implement IQN with a hybrid CNN/MLP model for continuous action spaces.

# Let us import the ActorModule that we are supposed to implement.
# We will use PyTorch in this tutorial.
# TMRL readily provides a PyTorch-specific subclass of ActorModule:
from tmrl.actor import TorchActorModule

# Plus a couple useful imports:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from math import floor


# In the full version of the TrackMania 2020 environment, the
# observation-space comprises a history of screenshots. Thus, we need
# Computer Vision layers such as a CNN in our model to process these.
# The observation space also comprises single floats representing speed,
# rpm and gear. We will merge these with the information contained in
# screenshots thanks to an MLP following our CNN layers.


# Here is the MLP:
def mlp(sizes, activation, output_activation=nn.Identity):
    """
    A simple MLP (MultiLayer Perceptron).

    Args:
        sizes: list of integers representing the hidden size of each layer
        activation: activation function of hidden layers
        output_activation: activation function of the last layer

    Returns:
        Our MLP in the form of a Pytorch Sequential module
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# The next utility computes the dimensionality of CNN feature maps when flattened together:
def num_flat_features(x):
    size = x.size()[1:]  # dimension 0 is the batch dimension, so it is ignored
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# The next utility computes the dimensionality of the output in a 2D CNN layer:
def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out


# Let us now define an IQN-based network for continuous action spaces:
class IQNCNN(nn.Module):
    def __init__(self, q_net, n_quantiles=8, cosine_dim=64, use_dueling=True):
        """
        IQN-based CNN model for distributional RL with continuous actions.

        This network learns a distribution over Q-values using quantile regression,
        providing richer value estimates than standard Q-learning.

        Args:
            q_net (bool): indicates whether this neural net is a critic network
            n_quantiles (int): number of quantile samples for IQN
            cosine_dim (int): dimension of cosine embedding for quantiles
            use_dueling (bool): whether to use dueling architecture
        """
        super(IQNCNN, self).__init__()

        self.q_net = q_net
        self.n_quantiles = n_quantiles
        self.cosine_dim = cosine_dim
        self.use_dueling = use_dueling

        # Convolutional layers processing screenshots:
        # The default config.json gives 4 grayscale images of 64 x 64 pixels
        self.h_out, self.w_out = img_height, img_width
        self.conv1 = nn.Conv2d(imgs_buf_len, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels

        # Dimensionality of the CNN output:
        self.flat_features = self.out_channels * self.h_out * self.w_out

        # Dimensionality of the feature input:
        float_features = 12 if self.q_net else 9
        self.feature_dim = self.flat_features + float_features

        # Feature processing layers
        self.fc_features = nn.Linear(self.feature_dim, 256)

        # Quantile embedding for IQN
        self.tau_embedding_fc = nn.Linear(cosine_dim, 256)

        # Output layers (dueling or standard)
        if use_dueling:
            self.value_fc = nn.Linear(256, 256)
            self.value_out = nn.Linear(256, 1)

            self.advantage_fc = nn.Linear(256, 256)
            # For continuous actions, we output advantage for each action dimension
            self.advantage_out = nn.Linear(256, 3 if q_net else 1)
        else:
            self.fc_out1 = nn.Linear(256, 256)
            self.fc_out2 = nn.Linear(256, 3 if q_net else 1)

    def tau_forward(self, batch_size, n_tau, device):
        """Generate quantile embeddings using cosine basis functions."""
        taus = torch.rand((batch_size, n_tau, 1), device=device)
        cosine_values = torch.arange(self.cosine_dim, device=device) * torch.pi
        cosine_values = cosine_values.unsqueeze(0).unsqueeze(0)

        embedded_taus = torch.cos(taus * cosine_values)
        tau_x = F.relu(self.tau_embedding_fc(embedded_taus))
        return tau_x, taus

    def forward(self, x, n_tau=None):
        """
        Forward pass with IQN quantile regression.

        Args:
            x (tuple): input observation tuple
            n_tau (int): number of quantile samples (defaults to self.n_quantiles)

        Returns:
            quantile values and tau samples
        """
        if n_tau is None:
            n_tau = self.n_quantiles

        if self.q_net:
            speed, gear, rpm, images, act1, act2, act = x
        else:
            speed, gear, rpm, images, act1, act2 = x

        # CNN feature extraction
        features = F.relu(self.conv1(images))
        features = F.relu(self.conv2(features))
        features = F.relu(self.conv3(features))
        features = F.relu(self.conv4(features))

        # Flatten
        flat_features = num_flat_features(features)
        features = features.view(-1, flat_features)

        # Concatenate with other features
        if self.q_net:
            features = torch.cat((speed, gear, rpm, features, act1, act2, act), -1)
        else:
            features = torch.cat((speed, gear, rpm, features, act1, act2), -1)

        # Process features
        batch_size = features.shape[0]
        state_features = F.relu(self.fc_features(features))

        # Generate quantile embeddings
        tau_features, taus = self.tau_forward(batch_size, n_tau, features.device)

        # Merge state and quantile embeddings
        state_features = state_features.unsqueeze(1)  # (batch, 1, 256)
        merged = state_features * tau_features  # (batch, n_tau, 256)

        # Output layer
        if self.use_dueling:
            # Value stream
            v = F.relu(self.value_fc(merged))
            v = self.value_out(v)  # (batch, n_tau, 1)

            # Advantage stream
            a = F.relu(self.advantage_fc(merged))
            a = self.advantage_out(a)  # (batch, n_tau, n_actions or 1)

            # Combine with dueling
            if not self.q_net:
                # For actor, we just need value estimates
                q = v
            else:
                # For critic, combine value and advantage
                a_mean = a.mean(dim=2, keepdim=True)
                q = v + (a - a_mean)
        else:
            q = F.relu(self.fc_out1(merged))
            q = self.fc_out2(q)

        return q, taus


# We can now implement the TMRL ActorModule interface that we are supposed to submit for this competition.

# During training, TMRL will regularly save our trained ActorModule in the TmrlData/weights folder.
# By default, this would be done using the torch (i.e., pickle) serializer.
# However, while saving and loading your own pickle files is fine,
# it is highly dangerous to load other people's pickle files.
# Therefore, the competition submission does not accept pickle files.
# Instead, we can submit our trained weights in the form of a human-readable JSON file.
# The ActorModule interface defines save() and load() methods that we will override with our own JSON serializer.

import json


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct


class MyActorModule(TorchActorModule):
    """
    IQN-based policy wrapped in the TMRL ActorModule class.

    This uses distributional RL (IQN) for better value estimation,
    combined with a stochastic policy for continuous action spaces.

    (Note: TorchActorModule is a subclass of ActorModule and torch.nn.Module)
    """
    def __init__(self, observation_space, action_space, n_quantiles=8):
        """
        Initialize the IQN-based actor.

        Args:
            observation_space: observation space of the Gymnasium environment
            action_space: action space of the Gymnasium environment
            n_quantiles: number of quantile samples for distributional learning
        """
        # We must call the superclass __init__:
        super().__init__(observation_space, action_space)

        # And initialize our attributes:
        dim_act = action_space.shape[0]  # dimensionality of actions (3 for TrackMania)
        act_limit = action_space.high[0]  # maximum amplitude of actions

        # IQN network for distributional value estimates
        self.net = IQNCNN(q_net=False, n_quantiles=n_quantiles)
        self.n_quantiles = n_quantiles

        # Policy head that outputs action distribution parameters
        # Takes IQN features and outputs mean and std for continuous actions
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

        # Additional layer to aggregate quantile information for action selection
        self.quantile_aggregation = nn.Linear(n_quantiles, 1)

    def save(self, path):
        """
        JSON-serialize a detached copy of the ActorModule and save it in path.

        IMPORTANT: FOR THE COMPETITION, WE ONLY ACCEPT JSON AND PYTHON FILES.
        IN PARTICULAR, WE *DO NOT* ACCEPT PICKLE FILES (such as output by torch.save()...).

        All your submitted files must be human-readable, for everyone's safety.
        Indeed, untrusted pickle files are an open door for hackers.

        Args:
            path: pathlib.Path: path to where the object will be stored.
        """
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        # torch.save(self.state_dict(), path)

    def load(self, path, device):
        """
        Load the parameters of your trained ActorModule from a JSON file.

        Adapt this method to your submission so that we can load your trained ActorModule.

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
        # (Skip the IQN quantile computation during inference for speed)
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

    # Now, the only method that all participants are required to implement is act()
    # act() is the interface for TMRL to use your ActorModule as the policy it tests in TrackMania.
    # For the evaluation, the "test" argument will be set to True.
    def act(self, obs, test=False):
        """
        Computes an action from an observation.

        This method is the one all participants must implement.
        It is the policy that TMRL will use in TrackMania to evaluate your submission.

        Args:
            obs (object): the input observation (when using TorchActorModule, this is a torch.Tensor)
            test (bool): True at test-time (e.g., during evaluation...), False otherwise

        Returns:
            act (numpy.array): the computed action, in the form of a numpy array of 3 values between -1.0 and 1.0
        """
        # Since we have already implemented our policy in the form of a neural network,
        # act() is now pretty straightforward.
        # We don't need to compute the log probabilities here (they will be for our SAC training algorithm).
        # Also note that, when using TorchActorModule, TMRL calls act() in a torch.no_grad() context.
        # Thus, you don't need to use "with torch.no_grad()" here.
        # But let us do it anyway to be extra sure, for the people using ActorModule instead of TorchActorModule.
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


# The IQN-based critic module for distributional RL:
class IQNCNNQFunction(nn.Module):
    """
    IQN-based critic module for distributional RL.

    This network learns a distribution over Q-values using quantile regression,
    providing richer value estimates than standard Q-learning.
    """
    def __init__(self, observation_space, action_space, n_quantiles=32):
        super().__init__()
        self.net = IQNCNN(q_net=True, n_quantiles=n_quantiles)
        self.n_quantiles = n_quantiles

    def forward(self, obs, act, n_tau=None):
        """
        Estimates the distribution of action-values for the (obs, act) pair.

        Unlike standard Q-learning which outputs a single value, IQN outputs
        a distribution over Q-values represented by quantiles.

        Args:
            obs: current observation
            act: tried next action
            n_tau: number of quantile samples (defaults to self.n_quantiles)

        Returns:
            quantile_values: (batch, n_tau, 1) - distribution over Q-values
            taus: (batch, n_tau, 1) - quantile locations
        """
        # Append action to observation
        x = (*obs, act)
        # Get distributional Q-values
        q_quantiles, taus = self.net(x, n_tau=n_tau if n_tau is not None else self.n_quantiles)
        return q_quantiles, taus


# Actor-critic module combining IQN-based actor and critics
class IQNCNNActorCritic(nn.Module):
    """
    IQN-based actor-critic module for distributional RL.

    Uses two parallel critics (double Q-learning) with IQN to learn
    distributional value estimates.
    """
    def __init__(self, observation_space, action_space, n_quantiles_actor=8, n_quantiles_critic=32):
        super().__init__()

        # Policy network (actor) with fewer quantiles for efficiency
        self.actor = MyActorModule(observation_space, action_space, n_quantiles=n_quantiles_actor)

        # Two value networks (critics) with more quantiles for accuracy
        self.q1 = IQNCNNQFunction(observation_space, action_space, n_quantiles=n_quantiles_critic)
        self.q2 = IQNCNNQFunction(observation_space, action_space, n_quantiles=n_quantiles_critic)


# =====================================================================
# CUSTOM TRAINING ALGORITHM
# =====================================================================
# So far, we have implemented our custom model.
# We have also wrapped it in an ActorModule, which we will train and
# submit as an entry to the TMRL competition.
# Our ActorModule will be used in Workers to collect training data.
# Our VanillaCNNActorCritic will be used in the Trainer for training
# this ActorModule. Let us now tackle the training algorithm per-se.
# In TMRL, this is done by implementing a custom TrainingAgent.

from tmrl.training import TrainingAgent

# We will also use a couple utilities, and the Adam optimizer:

from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from copy import deepcopy
import itertools
from torch.optim import Adam


# A TrainingAgent must implement two methods:
# -> train(batch): optimizes the model from a batch of RL samples
# -> get_actor(): outputs a copy of the current ActorModule
# In this tutorial, we implement the Soft Actor-Critic algorithm
# by adapting the OpenAI Spinup implementation.

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

    # no-grad copy of the model used to send the Actor weights in get_actor():
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=IQNCNNActorCritic,  # IQN-based actor-critic module
                 gamma=0.99,  # Discount factor
                 polyak=0.995,  # Exponential averaging factor for target network
                 alpha=0.2,  # Entropy coefficient
                 lr_actor=1e-3,  # Actor learning rate
                 lr_critic=1e-3,  # Critic learning rate
                 n_quantiles_policy=8,  # Number of quantiles for policy network
                 n_quantiles_target=32,  # Number of quantiles for target Q-network
                 kappa=1.0):  # Huber loss threshold for quantile regression

        # Initialize superclass
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
        self.kappa = kappa  # Huber loss threshold

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
        # quantiles: (batch, N, 1) -> (batch, N, 1, 1)
        # target: (batch, N', 1) -> (batch, 1, N', 1)
        pairwise_delta = target[:, None, :, :] - quantiles[:, :, None, :]  # (batch, N, N', 1)
        abs_pairwise_delta = torch.abs(pairwise_delta)

        # Huber loss
        huber_loss = torch.where(abs_pairwise_delta > self.kappa,
                                 self.kappa * (abs_pairwise_delta - 0.5 * self.kappa),
                                 0.5 * pairwise_delta ** 2)

        # Quantile regression weights
        # taus: (batch, N, 1) -> (batch, N, 1, 1)
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
        # q_quantiles shape: (batch, n_quantiles, 1)
        # After mean: (batch, 1)
        # After squeeze: (batch,)
        q1_pi = q1_pi_quantiles.mean(dim=1).squeeze(-1)  # (batch,)
        q2_pi = q2_pi_quantiles.mean(dim=1).squeeze(-1)  # (batch,)
        q_pi = torch.min(q1_pi, q2_pi)  # (batch,)

        # Policy loss with entropy regularization
        # Both logp_pi and q_pi should be (batch,) now
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


# Great! We are almost done.
# Now that our IQN TrainingAgent class is defined, let us partially instantiate it.
# IQN has several hyperparameters for distributional RL:
# - n_quantiles_policy: number of quantiles for policy evaluation (fewer for speed)
# - n_quantiles_target: number of quantiles for target Q-values (more for accuracy)
# - kappa: Huber loss threshold for quantile regression

# The following hyperparameters provide a good starting point for TrackMania:
training_agent_cls = partial(IQNTrainingAgent,
                             model_cls=IQNCNNActorCritic,
                             gamma=0.995,  # Discount factor
                             polyak=0.995,  # Target network update rate
                             alpha=0.01,  # Entropy coefficient
                             lr_actor=0.00001,  # Actor learning rate
                             lr_critic=0.00005,  # Critic learning rate
                             n_quantiles_policy=8,  # Quantiles for policy (efficiency)
                             n_quantiles_target=32,  # Quantiles for target (accuracy)
                             kappa=1.0)  # Huber loss threshold


# =====================================================================
# TMRL TRAINER
# =====================================================================

training_cls = partial(
    TrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device_trainer)


# =====================================================================
# RUN YOUR TRAINING PIPELINE
# =====================================================================
# The training pipeline configured in this tutorial runs with the "TM20FULL" environment.

# You can configure the "TM20FULL" environment by following the instruction on GitHub:
# https://github.com/trackmania-rl/tmrl#full-environment

# In TMRL, a training pipeline is made of
# - one Trainer (encompassing the training algorithm that we have coded in this tutorial)
# - one to several RolloutWorker(s) (encompassing our ActorModule and the Gymnasium environment of the competition)
# - one central Server (through which RolloutWorker(s) and Trainer communicate)
# Let us instantiate these via an argument that we will pass when calling this script:

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode')
    args = parser.parse_args()

    if args.trainer:
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip=server_ip_for_trainer,
                             server_port=server_port,
                             password=password,
                             security=security)
        my_trainer.run()

        # Note: if you want to log training metrics to wandb, replace my_trainer.run() with:
        # my_trainer.run_with_wandb(entity=wandb_entity,
        #                           project=wandb_project,
        #                           run_id=wandb_run_id)

    elif args.worker or args.test:
        # Partially instantiate the actor module with the n_quantiles parameter
        actor_module_cls = partial(MyActorModule, n_quantiles=8)

        rw = RolloutWorker(env_cls=env_cls,
                           actor_module_cls=actor_module_cls,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip=server_ip_for_worker,
                           server_port=server_port,
                           password=password,
                           security=security,
                           max_samples_per_episode=max_samples_per_episode,
                           obs_preprocessor=obs_preprocessor,
                           standalone=args.test)
        rw.run(test_episode_interval=10)
    elif args.server:
        import time
        serv = Server(port=server_port,
                      password=password,
                      security=security)
        while True:
            time.sleep(1.0)