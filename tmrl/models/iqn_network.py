"""
IQN (Implicit Quantile Network) base network for distributional RL.

This network learns a distribution over Q-values using quantile regression,
providing richer value estimates than standard Q-learning.
"""

import sys
from pathlib import Path

# Ensure parent directory is in path for local imports
models_dir = Path(__file__).parent.parent
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
import tmrl.config.config_constants as cfg
from utils.nn_utils import num_flat_features, conv2d_out_dims


class IQNCNN(nn.Module):
    """
    IQN-based CNN model for distributional RL with continuous actions.

    This network learns a distribution over Q-values using quantile regression,
    providing richer value estimates than standard Q-learning.
    """

    def __init__(self, q_net, n_quantiles=8, cosine_dim=64, use_dueling=True):
        """
        Initialize IQN-based CNN model.

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

        # Image dimensions from config
        img_height = cfg.IMG_HEIGHT
        img_width = cfg.IMG_WIDTH
        imgs_buf_len = cfg.IMG_HIST_LEN

        # Convolutional layers processing screenshots
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

        # Dimensionality of the CNN output
        self.flat_features = self.out_channels * self.h_out * self.w_out

        # Dimensionality of the feature input
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
        """
        Generate quantile embeddings using cosine basis functions.

        Args:
            batch_size: size of the batch
            n_tau: number of quantile samples
            device: torch device

        Returns:
            Tuple of (tau_x, taus) - embedded quantiles and quantile fractions
        """
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
