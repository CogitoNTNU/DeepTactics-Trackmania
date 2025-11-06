import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This code is copied from https://github.com/VIPTankz/BTR/blob/main/networks.py
"""

class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func, activation=nn.ReLU):
        super().__init__()

        self.activation = activation()

        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_ = self.conv_0(self.activation(x))
        x_ = self.conv_1(self.activation(x_))
        return x + x_


class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func=torch.nn.utils.spectral_norm, activation=nn.ReLU, layer_norm=False,
                 layer_norm_shapes=False):
        super().__init__()
        self.layer_norm = layer_norm

        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)

        if self.layer_norm:
            self.norm_layer1 = nn.LayerNorm(layer_norm_shapes[0])

        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation)

    def forward(self, x):
        x = self.conv(x)

        if self.layer_norm:
            x = self.norm_layer1(x)

        x = self.max_pool(x)

        x = self.residual_0(x)

        x = self.residual_1(x)

        return x