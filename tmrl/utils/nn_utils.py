"""
Neural network utility functions for IQN implementation.
"""

import torch.nn as nn
from math import floor


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


def num_flat_features(x):
    """
    Computes the dimensionality of CNN feature maps when flattened together.

    Args:
        x: tensor with shape (batch, channels, height, width)

    Returns:
        Number of features when flattened
    """
    size = x.size()[1:]  # dimension 0 is the batch dimension, so it is ignored
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def conv2d_out_dims(conv_layer, h_in, w_in):
    """
    Computes the dimensionality of the output in a 2D CNN layer.

    Args:
        conv_layer: Conv2d layer
        h_in: input height
        w_in: input width

    Returns:
        Tuple of (h_out, w_out)
    """
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out
