import torch.nn as nn

ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'swish': nn.SiLU,
    'leaky_relu': nn.LeakyReLU
}

def get_activation(activation):
    return ACTIVATION_MAP.get(activation, nn.ReLU)()
