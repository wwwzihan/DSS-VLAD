# Ablation of cnn model with different padding strategies

import torch
from torch import nn

def replace_padding_mode(model, mode='zeros'):
    """
    reference: "MIND THE PAD: CNNS CAN DEVELOP BLIND SPOTS"
    Args:
        model: nn.Module
        mode: str - 'zeros', 'reflect', 'replicate', 'circular'
    Returns:
        model with updated padding mode
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.padding_mode = mode
    return model
