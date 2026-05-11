from bases.nn.conv2d import DenseConv2d, SparseConv2d
from bases.nn.linear import DenseLinear, SparseLinear

from bases.nn.conv2d import Bias_DenseConv2d
from bases.nn.linear import Bias_DenseLinear
from bases.nn.embedding import DenseEmbedding

def is_fc(layer):
    return isinstance(layer, DenseLinear) or isinstance(layer, SparseLinear) or isinstance(layer, Bias_DenseLinear) 


def is_conv(layer):
    return isinstance(layer, DenseConv2d) or isinstance(layer, SparseConv2d) or isinstance(layer, Bias_DenseConv2d)

def is_embb(layer):
    return isinstance(layer, DenseEmbedding)


def traverse_module(module, criterion, layers: list, names: list, prefix="", leaf_only=True):
    if leaf_only:
        for key, submodule in module._modules.items():
            new_prefix = prefix
            if prefix != "":
                new_prefix += '.'
            new_prefix += key
            # is leaf and satisfies criterion
            if len(submodule._modules.keys()) == 0 and criterion(submodule):
                layers.append(submodule)
                names.append(new_prefix)
            traverse_module(submodule, criterion, layers, names, prefix=new_prefix, leaf_only=leaf_only)
    else:
        raise NotImplementedError("Supports only leaf modules")


import torch.nn as nn


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output
