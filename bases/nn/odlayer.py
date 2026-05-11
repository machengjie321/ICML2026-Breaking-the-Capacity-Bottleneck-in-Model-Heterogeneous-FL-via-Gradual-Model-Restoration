import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import math

__all__ = ["ODLinear", "ODConv1d", "ODConv2d", "ODConv3d", "ODWrapper","ODGroupNorm"]


class ODLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, norm=False, **kwargs):
        # ✅ 提前取出初始化参数，不传给 super().__init__()
        self.init_kwargs = {}  # 用于存储初始化参数
        for key in ['a', 'mode', 'nonlinearity']:
            if key in kwargs:
                self.init_kwargs[key] = kwargs.pop(key)

        self.norm = norm  # 自定义字段

        # ✅ 其余 kwargs 是合法的 Linear 构造参数，如 device、dtype
        super().__init__(in_features, out_features, bias=bias, **kwargs)

        # ✅ 传入自定义初始化参数
        self.reset_parameters(**self.init_kwargs)

    def reset_parameters(self, **kwargs):
        # ✅ 没有指定参数，使用默认初始化（等价于 Linear 源码）
        if len(kwargs) == 0:
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            # ✅ 使用用户提供的参数
            init.kaiming_uniform_(self.weight, **kwargs)

        # ✅ 初始化 bias（不受 init.kwargs 影响）
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, p=None):
        in_dim = x.size(1)  # second dimension is input dimension
        if not p:  # i.e., don't apply OD
            out_dim = self.out_features
        else:
            if self.norm:
                out_dim = int(int(np.ceil(self.out_features * p ** 0.5))/4)*4
            else:
                out_dim = int(np.ceil(self.out_features * p ** 0.5))
        # subsampled weights and bias
        weights_red = self.weight[:out_dim, :in_dim]
        bias_red = self.bias[:out_dim] if self.bias is not None else None
        return F.linear(x, weights_red, bias_red)
    
    


def od_conv_forward(layer, x, p=None,norm = False):
    in_dim = x.size(1)  # second dimension is input dimension
    if not p:  # i.e., don't apply OD
        out_dim = layer.out_channels
    else:
        if norm:
            out_dim = int(int(np.ceil(layer.out_channels * p ** 0.5))/4)*4
        else:
            out_dim = int(np.ceil(layer.out_channels * p ** 0.5))
                

    # subsampled weights and bias
    weights_red = layer.weight[:out_dim, :in_dim]
    bias_red = layer.bias[:out_dim] if layer.bias is not None else None
    return layer._conv_forward(x, weights_red, bias_red)


class ODConv1d(nn.Conv1d):
    def __init__(self, *args,norm = False, **kwargs):
        super(ODConv1d, self).__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, x, p=None):
        return od_conv_forward(self, x, p, self.norm)


class ODConv2d(nn.Conv2d):
    def __init__(self, *args, norm = False, **kwargs):
        super(ODConv2d, self).__init__(*args, **kwargs)
        self.norm = norm
        
    def forward(self, x, p=None):
        return od_conv_forward(self, x, p, self.norm)


class ODConv3d(nn.Conv3d):
    def __init__(self, *args,norm = False, **kwargs):
        super(ODConv3d, self).__init__(*args, **kwargs, )
        self.norm = norm
        
    def forward(self, x, p=None):
        return od_conv_forward(self, x, p, self.norm)
    
class ODWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, p=None):
        return self.layer(x, p=p)
    
    
    
    
import torch.nn.functional as F
import torch
class ODGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        c = x.size(1)  # current channel

        weight = self.weight[:c]
        bias = self.bias[:c]
        return F.group_norm(x, num_groups=self.num_groups, weight=weight, bias=bias, eps=self.eps)
