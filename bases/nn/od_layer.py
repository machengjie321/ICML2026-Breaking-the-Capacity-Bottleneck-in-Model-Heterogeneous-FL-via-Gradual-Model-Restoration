import numpy as np
import torch.nn.init as init
import math
import torch.nn.functional as F
import torch
import torch.nn as nn

from typing import Literal, Optional
from typing import Any, Literal
from bases.nn.models.fiarse import Bern


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
        in_dim = x.size(-1) # second dimension is input dimension
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






class ODEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim,
                 padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False,
                 norm=False, **kwargs):
        """
        norm: 与 ODLinear 一致，若为 True，则输出维度对齐到 4 的倍数
        其他参数与 nn.Embedding 相同
        """
        super().__init__(
            num_embeddings, embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            **kwargs
        )

        self.norm = norm  # 是否对齐到 4 的倍数

    def forward(self, input, p=None):
        weight = self.weight  # [num_embeddings, embedding_dim]
        full_emb_dim = weight.size(1)

        # ===== p=None 或 p=0：不裁剪 =====
        if not p:
            emb_dim = full_emb_dim
        else:
            # 和 ODLinear 一样的裁剪规则
            raw = int(np.ceil(full_emb_dim * (p ** 0.5)))
            if self.norm:
                # 对齐到 4 的倍数
                emb_dim = max(1, int(raw / 4) * 4)
            else:
                emb_dim = max(1, raw)

            # 防止越界
            emb_dim = min(emb_dim, full_emb_dim)

        # ===== 裁剪列方向的维度 =====
        weight_red = weight[:, :emb_dim]  # [num_embeddings, emb_dim]

        # ===== 调用 F.embedding =====
        return F.embedding(
            input,
            weight_red,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
