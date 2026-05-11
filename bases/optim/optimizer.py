import torch
import torch.optim as optim
from torch.onnx.symbolic_opset9 import prim_type
from torch.optim.optimizer import Optimizer
from collections import OrderedDict
from typing import Union, List
#
#
class SGD(optim.SGD):
    def __init__(self, params, name_params = None, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False,lr_dict=None):
        super(SGD, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        self.lr_dict = lr_dict
        self.name_params = name_params

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("closure not supported")

        list_grad = []
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            if self.name_params is None:

                for p in group['params']:
                    if (p.grad is None and not hasattr(p, "is_sparse_param")) or hasattr(p, "is_placeholder"):
                        # exclude 1) dense param with None grad and 2) dense placeholders for sparse params
                        continue
                    elif hasattr(p, "is_sparse_param"):
                        d_p = p.dense.grad.masked_select(p.mask)
                        if weight_decay != 0:
                            d_p = d_p.add(p._values(), alpha=weight_decay)
                        if momentum != 0:
                            param_state = self.state[p]
                            if 'momentum_buffer' not in param_state:
                                buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                            else:
                                buf = param_state['momentum_buffer']
                                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                            if nesterov:
                                d_p = d_p.add(buf, alpha=momentum)
                            else:
                                d_p = buf
                            p._values().add_(d_p, alpha=-group['lr'])

                    else:
                        d_p = p.grad
                        if weight_decay != 0:
                            d_p = d_p.add(1, alpha=weight_decay)
                        if momentum != 0:
                            param_state = self.state[p]
                            if 'momentum_buffer' not in param_state:
                                buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                            else:
                                buf = param_state['momentum_buffer']
                                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                            if nesterov:
                                d_p = d_p.add(buf, alpha=momentum)
                            else:
                                d_p = buf
                        p.add_(d_p, alpha=-group['lr'])
                    list_grad.append(d_p.clone())
            else:
                for name, p in zip(self.name_params, group['params']):
                    if (p.grad is None and not hasattr(p, "is_sparse_param")) or hasattr(p, "is_placeholder"):
                        # exclude 1) dense param with None grad and 2) dense placeholders for sparse params
                        continue
                    elif hasattr(p, "is_sparse_param"):
                        d_p = p.dense.grad.masked_select(p.mask)
                        if weight_decay != 0:
                            d_p = d_p.add(p._values(), alpha=weight_decay)
                        if momentum != 0:
                            param_state = self.state[p]
                            if 'momentum_buffer' not in param_state:
                                buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                            else:
                                buf = param_state['momentum_buffer']
                                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                            if nesterov:
                                d_p = d_p.add(buf, alpha=momentum)
                            else:
                                d_p = buf

                        if name in self.lr_dict.keys():
                            p._values().add_(d_p * (-group['lr'] * self.lr_dict[name] * 1 - group['lr']))
                        else:
                            p._values().add_(d_p, alpha=-group['lr'])

                    else:
                        d_p = p.grad
                        if weight_decay != 0:
                            d_p = d_p.add(1, alpha=weight_decay)
                        if momentum != 0:
                            param_state = self.state[p]
                            if 'momentum_buffer' not in param_state:
                                buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                            else:
                                buf = param_state['momentum_buffer']
                                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                            if nesterov:
                                d_p = d_p.add(buf, alpha=momentum)
                            else:
                                d_p = buf
                        if name in self.lr_dict.keys():
                            p.add_(d_p * (-group['lr'] * self.lr_dict[name] * 3 - group['lr']))
                        else:
                            p.add_(d_p, alpha=-group['lr'])
                    list_grad.append(d_p.clone())
        return list_grad

    def clear_state(self):
        for state in self.state.values():
            if "momentum_buffer" in state:
                del state["momentum_buffer"]


import torch
from torch.optim import Optimizer


class MaskedSGD(Optimizer):
    def __init__(self, named_params, lr=1e-2, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        """
        自定义 MaskedSGD 优化器
        :param params: 可以是 model.named_parameters() 或 model.parameters()
        :param lr: 学习率
        :param momentum: 动量因子
        :param dampening: 动量抑制因子
        :param weight_decay: 权重衰减因子
        :param nesterov: 是否使用 Nesterov 加速
        :param masks: 掩码字典，key 为参数名，value 为对应的 mask tensor
        """
        self.name_params = []  # 存储参数名称
        param_groups = []  # 存储实际的参数对象
        # 如果 params 是生成器，先将其转为列表
        params = []

        for name, param in named_params:
            if param.requires_grad:
                self.name_params.append(name)
                params.append(param)

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(MaskedSGD, self).__init__(params, defaults)

        # 存储 masks，如果传入的话
        self.masks = None

        if nesterov and (momentum <= 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

    @torch.no_grad()
    def step(self, closure=None, masks = None):
        """
        执行一次参数更新。若使用 closure，可通过多次 forward-backward 来重新计算梯度。
        """
        self.masks = masks


        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历每一个 param_group
        for group_idx, group in enumerate(self.param_groups):

            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            # 遍历该参数组下的所有参数
            for name, p in zip(self.name_params, group['params']):

                if p.grad is None:
                    continue
                d_p = p.grad

                # ----------------------
                # 1) Weight Decay
                # ----------------------
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # ----------------------
                # 2) Momentum Buffer
                # ----------------------
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        # Nesterov: d_p = d_p + momentum * buf
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # ----------------------
                # 3) 应用 mask（核心逻辑）
                # ----------------------
                # 如果 self.masks 是 None，表示不做 mask；否则需要获取相应 mask
                mask = None
                if self.masks is not None:

                    # 假设用户传进来的是一个 dict，以 param 为 key
                    # 或者是一个 list，与 param_group / param_idx 对应
                    if isinstance(self.masks, dict):

                        if name in self.masks.keys():

                            mask = self.masks[name]
                            num_ones = mask.sum().item()


                    # 若拿到了 mask，就对梯度做 element-wise 乘
                    if mask is not None:

                        d_p.mul_(mask)

                # ----------------------
                # 4) 最终更新参数
                # ----------------------
                p.add_(d_p, alpha=-lr)
                # print(p,d_p)


            
        return loss
