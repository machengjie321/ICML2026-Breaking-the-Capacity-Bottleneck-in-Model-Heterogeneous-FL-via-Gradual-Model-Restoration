from abc import ABC, abstractmethod
from typing import Union, Sized
import torch.nn.functional as F
import torch
from torch import nn as nn

from .utils import traverse_module


class BaseModel(nn.Module, ABC):
    def __init__(self, loss_func, dict_module: dict):
        super(BaseModel, self).__init__()

        for module_name, module in dict_module.items():
            self.add_module(module_name, module)

        self.loss_func = loss_func
        self.param_layers: list = []
        self.param_layer_prefixes: list = []
        self.prunable_layers: list = []
        self.prunable_layer_prefixes: list = []

        self.collect_layers()


    def traverse(self, criterion, layers: list, names: list):
        traverse_module(self, criterion, layers, names)

    def get_param_layers(self, layers: list, names: list, criterion=None):
        self.traverse(lambda x: len(list(x.parameters())) != 0 and str(x)[0:5] != 'Batch' and not isinstance(x, nn.LayerNorm) and not isinstance(x, nn.Linear), layers, names)

    @abstractmethod
    def collect_layers(self):
        pass
    
    def set_threshold(self, thr_arg):
        prunable_layers = self.prunable_layers
        # print(thr_arg)
        if isinstance(thr_arg, (list, tuple)):
            assert len(prunable_layers) == len(thr_arg)
        else:
            thr_arg = [thr_arg] * len(prunable_layers)
        for thr, layer in zip(thr_arg, prunable_layers):
            if thr is not None:
                layer.set_threshold(thr)
                
    @abstractmethod
    def forward(self, inputs) -> torch.Tensor:
        pass

    def loss(self, inputs, labels: torch.IntTensor) -> torch.FloatTensor:
        return self.loss_func(self(inputs), labels)

    @torch.no_grad()
    def evaluate2(self, all_data, all_labels, batch_size=200):
        self.eval()  # 确保模型在评估模式
        PAD_ID = 0
        if 'transformer' in list(self.state_dict().keys())[0]:
            total_loss, total_tokens, total_correct_top5 = 0.0, 0, 0
            
            device = all_data.device
            n_samples = all_data.size(0)

            for i in range(0, n_samples, batch_size):
                # 获取当前 Batch [B, T]
                batch = all_data[i : i + batch_size] 
                
                # 1. 构造输入：取前 T-1 个 (Shift-Right)
                inputs = batch[:, :-1] 
                
                # 2. 前向传播
                # 假设输出 shape 是 [B, Vocab, S]
                logits = self(inputs) 
                
                # 3. 动态对齐 Target (最关键的一步)
                # 既然报错里 logits 展平后是 9728 (512 * 19)
                # 而 mask 是 10240 (512 * 20)
                # 说明我们需要根据 logits 的实际长度 S 来截取 target
                S = logits.size(-1) 
                target = batch[:, 1 : 1 + S] # 从第1位开始截取与 logits 等长的部分作为目标
                
                B, V, S_out = logits.shape
                
                # 4. 展平
                # logits_flat: [B*S, V]
                logits_flat = logits.permute(0, 2, 1).reshape(-1, V)
                # target_flat: [B*S]
                target_flat = target.reshape(-1)

                # 5. 此时 logits_flat.size(0) 一定等于 target_flat.size(0)
                mask = (target_flat != PAD_ID)
                
                valid_logits = logits_flat[mask]
                valid_target = target_flat[mask]
                
                if valid_target.numel() == 0:
                    continue

                # 6. 计算指标
                loss = F.cross_entropy(valid_logits, valid_target, reduction='sum')
                
                # Top-5 Accuracy
                top5 = valid_logits.topk(5, dim=1).indices
                correct_top5 = (top5 == valid_target.unsqueeze(1)).any(dim=1).sum().item()

                total_loss += loss.item()
                total_tokens += valid_target.numel()
                total_correct_top5 += correct_top5

            self.train()
            if total_tokens == 0: return 0.0, 0.0
            return total_loss / total_tokens, total_correct_top5 / total_tokens
        else:
        
            total_loss = 0.0
            n_correct = 0
            n_total = all_labels.size(0)
            
            # 按 batch_size 分批处理
            for i in range(0, n_total, batch_size):
                # 1. 截取当前批次数据
                batch_x = all_data[i : i + batch_size]
                batch_y = all_labels[i : i + batch_size]
                
                # 2. 前向传播
                outputs = self(batch_x)
                
                # 3. 计算本批次损失 (这里假设你的 loss_func 返回的是 Mean)
                # 为了后续求总平均，我们先乘上当前 batch 的大小
                batch_loss = self.loss_func(outputs, batch_y)
                total_loss += batch_loss.item() * batch_x.size(0)

                # 4. 预测与准确率统计
                labels_predicted = torch.argmax(outputs, dim=1)
                
                # 处理 one-hot 类型的 label (如果是 Transformer 任务通常不需要这步，保留你的逻辑)
                curr_y = batch_y
                if curr_y.dim() == 2:
                    curr_y = torch.argmax(curr_y, dim=1)
                    
                n_correct += torch.sum(torch.eq(labels_predicted, curr_y)).item()

            # 5. 计算整体平均值
            test_loss = total_loss / n_total
            test_acc = n_correct / n_total

            self.train()  # 恢复训练模式
            return test_loss, test_acc

    @torch.no_grad()
    def evaluate(self, test_loader, mode="sum"):

        assert mode in ["sum", "mean"], "mode must be sum or mean"
        self.eval()
        device = next(self.parameters()).device

        if 'transformer' in list(self.state_dict().keys())[0]:
            total_loss = 0.0
            total_tokens = 0
            total_correct_top5 = 0

            PAD_ID = 0   # 你的词表里 <pad> = 0

            with torch.no_grad():
                for batch in test_loader:
                    labels = batch.to(device)          # [B, T]
                    input_dict = labels               

                    logits = self(input_dict)          # [B, vocab, T-1]
                    target = labels[:, 1:]             # [B, T-1]
                    B, V, Tm1 = logits.shape

                    # -------- 计算 loss（包含 PAD，不影响梯度） --------
                    loss = F.cross_entropy(logits, target)

                    # -------- 展平并 mask 掉 PAD --------
                    logits_flat = logits.permute(0, 2, 1).reshape(-1, V)  # [N, V]
                    target_flat = target.reshape(-1)                      # [N]

                    # mask PAD
                    mask = (target_flat != PAD_ID)
                    logits_flat = logits_flat[mask]        # [N_valid, V]
                    target_flat = target_flat[mask]        # [N_valid]
                    n_tokens = target_flat.numel()

                    # -------- top-5 accuracy --------
                    top5 = logits_flat.topk(5, dim=1).indices    # [N_valid, 5]
                    correct_top5 = (top5 == target_flat.unsqueeze(1)).any(dim=1).sum().item()

                    # -------- 累加 --------
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens
                    total_correct_top5 += correct_top5

            # 全局平均
            avg_loss = total_loss / total_tokens
            top5_acc = total_correct_top5 / total_tokens

            return avg_loss, top5_acc
        else: 
            test_loss = 0
            n_correct = 0
            n_total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                batch_loss = self.loss_func(outputs, labels)
                test_loss += batch_loss.item()

                labels_predicted = torch.argmax(outputs, dim=1)
                if labels.dim() == 2:
                    labels = torch.argmax(labels, dim=1)

                n_total += labels.size(0)
                n_correct += torch.sum(torch.eq(labels_predicted, labels)).item()

            if mode == "mean":
                test_loss /= n_total
            self.train()
            return test_loss, n_correct / n_total
    



    def get_thresholds(self):
        """
        遍历模型中的所有子模块，收集具有 'threshold' 属性的值并放入列表中。

        Returns:
            一个包含所有找到的 threshold 值的列表。
        """
        thresholds_list = []
        
        # self.named_modules() 是 nn.Module 内置的遍历器
        for name, module in self.named_modules():
            # 跳过根模块，因为它通常是整个模型
            if name == "":
                continue
                
            if hasattr(module, "threshold") and isinstance(module.threshold, torch.Tensor):
                thresholds_list.append(module.threshold.clone().detach())
                
        return thresholds_list
    def set_thresholds(self, new_thresholds):
            """
            遍历模型中的所有子模块，并将 new_thresholds 列表中的值按顺序赋给它们。

            Args:
                new_thresholds: 包含新 threshold 值的列表。列表的顺序必须与 
                                get_thresholds() 方法返回的顺序一致。
            """
            threshold_idx = 0
            
            for name, module in self.named_modules():
                # 跳过根模块
                if name == "":
                    continue
                    
                if hasattr(module, "threshold"):
                    
                    if threshold_idx >= len(new_thresholds):
                        # 如果提供的列表不够长，则停止赋值
                        print("Warning: new_thresholds list is shorter than the number of modules with threshold.")
                        break
                    
                    new_value = new_thresholds[threshold_idx]
                    
                    # 检查模块是否有自定义的 set_threshold 方法 (如您在 DenseLinear 中定义的)
                    if hasattr(module, "set_threshold") and callable(module.set_threshold):
                        # 如果有，优先调用模块内部的设置方法
                        module.set_threshold(new_value)
                    else:
                        # 否则，直接赋值给属性
                        # 注意：如果 module.threshold 是注册的 Buffer，直接赋值可能会替换 Buffer，
                        # 但在 PyTorch 稀疏化中，通常是替换值或调用 set_threshold。
                        if isinstance(module.threshold, torch.Tensor) and isinstance(new_value, (float, int)):
                            # 如果 module.threshold 是一个标量 Tensor，替换它的值
                            module.threshold.data.fill_(new_value)
                        elif isinstance(module.threshold, torch.Tensor) and isinstance(new_value, torch.Tensor):
                            # 如果都是 Tensor，替换它的值
                            module.threshold.data.copy_(new_value.data)
                        else:
                            # 否则，直接赋值 Python 对象
                            module.threshold = new_value

                    threshold_idx += 1
            
            if threshold_idx < len(new_thresholds):
                print(f"Warning: set_thresholds finished, but {len(new_thresholds) - threshold_idx} values in the list were unused.")
  
    



    def prune_by_threshold(self, thr_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(thr_arg, Sized):
            assert len(prunable_layers) == len(thr_arg)
        else:
            thr_arg = [thr_arg] * len(prunable_layers)
        for thr, layer in zip(thr_arg, prunable_layers):
            if thr is not None:#递归调用
                layer.prune_by_threshold(thr)

        return self

    def prune_by_rank(self, rank_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(rank_arg, Sized):
            assert len(prunable_layers) == len(rank_arg)
        else:
            rank_arg = [rank_arg] * len(prunable_layers)
        for rank, layer in zip(rank_arg, prunable_layers):
            if rank is not None:
                layer.prune_by_rank(rank)

        return self

    def retain_by_rank(self, rank_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(rank_arg, Sized):
            assert len(prunable_layers) == len(rank_arg)
        else:
            rank_arg = [rank_arg] * len(prunable_layers)
        for rank, layer in zip(rank_arg, prunable_layers):
            if rank is not None:
                layer.retain_by_rank(rank)

        return self

    def prune_by_pct(self, pct_arg: Union[int, float, Sized]):
        
        prunable_layers = self.prunable_layers
        if isinstance(pct_arg, Sized):
            assert len(prunable_layers) == len(pct_arg)
        else:
            pct_arg = [pct_arg] * len(prunable_layers)
        for pct, layer in zip(pct_arg, prunable_layers):
            if pct is not None:
                layer.prune_by_pct(pct)
        return self

    def random_prune_by_pct(self, pct_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(pct_arg, Sized):
            assert len(prunable_layers) == len(pct_arg)
        else:
            pct_arg = [pct_arg] * len(prunable_layers)

        for pct, layer in zip(pct_arg, prunable_layers):
            if pct is not None:
                layer.random_prune_by_pct(pct)

        return self

    @torch.no_grad()
    def reinit_from_model(self, final_model):
        assert isinstance(final_model, self.__class__)
        for self_layer, layer in zip(self.prunable_layers, final_model.prunable_layers):
            self_layer.mask = layer.mask.clone().to(self_layer.mask.device)

    def recover_model(self):
        for layer in self.prunable_layers:
            layer.mask = torch.ones_like(layer.mask)

    def calc_num_prunable_params(self, count_bias):
        total_param_in_use = 0
        total_param = 0
        for layer in self.prunable_layers:
            num_bias = layer.bias.nelement() if hasattr(layer, "bias") and layer.bias is not None and count_bias else 0
            num_weight = layer.num_weight
            num_params_in_use = num_weight + num_bias
            num_params = layer.weight.nelement() + num_bias#pytorch中的 nelement() 可以统计 tensor (张量) 的元素的个数
            total_param_in_use += num_params_in_use
            total_param += num_params

        return total_param_in_use, total_param


    def calc_num_all_active_params(self, count_bias):
        total_param = 0
        for layer in self.param_layers:
            num_bias = layer.bias.nelement() if hasattr(layer, "bias") and layer.bias is not None and count_bias else 0
            num_weight = layer.num_weight if hasattr(layer, "num_weight") else layer.weight.nelement()
            num_params = num_weight + num_bias
            total_param += num_params

        return total_param

    def nnz(self, count_bias=False):
        # number of parameters in use in prunable layers
        return self.calc_num_prunable_params(count_bias=count_bias)[0]

    def nelement(self, count_bias=False):
        # number of all parameters in prunable layers
        return self.calc_num_prunable_params(count_bias=count_bias)[1]

    def density(self, count_bias=False):
        total_param_in_use, total_param = self.calc_num_prunable_params(count_bias=count_bias)
        return total_param_in_use / total_param

    def _get_module_by_name_list(self, module_names: list):
        module = self
        for name in module_names:
            module = getattr(module, name)
        return module

    def get_module_by_name(self, module_name: str):
        return self._get_module_by_name_list(module_name.split('.'))

    def get_mask_by_name(self, param_name: str):
        if param_name.endswith("bias"):
            return None
        module = self._get_module_by_name_list(param_name.split('.')[:-1])
        return module.mask if hasattr(module, "mask") else None

    @abstractmethod
    def to_sparse(self):
        pass

    def to(self, *args, **kwargs):
        
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            # move data to device
            for m in self.prunable_layers:
                m.move_data(device)
        return super(BaseModel, self).to(*args, **kwargs)

    def device(self):
        return next(self.parameters()).device
