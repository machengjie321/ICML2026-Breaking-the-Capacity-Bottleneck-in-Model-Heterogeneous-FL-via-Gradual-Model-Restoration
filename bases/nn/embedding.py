import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import Literal, Optional
from typing import Any, Literal
from bases.nn.models.fiarse import Bern

class SparseEmbedding(torch.nn.Module):
    pass
    

class DenseEmbedding(nn.Embedding):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 # 新增的属性
                use_mask=True, bern = False):
        """
        MaskedEmbedding:
        - 和 nn.Embedding 构造函数保持完全兼容
        - 额外增加:
            threshold: 剪枝阈值
            bern: 是否使用 Bernoulli(mask) 方式
            use_score_param: 是否使用可训练的 score 参数来生成 mask
        """
        super().__init__(num_embeddings,
                         embedding_dim,
                         padding_idx=padding_idx,
                         max_norm=max_norm,
                         norm_type=norm_type,
                         scale_grad_by_freq=scale_grad_by_freq,
                         sparse=sparse)

        # 剪枝相关属性
        self.bern = bern
        # self.threshold = torch.tensor(0.)

        self.use_mask = use_mask
        self.threshold = torch.tensor(0.)
        self.mask = torch.ones_like(self.weight, dtype=torch.bool)
        

        
        
    def forward(self, input):
           
        if self.bern:
            self.mask =  Bern.apply(torch.abs(self.weight), self.threshold)
            
        masked_weight = self.weight * self.mask if self.use_mask else self.weight
        return F.embedding(
            input,  masked_weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def set_threshold(self, value):
        self.threshold = value


    def prune_by_threshold(self, thr):
        self.mask *= (torch.abs(self.weight) >= thr)

    def retain_by_threshold(self, thr):
        self.mask *= (torch.abs(self.weight) >= thr)

    def prune_by_rank(self, rank):
        if rank == 0:
            return
        
        weights_val = self.weight[self.mask == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[rank]
        self.set_threshold(thr)
        self.prune_by_threshold(thr)

    def retain_by_rank(self, rank):
        weights_val = self.weight[self.mask == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val), descending=True)[0]
        thr = sorted_abs_weights[rank]
        self.retain_by_threshold(thr)

    def prune_by_pct(self, pct):
        if pct == 0:
            return
        prune_idx = int(self.num_weight * pct)
        self.prune_by_rank(prune_idx)

    def random_prune_by_pct(self, pct):

        prune_idx = int(self.num_weight * pct)
        rand = torch.rand(self.mask.size(), device=self.mask.device)
        rand_val = rand[self.mask == 1]
        sorted_abs_rand = torch.sort(rand_val)[0]
        thr = sorted_abs_rand[prune_idx]
        self.mask *= (rand >= thr)
        
    @property
    def size(self):
        return self.weight.numel()

    def move_data(self, device: torch.device):
        self.mask = self.mask.to(device)
    
    def to_sparse(self):
        print('Waiting to implement')
        pass
        

    @property
    def num_weight(self):
        return torch.sum(self.mask).int().item()

    
    def to_vector(self, attr:Literal['param', 'score', 'grad']):
        # weight is not None 
        if attr == 'param':
            vector = self.weight.view(-1)
        elif attr == 'score':
            vector = torch.randn_like(self.weight).view(-1)
        elif attr == 'grad':
            vector = self.weight.grad.view(-1)
        else:
            raise ValueError
        
        return [vector]