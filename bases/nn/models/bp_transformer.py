import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from bases.nn.models.fiarse import Bern
from configs.StackOverflow import *

from copy import deepcopy

from bases.nn.conv2d import Bias_DenseConv2d as DenseConv2d
from bases.nn.linear import Bias_DenseLinear as DenseLinear 

from bases.nn.models.base_model import BaseModel
from bases.nn.sequential import DenseSequential
from bases.nn.embedding import  DenseEmbedding
from .utils import is_conv, is_fc, is_embb
from .utils import Scaler

class StackedTransformerEncoder(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout, rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout, rate)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return src
    
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.positional_embedding = DenseEmbedding(128, embedding_size)

    def forward(self, x):
        N, S = x.size()
        position = torch.arange(S, dtype=torch.long, device=x.device).unsqueeze(0).expand((N, S))
        x = self.positional_embedding(position)
        return x


class TransformerEmbedding(nn.Module):
    def __init__(self, num_tokens, embedding_size, dropout, rate):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.positional_embedding = PositionalEmbedding(embedding_size)
        self.embedding = DenseEmbedding(num_tokens + 1, embedding_size)
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.scaler = Scaler(rate)

    def forward(self, src):
        src = self.scaler(self.embedding(src)) + self.scaler(self.positional_embedding(src))
        src = self.dropout(self.norm(src))
        return src


class ScaledDotProduct(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        scores = q.matmul(k.transpose(-2, -1)) / self.temperature
        seq_len = scores.shape[-1]
        h = scores.shape[0]
        mask = torch.tril(torch.ones((h, seq_len, seq_len))).to(str(scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, rate):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.linear_q = DenseLinear(embedding_size, embedding_size)
        self.linear_k = DenseLinear(embedding_size, embedding_size)
        self.linear_v = DenseLinear(embedding_size, embedding_size)
        self.linear_o = DenseLinear(embedding_size, embedding_size)
        self.attention = ScaledDotProduct(temperature=(embedding_size // num_heads) ** 0.5)
        self.scaler = Scaler(rate)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim).permute(0, 2, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature).permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.scaler(self.linear_q(q)), self.scaler(self.linear_k(k)), self.scaler(self.linear_v(v))
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        q, attn = self.attention(q, k, v, mask)
        q = self._reshape_from_batches(q)
        q = self.scaler(self.linear_o(q))
        return q, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout, rate):
        super().__init__()
        self.mha = MultiheadAttention(embedding_size, num_heads, rate=rate)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear1 = DenseLinear(embedding_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = DenseLinear(hidden_size, embedding_size)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.scaler = Scaler(rate)
        self.activation = nn.GELU()
        self.init_param()

    def init_param(self):
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.weight.data.fill_(1.0)
        self.norm1.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        return

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, _ = self.mha(src, src, src, mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        src2 = self.scaler(self.linear2(self.dropout1(self.activation(self.scaler(self.linear1(src))))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Decoder(nn.Module):
    def __init__(self, num_tokens, embedding_size, rate):
        super().__init__()
        self.linear1 = DenseLinear(embedding_size, embedding_size)
        self.scaler = Scaler(rate)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear2 = nn.Linear(embedding_size, num_tokens)

    def forward(self, src):
        out = self.linear2(self.norm1(self.activation(self.scaler(self.linear1(src)))))
        return out


class Transformer(BaseModel):
    def __init__(self, num_tokens, embedding_size, num_heads, hidden_size, num_layers, dropout, rate, bern = False):
        dict_module = dict()
        self.num_tokens = num_tokens
        
        dict_module['transformer_embedding'] = TransformerEmbedding(num_tokens, embedding_size, dropout, rate)
        dict_module['transformer_encoder'] = StackedTransformerEncoder(
            embedding_size, num_heads, hidden_size, dropout, rate, num_layers
        )
        dict_module['decoder'] = Decoder(num_tokens, embedding_size, rate)
        self.dict_module = dict_module
        self.output_layer_prefix =  'decoder.linear2.'
        super(Transformer, self).__init__(F.cross_entropy, dict_module)
        

    def forward(self, input, labels=None):
        """
        只输出预测值 logits，shape: [B, vocab, T-1]
        不在这里改写 input['label']，也不计算 loss
        """
        # 原始整句 label 序列：[w1, w2, ..., w_T]
        labels = input                   # [B, T]
        # 模型输入是前 T-1 个 token
        src = labels[:, :-1]                        # [B, T-1]
        # 正常走 embedding + encoder + decoder
        src = self.transformer_embedding(src)
        src = self.transformer_encoder(src)
        out = self.decoder(src)                     # [B, T-1, vocab]
        # 调整成 cross_entropy 需要的格式：[B, vocab, T-1]
        out = out.permute(0, 2, 1)
        return out

    def loss(self, input, label=None) -> torch.Tensor:
        """
        使用 forward 的预测 + 原始 input 中的 label 计算 loss。
        forward 只负责预测，这里负责构造 target。
        """
        logits = self(input)                        # [B, vocab, T-1]
        # 目标是“下一词”：从第二个 token 开始
        target = input[:, 1:]              # [B, T-1]
        return self.loss_func(logits, target)
    
    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)#修改了param_layers和param_layer_predixes
        self.prunable_layers = self.param_layers
        self.prunable_layer_prefixes = self.param_layer_prefixes
    

    def to_sparse(self):
        return 1


def transformer(model_rate=1, bern = False):
    model_rate = model_rate**0.5
    D = BASE_EMB // NUM_HEADS 
    model_rate = max(1, min(D, int(round(model_rate * D))))/D
    num_tokens = 10000
    num_heads = NUM_HEADS
    embedding_size = int(np.ceil(model_rate * BASE_EMB))
    hidden_size = int(np.ceil(model_rate * BASE_HID))
    num_layers = 2
    dropout = 0.1
    scaler_rate = model_rate / 1
    model = Transformer(num_tokens, embedding_size, num_heads, hidden_size, num_layers, dropout, scaler_rate, bern)
    return model
