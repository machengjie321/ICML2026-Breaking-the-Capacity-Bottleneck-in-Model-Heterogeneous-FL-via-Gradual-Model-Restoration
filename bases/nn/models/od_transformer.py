import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from bases.nn.models.fiarse import Bern
from configs.StackOverflow import *
from bases.nn.od_layer import ODLinear,ODWrapper,ODConv2d,ODGroupNorm,ODEmbedding
from copy import deepcopy



from .utils import Scaler

class StackedTransformerEncoder(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout, rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout, rate)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None, p = 1.0):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, p = p)
        return src
    
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.positional_embedding = ODEmbedding(128, embedding_size)

    def forward(self, x, p = 1.0):
        N, S = x.size()
        position = torch.arange(S, dtype=torch.long, device=x.device).unsqueeze(0).expand((N, S))
        x = self.positional_embedding(position,p)
        return x


class TransformerEmbedding(nn.Module):
    def __init__(self, num_tokens, embedding_size, dropout, rate):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.positional_embedding = PositionalEmbedding(embedding_size)
        self.embedding = ODEmbedding(num_tokens + 1, embedding_size)
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.scaler = Scaler(rate)

    def forward(self, src, p = 1.0):
        src = self.scaler(self.embedding(src, p)) + self.scaler(self.positional_embedding(src, p))
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
        self.linear_q = ODLinear(embedding_size, embedding_size)
        self.linear_k = ODLinear(embedding_size, embedding_size)
        self.linear_v = ODLinear(embedding_size, embedding_size)
        self.linear_o = ODLinear(embedding_size, embedding_size)
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

    def forward(self, q, k, v, mask=None, p = 1.0):
        q, k, v = self.scaler(self.linear_q(q,p)), self.scaler(self.linear_k(k,p)), self.scaler(self.linear_v(v,p))
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        q, attn = self.attention(q, k, v, mask)
        q = self._reshape_from_batches(q)
        q = self.scaler(self.linear_o(q,p))
        return q, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout, rate):
        super().__init__()
        self.mha = MultiheadAttention(embedding_size, num_heads, rate=rate)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear1 = ODLinear(embedding_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = ODLinear(hidden_size, embedding_size)
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

    def forward(self, src, src_mask=None, src_key_padding_mask=None, p = 1.0):
        attn_output, _ = self.mha(src, src, src, mask=src_mask, p = p)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        src2 = self.scaler(self.linear2(self.dropout1(self.activation(self.scaler(self.linear1(src,p)))),p))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Decoder(nn.Module):
    def __init__(self, num_tokens, embedding_size, rate):
        super().__init__()
        self.linear1 = ODLinear(embedding_size, embedding_size)
        self.scaler = Scaler(rate)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear2 = nn.Linear(embedding_size, num_tokens)

    def forward(self, src, p):
        out = self.linear2(self.norm1(self.activation(self.scaler(self.linear1(src,p)))))
        return out


class Transformer(nn.Module):
    def __init__(self, num_tokens, embedding_size, num_heads,
                 hidden_size, num_layers, dropout, rate, bern=False):
        super(Transformer, self).__init__()

        self.num_tokens = num_tokens
        self.rate = rate
        self.bern = bern

        self.transformer_embedding = TransformerEmbedding(
            num_tokens, embedding_size, dropout, rate
        )
        self.transformer_encoder = StackedTransformerEncoder(
            embedding_size, num_heads, hidden_size, dropout, rate, num_layers
        )
        self.decoder = Decoder(num_tokens, embedding_size, rate)

        self.output_layer_prefix = 'decoder.linear2.'
        

    def forward(self, input, labels=None, p=1.0):
        """
        只输出预测值 logits，shape: [B, vocab, T-1]
        不在这里改写 input['label']，也不计算 loss
        """
        # 原始整句 label 序列：[w1, w2, ..., w_T]
        labels = input                   # [B, T]
        # 模型输入是前 T-1 个 token
        src = labels[:, :-1]                        # [B, T-1]
        # 正常走 embedding + encoder + decoder
        src = self.transformer_embedding(src, p=1.0)
        src = self.transformer_encoder(src, p=1.0)
        out = self.decoder(src, p=1.0)                     # [B, T-1, vocab]
        # 调整成 cross_entropy 需要的格式：[B, vocab, T-1]
        out = out.permute(0, 2, 1)
        return out

    def loss(self, input, label=None, rate = 1.0) -> torch.Tensor:
        """
        使用 forward 的预测 + 原始 input 中的 label 计算 loss。
        forward 只负责预测，这里负责构造 target。
        """
        logits = self(input,label,rate)                        # [B, vocab, T-1]
        # 目标是“下一词”：从第二个 token 开始
        target = input[:, 1:]              # [B, T-1]
        return F.cross_entropy(logits, target)
    
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



def transformer(model_rate=1, bern = False):
    model_rate = model_rate**0.5
    D = BASE_EMB // NUM_HEADS 
    model_rate = max(1, min(D, int(round(model_rate * D))))/D
    num_tokens = 10000
    num_heads = NUM_HEADS
    embedding_size = int(np.ceil(model_rate * BASE_EMB))
    hidden_size = int(np.ceil(model_rate * BASE_HID))
    num_layers = 3
    dropout = 0.1
    scaler_rate = model_rate / 1
    model = Transformer(num_tokens, embedding_size, num_heads, hidden_size, num_layers, dropout, scaler_rate, bern)
    return model
