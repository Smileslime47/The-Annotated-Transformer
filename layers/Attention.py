import torch
import torch.nn as nn
import math

from layers.Layer import clones


def attention(query, key, value, mask=None, dropout=None):
    """计算点积缩放注意力"""
    d_k = query.size(-1)
    "计算查询向量和键向量的点积，并除以sqrt(d_k)"
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    "调用softmax函数，将注意力得分转换为概率权重"
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    "返回权重和值向量的点积结果"
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """输入模型尺寸和注意力头数量"""
        super(MultiHeadedAttention, self).__init__()
        # 模型尺寸必然能够整除注意力头数量
        assert d_model % h == 0
        # 假定d_v永远等于d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """实现 Figure 2 中的算法"""
        if mask is not None:
            # 对所有 h 个注意力头应用相同的mask
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) 从 d_model => h x d_k 批量进行所有线性投影，即将 d_model 分割为 h 个 d_k 分配给注意力头
        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) 对所有投影向量批量应用注意力
        # Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) 通过一个视图拼接并应用最终的线性变换，即将被分割为 h 个 d_k 维的注意力头重新拼接为一个 d_model 的向量
        # "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)