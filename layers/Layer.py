import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


class Generator(nn.Module):
    """定义线性变换+应用softmax函数步骤"""

    def __init__(self, d_model, vocab):
        """定义一个线性变换函数"""
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """__call__调用，对x应用线性变换后再应用softmax函数"""
        return log_softmax(self.proj(x), dim=-1)

class SublayerConnection(nn.Module):
    """
    子层间连接（每个Encoder/Decoder层包含多个子层）
    残差连接和层归一化部分
    为了代码的简洁性，这里先对x进行了归一化

    原文的公式为layerNorm(x+sublayer(x))
    这里的公式是x+sublayer(layerNorm(x))
    """

    def __init__(self, size, dropout):
        """
        定义dropout函数和层归一化函数
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        __call__调用
        对每个相同尺寸的子层输出应用残差链接
        """
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    """构建一个层归一化模块"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def clones(module, N):
    """将torch module深拷贝N次组成一个list返回"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])