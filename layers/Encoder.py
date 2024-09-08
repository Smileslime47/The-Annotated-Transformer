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

from Layer import clones, SublayerConnection, LayerNorm


class Encoder(nn.Module):
    """由N层EncoderLayer构成"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """将x逐层编码后，经过层归一化丢给Decoder"""
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    """Encoder由self_attn和feed_forward构成"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        EncoderLayer由两个子层构成，通过SublayerConnection连接
        sublayer的具体行为在下文通过forward调用时定义

        :param size:
        :param self_attn: 自注意力层函数
        :param feed_forward: 前馈层函数
        :param dropout: Dropout函数
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        调用一个Encoder层的全部行为

        :param x: 输入
        :param mask:
        :return: Encoder层输出
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)