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


class Decoder(nn.Module):
    """由带有masking的N层DecoderLayer构成"""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """Decoder由self_attn、src_attn和feed_forward组成"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        DecoderLayer由三个子层构成，通过SublayerConnection连接
        sublayer的具体行为在下文通过forward调用时定义

        :param size:
        :param self_attn: 自注意力层函数
        :param src_attn:
        :param feed_forward: 前馈层函数
        :param dropout: Dropout函数
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        调用一个Decoder层的全部行为

        :param x: 输入
        :param memory
        :param src_mask
        :param tgt_mask
        :return: Encoder层输出
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)