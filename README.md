---
title: 《The Annotated Transformer》
author: Liu Yibang
date: 2024/09/07
categories: 
    - Artificial Intelligence
    - Article
mathjax: true
---

## 前置准备

> [清华镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)
> 
> [换源教程](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)

Anaconda官网只能下载到Python3.12版本的发行版，但是该论文的许多库在最新版Python下不兼容。例如torchtext 0.12.0最新版只有3.11版本，而新版的torchtext又无法运行下面的代码
- 可以考虑下载Anaconda3-2020.07版本的，该版本使用的是Python3.8  

### requirements.txt

```
--find-links https://download.pytorch.org/whl/torch_stable.html

pandas==1.3.5
torch==1.11.0+cu113
torchdata==0.3.0
torchtext==0.12
spacy==3.2
altair==4.1
jupytext==1.13
flake8
black
GPUtil
wandb
```

### 安装依赖环境

```commandline
pip install -r requirements.txt
pip install -q torchdata==0.3.0 torchtext==0.12 spacy==3.2 altair GPUtil
pip install de_core_news_sm-3.2.0-py3-none-any.whl --no-deps
pip install en_core_web_sm-3.2.0-py3-none-any.whl --no-deps

```

### 导包

```py
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


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True
```
