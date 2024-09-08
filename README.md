---
title: 《The Annotated Transformer》
author: Liu Yibang
date: 2024/09/07
categories: 
    - Artificial Intelligence
    - Article
mathjax: true
---

> [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

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

- [Torch whl](https://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-win_amd64.whl)
- [en_core_web_sm](https://objects.githubusercontent.com/github-production-release-asset-2e65be/84940268/1b46d25d-fb12-424a-b108-38788bbefc92?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20240907%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240907T094245Z&X-Amz-Expires=300&X-Amz-Signature=2a646b62e82500c91830445592ceec146e9b8fe623393f90d08832844e964513&X-Amz-SignedHeaders=host&actor_id=77948910&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Den_core_web_sm-3.2.0-py3-none-any.whl&response-content-type=application%2Foctet-stream)
- [de_core_news_sm](https://objects.githubusercontent.com/github-production-release-asset-2e65be/84940268/ebef9b23-442c-43cf-8dd9-82b90d11f2b3?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20240907%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240907T094313Z&X-Amz-Expires=300&X-Amz-Signature=7c11d52c68f2feaa4a9e79907f22cf280b7fcc3d8a5e551d470fa661a1a6bd73&X-Amz-SignedHeaders=host&actor_id=77948910&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Dde_core_news_sm-3.2.0-py3-none-any.whl&response-content-type=application%2Foctet-stream)

```commandline
pip install -r requirements.txt
pip install -q torchdata==0.3.0 torchtext==0.12 spacy==3.2 altair GPUtil
pip install de_core_news_sm-3.2.0-py3-none-any.whl --no-deps
pip install en_core_web_sm-3.2.0-py3-none-any.whl --no-deps
```
