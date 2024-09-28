from torch import nn


class PositionwiseFeedForward(nn.Module):
    """实现FFN的方程"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """定义两个线性变换和dropout函数"""
        super(PositionwiseFeedForward, self).__init__()
        "从d_model变维到d_ff"
        self.w_1 = nn.Linear(d_model, d_ff)
        "从d_ff变维到d_model"
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """在执行w_1函数后丢给ReLU，然后再执行w_2函数"""
        return self.w_2(self.dropout(self.w_1(x).relu()))