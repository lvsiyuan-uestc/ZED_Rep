# src/model_head.py
"""
Residual Logistic Head (level-agnostic)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_hidden):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_hidden, 3, padding=1)
        self.c2 = nn.Conv2d(c_hidden, c_hidden, 3, padding=1)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.act(self.c1(x)); x = self.act(self.c2(x)); return x

class ResidualLogisticHead(nn.Module):
    def __init__(self, c_hidden: int = 64, min_log_scale: float = -1.5, max_log_scale: float = 3.0):
        super().__init__()
        self.stem   = nn.Conv2d(3, c_hidden, 3, padding=1)
        self.block1 = ConvBlock(c_hidden, c_hidden)
        self.block2 = ConvBlock(c_hidden, c_hidden)
        self.head   = nn.Conv2d(c_hidden, 6, 1)   # delta(3)+log_s(3)
        self.act    = nn.GELU()
        self.min_log_scale = min_log_scale
        self.max_log_scale = max_log_scale
        # 初始化：log_s 初始≈1.5（等价基线），delta 初始≈0
        with torch.no_grad():
            if self.head.bias is not None:
                self.head.bias.zero_()
                self.head.bias[3:].fill_(1.5)

    def forward(self, y_up: torch.Tensor):
        """
        y_up: [B,3,H,W]  —— y^{l+1} 上采样到当前层的分辨率
        返回: mean, log_scale (各 [B,3,H,W])
        """
        x = self.act(self.stem(y_up))
        x = self.block1(x); x = self.block2(x)
        out = self.head(x)
        delta, log_s = torch.chunk(out, 2, dim=1)
        mean = y_up + delta
        log_s = torch.clamp(log_s, self.min_log_scale, self.max_log_scale)
        return mean, log_s
