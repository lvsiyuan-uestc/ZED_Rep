# src/model_head_mix.py  (v2.1: GN + gentle heads + asym pi bias)
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.ReLU(inplace=True)
        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)
        nn.init.ones_(self.norm1.weight); nn.init.zeros_(self.norm1.bias)
        # zero-gamma：残差第二层先“收敛为 0”，整体从近似恒等开始，更稳
        nn.init.zeros_(self.norm2.weight); nn.init.zeros_(self.norm2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act(out + x)
        return out

class MixtureLogisticHead(nn.Module):
    def __init__(self, K: int = 5, hidden: int = 128, num_blocks: int = 6, groups: int = 8):
        super().__init__()
        self.K = int(K); self.hidden = int(hidden); self.num_blocks = int(num_blocks)

        self.stem = nn.Conv2d(3, self.hidden, 3, padding=1, bias=True)
        self.trunk = nn.ModuleList([ResidualBlock(self.hidden, groups=groups) for _ in range(self.num_blocks)])

        self.head_pi = nn.Conv2d(self.hidden, self.K, 3, padding=1, bias=True)
        self.head_mu = nn.Conv2d(self.hidden, self.K * 3, 3, padding=1, bias=True)
        self.head_ls = nn.Conv2d(self.hidden, self.K * 3, 3, padding=1, bias=True)

        self._init_params()

    def _init_params(self):

        nn.init.kaiming_normal_(self.stem.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.stem.bias)


        nn.init.zeros_(self.head_pi.weight); nn.init.zeros_(self.head_mu.weight); nn.init.zeros_(self.head_ls.weight)

        with torch.no_grad():
            self.head_pi.bias.copy_(torch.linspace(-0.2, 0.2, steps=self.K))

            self.head_mu.bias.zero_()

            self.head_ls.bias.fill_(-0.8)

    def forward(self, y_up: torch.Tensor):
        feat = self.stem(y_up)
        for blk in self.trunk:
            feat = blk(feat)
        B, _, H, W = feat.shape
        pi_logits = self.head_pi(feat)                   # [B,K,H,W]
        mu        = self.head_mu(feat).view(B, self.K, 3, H, W)
        log_s     = self.head_ls(feat).view(B, self.K, 3, H, W)
        return pi_logits, mu, log_s

    def __repr__(self): return f"MixtureLogisticHead(K={self.K}, hidden={self.hidden}, num_blocks={self.num_blocks}, GN=True)"
    def __str__(self):  return f"Mixture(K={self.K}, hidden={self.hidden}, blocks={self.num_blocks}, GN)"
