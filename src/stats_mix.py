# -*- coding: utf-8 -*-
"""
Mixture logistic 的离散对数似然（按 0..255 像素单位）与熵期望
"""

import torch
import torch.nn.functional as F

_EPS = 1e-12

def disc_logistic_logpmf_per_comp(x_uint8, mu, log_s):
    x = x_uint8.to(mu.dtype)
    s = torch.exp(log_s).clamp_min(1e-4)
    u1 = (x.unsqueeze(1) - 0.5 - mu) / s
    u2 = (x.unsqueeze(1) + 0.5 - mu) / s
    cdf2 = torch.sigmoid(u2).clamp_min(_EPS)
    cdf1 = torch.sigmoid(u1).clamp_min(0.0)
    pmf  = (cdf2 - cdf1).clamp_min(_EPS)
    return torch.log(pmf)

def mixture_loglik_and_resp(x_uint8, pi_logits, mu, log_s):

    logpmf_kc = disc_logistic_logpmf_per_comp(x_uint8, mu, log_s)  # [B,K,3,H,W]

    logpmf_k = logpmf_kc.mean(dim=2)                                # [B,K,H,W]

    logw   = pi_logits - torch.logsumexp(pi_logits, dim=1, keepdim=True)  # log pi
    logp   = torch.logsumexp(logw + logpmf_k, dim=1)                 # [B,H,W]

    r = torch.softmax(logw + logpmf_k, dim=1)                        # [B,K,H,W]
    return logp, r, logpmf_k

def expected_H_from_resp(log_s, r):
    Hk = (log_s + 2.0).mean(dim=2)    # [B,K,H,W]
    H  = (r * Hk).sum(dim=1)          # [B,H,W]
    return H
