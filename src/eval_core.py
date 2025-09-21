# -*- coding: utf-8 -*-
"""
src/eval_core.py
统一评测核心
。
"""

from __future__ import annotations
from typing import Tuple, Dict, Optional
import torch
import torch.nn.functional as F


from .utils import load_image_uint8
from .pyramid import build_xy_pyramid
from .stats import upsample_to, discretized_logistic_logpmf
from .stats_mix import mixture_loglik_and_resp, expected_H_from_resp
from .model_head import ResidualLogisticHead
from .model_head_mix import MixtureLogisticHead


def _affine_pyr(pyr, a: float, b: float):

    out = {}
    for k, v in pyr.items():
        out[k] = (v.to(torch.float32) * a + b)
    return out


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _infer_num_blocks_from_state(state_dict: dict, default: int = 6) -> int:
    idx = set()
    for k in state_dict.keys():
        if k.startswith("trunk."):
            parts = k.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                idx.add(int(parts[1]))
    return (max(idx) + 1) if idx else default

def _inflate_1x1_to_3x3_(state: dict, key_weight: str) -> None:

    if key_weight not in state:
        return
    w = state[key_weight]
    if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[-2:] == (1, 1):
        out_c, in_c, _, _ = w.shape
        w3 = torch.zeros((out_c, in_c, 3, 3), dtype=w.dtype)
        w3[:, :, 1, 1] = w[:, :, 0, 0]
        state[key_weight] = w3


# ------------------------ load heads ------------------------
def _build_head_from_ckpt(ckpt_path: str, device: torch.device):

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "head_pi.weight" in state:  # Mixture
        _inflate_1x1_to_3x3_(state, "head_pi.weight")
        _inflate_1x1_to_3x3_(state, "head_mu.weight")
        _inflate_1x1_to_3x3_(state, "head_ls.weight")

        K = int(state["head_pi.weight"].shape[0])
        hidden = int(state["stem.weight"].shape[0])
        blocks = _infer_num_blocks_from_state(state, default=6)
        model = MixtureLogisticHead(K=K, hidden=hidden, num_blocks=blocks).to(device).eval()
        model.load_state_dict(state, strict=False)
        model.is_mixture = True
    else:  # Residual
        model = ResidualLogisticHead().to(device).eval()
        model.load_state_dict(state, strict=True)
        model.is_mixture = False
    return model

def load_heads(ckpt1: str, ckpt0: str, device: torch.device):

    m1 = _build_head_from_ckpt(ckpt1, device)
    m0 = _build_head_from_ckpt(ckpt0, device)
    return m1, m0


# ------------------------ per-level NLL/H/D ------------------------
@torch.no_grad()
def _nll_h_d_for_level(x_uint8: torch.Tensor,
                       y_low: torch.Tensor,
                       model: torch.nn.Module,
                       device: torch.device,
                       pi_temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


    xb = x_uint8.unsqueeze(0).to(device)
    y_up = upsample_to(y_low.unsqueeze(0), xb.shape[-2:]).to(device)

    if getattr(model, "is_mixture", False):

        pi_logits, mu, log_s = model(y_up)
        if pi_temp != 1.0:
            pi_logits = pi_logits / float(pi_temp)

        xb_f = xb.to(mu.dtype)
        logp, r, _ = mixture_loglik_and_resp(xb_f, pi_logits, mu, log_s)
        H = expected_H_from_resp(log_s, r)

        NLL = (-logp).squeeze(0).float().cpu()
        H   = H.squeeze(0).float().cpu()
        D   = NLL - H
        return NLL, H, D

    else:

        mean, log_s = model(y_up)
        logp = discretized_logistic_logpmf(xb, mean, log_s)
        NLL = (-logp).squeeze(0).float().cpu()
        H   = (log_s + 2.0).mean(dim=1).squeeze(0).float().cpu()
        D   = NLL - H
        return NLL, H, D


# ------------------------ per-image D1/D0 maps & scalars ------------------------
@torch.no_grad()
def compute_Ds_for_image(img_path, m1, m0, device,
                         max_side=1280, pi_temp1=1.0, pi_temp0=1.0,
                         scale1=("uint8", 1.0, 0.0),   # 新增：m1 的量纲
                         scale0=("uint8", 1.0, 0.0)):  # 新增：m0 的量纲

    x0 = load_image_uint8(img_path, require_divisible_by=8, max_long_side=max_side)
    pyr = build_xy_pyramid(x0, device="cpu")

    name1, a1, b1 = scale1
    name0, a0, b0 = scale0
    pyr1 = _affine_pyr(pyr, a1, b1)
    pyr0 = _affine_pyr(pyr, a0, b0)

    # D1 ← (x1|y2, m1) with scale1
    NLL1, H1, D1_map = _nll_h_d_for_level(pyr1["x1"], pyr1["y2"], m1, device, pi_temp=pi_temp1)
    # D0 ← (x0|y1, m0) with scale0
    NLL0, H0, D0_map = _nll_h_d_for_level(pyr0["x0"], pyr0["y1"], m0, device, pi_temp=pi_temp0)

    D1 = float(D1_map.mean().item())
    D0 = float(D0_map.mean().item())
    return {
        "D1": D1, "D0": D0, "Delta01": D0 - D1, "AbsDelta01": abs(D0 - D1),
        "NLL1": float(NLL1.mean().item()), "H1": float(H1.mean().item()),
        "NLL0": float(NLL0.mean().item()), "H0": float(H0.mean().item()),
    }


def compute_Dmaps_for_image(img_path, m1, m0, device,
                            max_side=1280, pi_temp1=1.0, pi_temp0=1.0,
                            scale1=("uint8", 1.0, 0.0),
                            scale0=("uint8", 1.0, 0.0)):
    x0 = load_image_uint8(img_path, require_divisible_by=8, max_long_side=max_side)
    pyr = build_xy_pyramid(x0, device="cpu")
    name1, a1, b1 = scale1
    name0, a0, b0 = scale0
    pyr1 = _affine_pyr(pyr, a1, b1)
    pyr0 = _affine_pyr(pyr, a0, b0)

    _, _, D1_map = _nll_h_d_for_level(pyr1["x1"], pyr1["y2"], m1, device, pi_temp=pi_temp1)
    _, _, D0_map = _nll_h_d_for_level(pyr0["x0"], pyr0["y1"], m0, device, pi_temp=pi_temp0)
    return D1_map, D0_map


@torch.no_grad()
def aligned_delta_map_for_visual(D0_map: torch.Tensor,
                                 D1_map: torch.Tensor,
                                 mode: str = "area") -> torch.Tensor:

    if D0_map.ndim != 2 or D1_map.ndim != 2:
        raise ValueError("D0_map/D1_map 都应是 [H,W] 的 2D tensor")
    H0, W0 = D0_map.shape
    if mode not in ("nearest", "bilinear", "bicubic", "area"):
        mode = "area"
    D1_up = F.interpolate(D1_map.unsqueeze(0).unsqueeze(0),
                          size=(H0, W0),
                          mode=mode).squeeze(0).squeeze(0)
    return (D0_map - D1_up).contiguous()
