#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_delta01.py — 单图 Δ01 评测（零侵入版，内置按头量纲缩放）

用法示例：
  python eval_delta01.py --img data/real/real_00123.jpg \
    --ckpt1 runs/level1_head_mixK5_v21.pt --scale1 uint8 \
    --ckpt0 runs/level0_head_mixK5_h96b4.pt --scale0 unit \
    --save_maps
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from PIL import Image


from src.eval_core import device_auto, aligned_delta_map_for_visual, _nll_h_d_for_level, load_heads
from src.utils import load_image_uint8
from src.pyramid import build_xy_pyramid

SCALE_PRESETS = {
    "uint8": (1.0, 0.0),
    "unit":  (1.0/255.0, 0.0),
    "tanh":  (1.0/127.5, -1.0),
}

def _apply_affine_pyr(pyr: Dict[str, torch.Tensor], a: float, b: float) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in pyr.items():
        out[k] = v.to(torch.float32) * a + b
    return out

def _to_uint8_gray(arr: torch.Tensor, robust: float = 1.0) -> Image.Image:
    x = arr.detach().cpu().float().numpy()
    if not np.isfinite(x).any():
        x = np.zeros_like(x, dtype=np.float32)
    if 0.5 < robust < 1.0:
        lo = float(np.quantile(x, 1.0 - robust))
        hi = float(np.quantile(x, robust))
    else:
        lo = float(np.nanmin(x)); hi = float(np.nanmax(x))
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi) or hi <= lo: hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return Image.fromarray((y * 255.0 + 0.5).astype(np.uint8), mode="L")

def _calc_for_level(model, device, pyr: Dict[str, torch.Tensor], level: int, pi_temp: float):
    if level == 0: xk, yk1 = "x0", "y1"
    elif level == 1: xk, yk1 = "x1", "y2"
    else:            xk, yk1 = "x2", "y3"
    NLL, H, D = _nll_h_d_for_level(pyr[xk], pyr[yk1], model, device, pi_temp=pi_temp)
    return NLL, H, D, float(NLL.mean()), float(H.mean()), float(D.mean())

def main():
    ap = argparse.ArgumentParser("Single-image Δ01 evaluator with per-head scaling")
    ap.add_argument("--img", type=str, required=True)
    ap.add_argument("--ckpt1", type=str, required=True, help="Level-1 head (用于 L1: x1|y2)")
    ap.add_argument("--ckpt0", type=str, required=True, help="Level-0 head (用于 L0: x0|y1)")
    ap.add_argument("--scale1", type=str, default="uint8", choices=SCALE_PRESETS.keys(), help="ckpt1 的量纲")
    ap.add_argument("--scale0", type=str, default="uint8", choices=SCALE_PRESETS.keys(), help="ckpt0 的量纲")
    ap.add_argument("--pi_temp1", type=float, default=1.0)
    ap.add_argument("--pi_temp0", type=float, default=1.0)
    ap.add_argument("--max_side", type=int, default=1536)
    ap.add_argument("--save_maps", action="store_true")
    ap.add_argument("--out", type=str, default="runs/debug")
    ap.add_argument("--robust_vis", type=float, default=1.0, help="0.5<r<1 使用分位裁剪；=1 用 min-max")
    args = ap.parse_args()

    device = device_auto()
    m1, m0 = load_heads(args.ckpt1, args.ckpt0, device)

    x0 = load_image_uint8(args.img, require_divisible_by=8, max_long_side=args.max_side)
    pyr_u8 = build_xy_pyramid(x0, device="cpu")

    a1, b1 = SCALE_PRESETS[args.scale1]
    a0, b0 = SCALE_PRESETS[args.scale0]
    pyr1 = _apply_affine_pyr(pyr_u8, a1, b1)
    pyr0 = _apply_affine_pyr(pyr_u8, a0, b0)

    _, _, D1_map, NLL1m, H1m, D1m = _calc_for_level(m1, device, pyr1, level=1, pi_temp=args.pi_temp1)
    _, _, D0_map, NLL0m, H0m, D0m = _calc_for_level(m0, device, pyr0, level=0, pi_temp=args.pi_temp0)

    D1, D0 = float(D1m), float(D0m)
    Delta01 = D0 - D1
    AbsDelta01 = abs(Delta01)
    print(f"D1={D1:.4f}  D0={D0:.4f}  Δ01={Delta01:.4f}  |Δ01|={AbsDelta01:.4f}")

    if args.save_maps:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        _to_uint8_gray(D1_map, robust=args.robust_vis).save(out / "D1_native.png")
        _to_uint8_gray(D0_map, robust=args.robust_vis).save(out / "D0_native.png")
        Delta_vis = aligned_delta_map_for_visual(D0_map, D1_map, mode="area")
        _to_uint8_gray(Delta_vis, robust=args.robust_vis).save(out / "Delta01_vis.png")
        print(f"[ok] saved D1_native/D0_native/Delta01_vis to {out}")

if __name__ == "__main__":
    main()
