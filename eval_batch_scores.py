#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_batch_scores.py — 批量 Δ01 评测（仅对 x* 做量纲缩放）
-------------------------------------------------------------------
示例：
  # 单类导出（用于分别得到 real/fake 的 CSV）
  python eval_batch_scores.py --dir data/real \
    --ckpt1 runs/level1_head_mixK5_v21.pt --scale1 uint8 \
    --ckpt0 runs/level0_head_mixK5_h128b6_v21.pt --scale0 uint8 \
    --csv runs/real_scores_u8.csv --label 1
  python eval_batch_scores.py --dir data/fake \
    --ckpt1 runs/level1_head_mixK5_v21.pt --scale1 uint8 \
    --ckpt0 runs/level0_head_mixK5_h128b6_v21.pt --scale0 uint8 \
    --csv runs/fake_scores_u8.csv --label 0
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np
import torch

from src.eval_core import device_auto, _nll_h_d_for_level, load_heads
from src.utils import load_image_uint8
from src.pyramid import build_xy_pyramid

SCALE_PRESETS = {
    "uint8": (1.0, 0.0),
    "unit":  (1.0/255.0, 0.0),
    "tanh":  (1.0/127.5, -1.0)
}
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def _iter_images(path: str) -> Iterable[str]:
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                yield str(f)
    else:
        if p.suffix.lower() in IMG_EXTS:
            yield str(p)
        else:
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield line

def _apply_affine_to_x_only(pyr: Dict[str, torch.Tensor], a: float, b: float) -> Dict[str, torch.Tensor]:

    out = dict(pyr)  # 浅拷贝
    for k in list(pyr.keys()):
        if k.startswith("x"):  # x0, x1, x2 ...
            out[k] = pyr[k].to(torch.float32) * a + b
    return out

@torch.no_grad()
def _mean_D_for_level(model, device, pyr: Dict[str, torch.Tensor], level: int, pi_temp: float) -> float:
    if level == 0:
        xk, yk1 = "x0", "y1"
    elif level == 1:
        xk, yk1 = "x1", "y2"
    else:
        xk, yk1 = "x2", "y3"
    NLL, H, D = _nll_h_d_for_level(pyr[xk], pyr[yk1], model, device, pi_temp=pi_temp)
    return float(D.mean().item())

def main():
    ap = argparse.ArgumentParser("Batch Δ01 evaluator (scale only x*, keep y* intact)")
    ap.add_argument("--dir", type=str, required=True, help="图像目录 / 单个文件 / 清单文件")
    ap.add_argument("--ckpt1", type=str, required=True, help="L1 头权重")
    ap.add_argument("--ckpt0", type=str, required=True, help="L0 头权重")
    ap.add_argument("--scale1", type=str, default="uint8", choices=SCALE_PRESETS.keys(),
                    help="L1 头所需的 x* 量纲")
    ap.add_argument("--scale0", type=str, default="uint8", choices=SCALE_PRESETS.keys(),
                    help="L0 头所需的 x* 量纲")
    ap.add_argument("--pi_temp1", type=float, default=1.0)
    ap.add_argument("--pi_temp0", type=float, default=1.0)
    ap.add_argument("--max_side", type=int, default=1536)
    ap.add_argument("--csv", type=str, required=True, help="输出 CSV 路径")
    ap.add_argument("--label", type=int, default=None, help="可选：常量标签(0=fake,1=real)写入 CSV")
    args = ap.parse_args()

    device = device_auto()
    m1, m0 = load_heads(args.ckpt1, args.ckpt0, device)
    a1, b1 = SCALE_PRESETS[args.scale1]
    a0, b0 = SCALE_PRESETS[args.scale0]

    rows: List[Tuple] = []
    total = 0
    ok = 0

    with torch.no_grad():
        for img_path in _iter_images(args.dir):
            total += 1
            try:
                x0 = load_image_uint8(img_path, require_divisible_by=8, max_long_side=args.max_side)
                pyr_u8 = build_xy_pyramid(x0, device="cpu")  # x*: uint8, y*: float
                # 仅缩放 x*，y* 原样
                pyr1 = _apply_affine_to_x_only(pyr_u8, a1, b1)
                pyr0 = _apply_affine_to_x_only(pyr_u8, a0, b0)

                D1 = _mean_D_for_level(m1, device, pyr1, level=1, pi_temp=args.pi_temp1)
                D0 = _mean_D_for_level(m0, device, pyr0, level=0, pi_temp=args.pi_temp0)
                Delta01 = D0 - D1
                AbsDelta01 = abs(Delta01)
                AbsD0 = abs(D0)

                if args.label is None:
                    rows.append((img_path, D1, D0, Delta01, AbsDelta01, AbsD0))
                else:
                    rows.append((img_path, args.label, D1, D0, Delta01, AbsDelta01, AbsD0))
                ok += 1
            except Exception as e:
                # 把错误也写入，便于排查
                if args.label is None:
                    rows.append((img_path, "ERR", "ERR", "ERR", f"{type(e).__name__}: {e}", "ERR"))
                else:
                    rows.append((img_path, args.label, "ERR", "ERR", "ERR", f"{type(e).__name__}: {e}", "ERR"))

    out = Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        if args.label is None:
            writer = csv.writer(fh)
            writer.writerow(["path", "D1", "D0", "Delta01", "AbsDelta01", "AbsD0"])
            writer.writerows(rows)
        else:
            writer = csv.writer(fh)
            writer.writerow(["path", "label", "D1", "D0", "Delta01", "AbsDelta01", "AbsD0"])
            writer.writerows(rows)

    # 仅对成功样本做个摘要
    vals = [r[4] if args.label is None else r[5] for r in rows
            if isinstance((r[4] if args.label is None else r[5]), (int, float))]
    if vals:
        arr = np.array(vals, dtype=np.float64)
        print(f"[done] wrote {len(rows)} rows → {out}  (ok={ok}, err={len(rows)-ok})")
        print(f"[summary] |Δ01| mean={arr.mean():.4f}  median={np.median(arr):.4f}  p95={np.percentile(arr,95):.4f}")
    else:
        print(f"[done] wrote {len(rows)} rows → {out}  (no valid rows)")

if __name__ == "__main__":
    main()
