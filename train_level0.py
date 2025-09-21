#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_level0.py — 训练 Level-0 条件像素模型 (x0 | y1)

用法示例：
  # 旧头(K=1)，调小 batch 以防 OOM
  python train_level0.py --steps 3000 --batch_size 4 --amp --save runs/level0_head_res_coco.pt
  # 混合头(K=5)
  python train_level0.py --steps 3000 --batch_size 2 --mix_k 5 --amp --channels_last --save runs/level0_head_mixK5.pt
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# project modules
from src.dset_realpairs import RealPairsDataset
from src.model_head import ResidualLogisticHead
from src.model_head_mix import MixtureLogisticHead
from src.stats import upsample_to, discretized_logistic_logpmf
from src.stats_mix import mixture_loglik_and_resp


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # 允许 TF32，加速并保持数值稳定
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id); random.seed(seed + worker_id)


def make_loader(list_or_dir: str, level: int = 0, batch_size: int = 4, num_workers: int = 4):
    ds = RealPairsDataset(
        list_or_dir=list_or_dir, level=level,
        jpeg_q=(80, 100), short_range=(512, 896)
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def save_ckpt(model: torch.nn.Module, path: str):
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[ckpt] saved to {path}")


def lr_schedule(step: int, total: int, base_lr: float, warmup: int) -> float:
    """线性 warmup + cosine 衰减（到 0）。"""
    if warmup > 0 and step <= warmup:
        return base_lr * step / max(1, warmup)
    # cosine from warmup..total
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


# ----------------------- train -----------------------
def train_level0(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True

    list_or_dir = args.data_list if Path(args.data_list).exists() else "data/coco/train2017"
    dl = make_loader(list_or_dir=list_or_dir, level=0,
                     batch_size=args.batch_size, num_workers=args.num_workers)

    # --- build model
    if args.mix_k > 0:
        model = MixtureLogisticHead(K=args.mix_k, hidden=args.hidden, num_blocks=args.blocks).to(device)
        model.is_mixture = True
        print(f"[info] using MixtureLogisticHead K={args.mix_k}, hidden={args.hidden}, blocks={args.blocks}")
    else:
        model = ResidualLogisticHead().to(device)
        model.is_mixture = False
        print("[info] using ResidualLogisticHead (K=1)")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # optional resume
    global_step = 0
    if args.resume and Path(args.resume).is_file():
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state)
        print(f"[resume] loaded weights from {args.resume}")

    ema = None
    ckpt_every = max(50, int(args.ckpt_every))
    it = iter(dl)

    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else max(100, args.steps // 20)

    for step in range(1, args.steps + 1):
        try:
            x0_uint8, y1_float = next(it)
        except StopIteration:
            it = iter(dl); x0_uint8, y1_float = next(it)

        xb = x0_uint8.to(device, non_blocking=True)
        y_up = upsample_to(y1_float, xb.shape[-2:]).to(device, non_blocking=True)

        if args.channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last)
            y_up = y_up.contiguous(memory_format=torch.channels_last)

        # --- set lr by schedule
        lr_t = lr_schedule(step, args.steps, args.lr, warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr_t

        with torch.amp.autocast('cuda', enabled=args.amp):
            if args.mix_k > 0:
                pi_logits, mu, log_s = model(y_up)
                logp, _, _ = mixture_loglik_and_resp(xb, pi_logits, mu, log_s) # [B,H,W]
                loss = -logp.mean()
            else:
                mean, log_s = model(y_up)                                      # [B,3,H,W]
                logp = discretized_logistic_logpmf(xb, mean, log_s)            # [B,H,W]
                loss = -logp.mean()

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt); scaler.update()

        loss_val = float(loss.detach().cpu())
        ema = loss_val if ema is None else (args.ema_momentum * ema + (1 - args.ema_momentum) * loss_val)

        if step % args.log_every == 0 or step == 1:
            print(f"[step {step:04d}] loss={loss_val:.4f}  ema={ema:.4f}  lr={lr_t:.2e}")

        if step % ckpt_every == 0 or step == args.steps:
            save_ckpt(model, args.save)

        global_step += 1

    print("[done] training finished.")


# ----------------------- cli -----------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Train Level-0 conditional model (x0 | y1)")
    # data
    ap.add_argument("--data_list", type=str, default="data/coco/train_list.txt",
                    help="真实图列表文件；若不存在则回退到 data/coco/train2017/")
    # training
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=4, help="L0 显存较大，默认更小的 batch")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_steps", type=int, default=0, help="0=自动取 steps/20，或手动指定")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    # model
    ap.add_argument("--mix_k", type=int, default=0, help="K=0 使用 K=1 旧头；K>0 使用 Logistic 混合头")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels_last", action="store_true", help="启用 NHWC 内存格式以加速卷积")
    # io
    ap.add_argument("--save", type=str, default="runs/level0_head_res.pt")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ema_momentum", type=float, default=0.9)
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train_level0(args)
