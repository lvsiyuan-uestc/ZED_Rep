#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_level1.py — 训练 Level-1 条件像素模型 (x1 | y2)

"""
from __future__ import annotations
import argparse
from pathlib import Path
import random, numpy as np
import torch
from torch.utils.data import DataLoader

# project modules
from src.dset_realpairs import RealPairsDataset
from src.model_head import ResidualLogisticHead
from src.model_head_mix import MixtureLogisticHead
from src.stats import upsample_to, discretized_logistic_logpmf
from src.stats_mix import mixture_loglik_and_resp

# ----------------------- utils -----------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id); random.seed(seed + worker_id)

def make_loader(list_or_dir: str, level: int = 1, batch_size: int = 8, num_workers: int = 4):
    ds = RealPairsDataset(list_or_dir=list_or_dir, level=level, jpeg_q=(80,100), short_range=(512,896))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

def save_ckpt(model: torch.nn.Module, path: str):
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[ckpt] saved to {path}")

def set_lr(optim: torch.optim.Optimizer, lr: float):
    for pg in optim.param_groups: pg["lr"] = lr

# ----------------------- train -----------------------
def train_level1(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True

    list_or_dir = args.data_list if Path(args.data_list).exists() else "data/coco/train2017"
    dl = make_loader(list_or_dir=list_or_dir, level=1, batch_size=args.batch_size, num_workers=args.num_workers)

    # model
    if args.mix_k > 0:
        model = MixtureLogisticHead(K=args.mix_k, hidden=args.hidden, num_blocks=args.blocks).to(device)
        model.is_mixture = True
        print(f"[info] using MixtureLogisticHead K={args.mix_k}, hidden={args.hidden}, blocks={args.blocks}")
    else:
        model = ResidualLogisticHead().to(device)
        print("[info] using ResidualLogisticHead (K=1)")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # optim & scaler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # optional resume
    if args.resume and Path(args.resume).is_file():
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state)
        print(f"[resume] loaded weights from {args.resume}")

    ema = None
    it = iter(dl)
    ckpt_every = max(50, int(args.ckpt_every))

    for step in range(1, args.steps + 1):
        # --------- warmup LR ---------
        if step <= args.warmup:
            set_lr(opt, args.lr * step / max(1, args.warmup))
        else:
            set_lr(opt, args.lr)

        # --------- batch ---------
        try:
            x1_uint8, y2_float = next(it)
        except StopIteration:
            it = iter(dl); x1_uint8, y2_float = next(it)

        xb = x1_uint8.to(device, non_blocking=True)
        y_up = upsample_to(y2_float, xb.shape[-2:]).to(device, non_blocking=True)

        if args.channels_last:
            xb = xb.to(memory_format=torch.channels_last)
            y_up = y_up.to(memory_format=torch.channels_last)

        # --------- forward ---------
        with torch.amp.autocast('cuda', enabled=args.amp):
            if args.mix_k > 0:
                pi_logits, mu, log_s = model(y_up)
                logp, _, _ = mixture_loglik_and_resp(xb, pi_logits, mu, log_s)  # [B,H,W]
                loss = -logp.mean()
            else:
                mean, log_s = model(y_up)
                logp = discretized_logistic_logpmf(xb, mean, log_s)
                loss = -logp.mean()

        # --------- backward ---------
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # AMP 正确顺序：unscale → clip → step
        scaler.unscale_(opt)
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(opt)
        scaler.update()

        # --------- log / ckpt ---------
        loss_val = float(loss.detach().cpu())
        ema = loss_val if ema is None else (args.ema_momentum * ema + (1 - args.ema_momentum) * loss_val)
        if step % args.log_every == 0 or step == 1:
            print(f"[step {step:04d}] loss={loss_val:.4f}  ema={ema:.4f}  lr={opt.param_groups[0]['lr']:.2e}")
        if step % ckpt_every == 0 or step == args.steps:
            save_ckpt(model, args.save)

    print("[done] training finished.")

def build_argparser():
    ap = argparse.ArgumentParser(description="Train Level-1 conditional model (x1 | y2)")
    # data
    ap.add_argument("--data_list", type=str, default="data/coco/train_list.txt",
                    help="真实图列表文件；若不存在则回退到 data/coco/train2017/")
    # training
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500, help="线性 warmup 的步数")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--channels_last", action="store_true", help="启用 NHWC 内存格式（建议配合 AMP）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    # model
    ap.add_argument("--mix_k", type=int, default=0, help="K=0 使用 K=1 旧头；K>0 使用 Logistic 混合头")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    # io
    ap.add_argument("--save", type=str, default="runs/level1_head_res.pt")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ema_momentum", type=float, default=0.9)
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train_level1(args)
