import os
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import torch

def ensure_dir(p:dir):
    Path(p).mkdir(parents=True, exist_ok=True)

def to_multiple_of(v:int,m:int)->int:
    return v-(v%m)

def load_image_uint8(path: str, require_divisible_by: int = 8, max_long_side: int | None = None) -> torch.Tensor:
    """
      读取任意格式图片；可选对最长边做等比缩放（保 8 的倍数）；
      处理调色板/透明通道，统一转 RGB。
    """
    img = Image.open(path)
    # 透明/调色板 → RGBA → 白底合成 → RGB
    if img.mode in ("RGBA",) or ("transparency" in img.info) or img.mode == "P":
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))  # 白底
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")

    # 最长边缩放（例如 1536）
    if max_long_side is not None:
        w, h = img.size
        m = max(w, h)
        if m > max_long_side:
            scale = max_long_side / m
            nw = int(round(w * scale))
            nh = int(round(h * scale))
            # 调整到 8 的倍数，避免后续金字塔裁剪
            nw = max(8, (nw // require_divisible_by) * require_divisible_by)
            nh = max(8, (nh // require_divisible_by) * require_divisible_by)
            img = img.resize((nw, nh), Image.LANCZOS)

    # 若仍非 8 的倍数，做最小幅度居中裁剪
    w, h = img.size
    if require_divisible_by:
        nw = (w // require_divisible_by) * require_divisible_by
        nh = (h // require_divisible_by) * require_divisible_by
        if nw != w or nh != h:
            left = (w - nw) // 2
            top  = (h - nh) // 2
            img = img.crop((left, top, left + nw, top + nh))

    arr = np.array(img, dtype=np.uint8, copy=True)         # [H,W,3]
    ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]
    return ten

def save_image_uint8(t: torch.Tensor, path: str):
    t = t.detach().cpu()
    if t.dtype != torch.uint8:
        t = t.clamp(0,255).round().to(torch.uint8)
    arr = t.permute(1,2,0).numpy()   # [H,W,C]
    Image.fromarray(arr).save(path)
