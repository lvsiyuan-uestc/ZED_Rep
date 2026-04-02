from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from PIL import Image

SCALE_PRESETS = {
    "uint8": (1.0, 0.0),
    "unit": (1.0 / 255.0, 0.0),
    "tanh": (1.0 / 127.5, -1.0),
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(path: str) -> Iterable[str]:
    """Yield image paths from a directory, file, or newline-delimited manifest."""
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                yield str(f)
        return

    if p.suffix.lower() in IMG_EXTS:
        yield str(p)
        return

    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def apply_affine_all(pyr: Dict[str, torch.Tensor], a: float, b: float) -> Dict[str, torch.Tensor]:
    """Apply x -> a*x+b for all tensors in pyramid dict."""
    return {k: (v.to(torch.float32) * a + b) for k, v in pyr.items()}


def apply_affine_to_x_only(pyr: Dict[str, torch.Tensor], a: float, b: float) -> Dict[str, torch.Tensor]:
    """Apply x -> a*x+b only on x* entries, preserving y* entries."""
    out = dict(pyr)
    for k, v in pyr.items():
        if k.startswith("x"):
            out[k] = v.to(torch.float32) * a + b
    return out


def to_uint8_gray(arr: torch.Tensor, robust: float = 1.0) -> Image.Image:
    """Normalize a 2D tensor to uint8 grayscale for quick diagnostics."""
    x = arr.detach().cpu().float().numpy()
    if not np.isfinite(x).any():
        x = np.zeros_like(x, dtype=np.float32)

    if 0.5 < robust < 1.0:
        lo = float(np.quantile(x, 1.0 - robust))
        hi = float(np.quantile(x, robust))
    else:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))

    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6

    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return Image.fromarray((y * 255.0 + 0.5).astype(np.uint8), mode="L")
