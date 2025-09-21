# src/stats.py
import torch
import torch.nn.functional as F

EPS = 1e-12

def upsample_to(x: torch.Tensor, target_hw, mode: str = "nearest") -> torch.Tensor:
    """
    (What) 将张量上采样到统一的 (H,W)。
    (Why) 预测当前层像素分布时，需要把下一层 y^{l+1} 对齐到当前分辨率。
    支持:
      - x.shape == [C,h,w]  -> 返回 [C,H,W]
      - x.shape == [B,C,h,w]-> 返回 [B,C,H,W]
    """
    need_squeeze = False
    if x.dim() == 3:
        x = x.unsqueeze(0)   # [1,C,h,w]
        need_squeeze = True
    elif x.dim() != 4:
        raise ValueError(f"upsample_to expects 3D or 4D, got {x.shape}")

    if mode in ("bilinear", "bicubic"):
        x = F.interpolate(x, size=target_hw, mode=mode, align_corners=False)
    else:
        x = F.interpolate(x, size=target_hw, mode=mode)

    if need_squeeze:
        x = x.squeeze(0)     # 回到 [C,H,W]
    return x.contiguous()

def discretized_logistic_logpmf(
    x_uint8: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor
) -> torch.Tensor:
    """
    计算离散logistic在像素箱 [-0.5,+0.5] 上的 log PMF。
    """
    x = x_uint8.float()
    s = torch.exp(log_scale)
    upper = (x + 0.5 - mean) / s
    lower = (x - 0.5 - mean) / s
    # logistic 的 CDF 是 sigmoid
    p = torch.sigmoid(upper) - torch.sigmoid(lower)
    p = torch.clamp(p, min=EPS)
    return torch.log(p)

def nll_h_maps_single_logistic(
    x_uint8: torch.Tensor, y_low: torch.Tensor, log_scale_const: float = 1.5
):
    """
    单张图的最小演示：用常数尺度+ y上采样的均值，生成 NLL/H/D 地图。

    """
    assert x_uint8.dim() == 3 and y_low.dim() == 3, "this demo fn is single-image only"
    C, H, W = x_uint8.shape
    device = x_uint8.device
    mean = upsample_to(y_low.to(device), (H, W))              # [C,H,W], float
    log_scale = torch.full_like(mean, float(log_scale_const)) # [C,H,W]

    logp = discretized_logistic_logpmf(x_uint8, mean, log_scale)  # [C,H,W]
    nll_map = (-logp).mean(dim=0)                            # [H,W]
    h_channel = (log_scale + 2.0).mean(dim=0)               # [H,W] 近似熵
    d_map = nll_map - h_channel
    return nll_map, h_channel, d_map
