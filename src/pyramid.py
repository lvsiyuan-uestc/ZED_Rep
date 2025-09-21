# src/pyramid.py

import torch
import torch.nn.functional as F

def avg_pool2x2_round_uint8(x: torch.Tensor) -> torch.Tensor:
    """
    What: 2x2 平均池化 + 四舍五入（保持像素值语义），输入/输出都在 CPU 上的 uint8。
    Why : 模拟 SReC 的逐层量化流程；uint8 存储更省内存与更贴近像素域。
    """
    assert x.dtype == torch.uint8 and x.dim() == 4
    y = F.avg_pool2d(x.float(), kernel_size=2, stride=2)
    y = torch.clamp(torch.round(y), 0, 255).to(torch.uint8)
    return y

def build_x_pyramid(x0_chw_uint8: torch.Tensor, device=None):
    """
    What: 构建 x0..x3（uint8）分辨率金字塔，**一律在 CPU** 返回。
    Why : Dataset 阶段保持 CPU，便于 DataLoader pin_memory 与批量搬运到 GPU。
    """
    # 默认改为 CPU
    if device is None:
        device = torch.device("cpu")
    # 确保输入先在 CPU
    x0 = x0_chw_uint8.unsqueeze(0).to("cpu")  # [1,C,H,W] uint8 on CPU
    x1 = avg_pool2x2_round_uint8(x0)
    x2 = avg_pool2x2_round_uint8(x1)
    x3 = avg_pool2x2_round_uint8(x2)
    return {"x0": x0.squeeze(0), "x1": x1.squeeze(0), "x2": x2.squeeze(0), "x3": x3.squeeze(0)}

def build_xy_pyramid(x0_chw_uint8: torch.Tensor, device=None):
    """
    What: 同时构建 x0..x3(uint8) 与 y1..y3(float32)，**全部在 CPU**。
    Why : 与 DataLoader 的工作方式一致；训练循环再决定何时搬到 GPU。
    """
    if device is None:
        device = torch.device("cpu")
    x0 = x0_chw_uint8.unsqueeze(0).to("cpu")  # [1,C,H,W] uint8 on CPU

    y1 = F.avg_pool2d(x0.float(), 2, 2)                 # float on CPU
    x1 = torch.clamp(torch.round(y1), 0, 255).to(torch.uint8)

    y2 = F.avg_pool2d(x1.float(), 2, 2)
    x2 = torch.clamp(torch.round(y2), 0, 255).to(torch.uint8)

    y3 = F.avg_pool2d(x2.float(), 2, 2)
    x3 = torch.clamp(torch.round(y3), 0, 255).to(torch.uint8)

    out = {
        "x0": x0.squeeze(0), "x1": x1.squeeze(0), "x2": x2.squeeze(0), "x3": x3.squeeze(0),
        "y1": y1.squeeze(0).to(torch.float32),
        "y2": y2.squeeze(0).to(torch.float32),
        "y3": y3.squeeze(0).to(torch.float32),
    }
    return out
