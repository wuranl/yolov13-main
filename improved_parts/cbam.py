"""CBAM 注意力模块（Convolutional Block Attention Module）。

这是一个可直接复用的 PyTorch 独立实现，适合在 YOLO 类模型的特征图上插入：
先做“通道注意力”（Channel Attention），再做“空间注意力”（Spatial Attention）。

使用示例：
    from cbam import CBAM
    x = torch.randn(1, 256, 80, 80)
    y = CBAM(256)(x)

输出形状与输入一致： (N, C, H, W) -> (N, C, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be > 0, got {reduction}")

        # MLP 的隐藏层维度（按 reduction 压缩），至少为 1
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 这里用 1x1 Conv 作为 MLP：C -> hidden -> C
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        # 平均池化/最大池化得到 (N, C, 1, 1)
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        # 通道注意力权重： (N, C, 1, 1)
        attn = self.act(avg + mx)
        # 按通道缩放输入特征
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            # CBAM paper uses 7; many codebases also allow 3.
            raise ValueError("kernel_size must be 3 or 7")

        # 空间注意力：用 (avg, max) 两张 1 通道特征图拼起来做卷积
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        # 在通道维做 avg/max，得到两张图： (N, 1, H, W)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接后 -> (N, 2, H, W)；卷积+sigmoid -> 空间权重 (N, 1, H, W)
        attn = self.act(self.conv(torch.cat([avg, mx], dim=1)))
        # 按空间位置缩放输入特征
        return x * attn


class CBAM(nn.Module):
    """CBAM block.

    Applies Channel Attention then Spatial Attention.

    Args:
        channels: input feature channels.
        reduction: channel reduction ratio in channel-attention MLP.
        spatial_kernel: kernel size for spatial attention conv (3 or 7).
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        # 先通道注意力、后空间注意力（论文默认顺序）
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        x = self.ca(x)
        x = self.sa(x)
        return x


if __name__ == "__main__":
    # Minimal smoke test
    x = torch.randn(2, 64, 32, 32)
    m = CBAM(64)
    y = m(x)
    print("OK", tuple(y.shape))
