"""AKConv 自适应卷积核注意力模块（Adaptive Kernel Convolution）。

这是一个“多分支不同卷积核 + 注意力加权融合”的实用实现，思想上接近
Selective Kernel / Dynamic Kernel 这一类结构：
1) 多个卷积分支（kernel size 不同）并行提取特征
2) 用一个轻量门控网络（gate）根据输入特征预测每个分支的重要性权重
3) 对各分支输出做加权求和得到最终输出

你可以把它当作一个“带注意力的卷积层”，用来替换单个 Conv 模块。

使用示例：
        from akconv import AKConv
        x = torch.randn(1, 256, 80, 80)
        y = AKConv(256, 256, kernels=(3, 5, 7))(x)

形状： (N, C_in, H, W) -> (N, C_out, H/stride, W/stride)

注意：
- 本文件是独立模块（不自动注册到 ultralytics YAML 解析器）。
- 如果要在 ultralytics 的模型 YAML 里直接写 `AKConv`，通常需要把模块放到
    ultralytics 包内并完成注册/解析（需要我也可以继续帮你接入）。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def _autopad(kernel_size: int, dilation: int = 1) -> int:
    # 对奇数卷积核做 same padding（保持特征图尺寸不变，stride=1 时）
    return (kernel_size - 1) // 2 * dilation


class AKConv(nn.Module):
    """Adaptive Kernel Convolution with attention over multiple kernel branches.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        kernels: kernel sizes for each branch (odd ints recommended, e.g. (3,5,7)).
        stride: stride for all branches.
        dilation: dilation for all branches.
        groups: groups for branch convolutions (1 for normal conv, in_channels for depthwise).
        reduction: reduction ratio for the gating MLP.
        min_hidden: minimum hidden size in gating MLP.
        norm: whether to use BatchNorm after each branch conv.
        act: activation function (default SiLU).

    Shape:
        input: (N, C_in, H, W)
        output: (N, C_out, H/stride, W/stride)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: Sequence[int] = (3, 5),
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        reduction: int = 16,
        min_hidden: int = 32,
        norm: bool = True,
        act: nn.Module | None = None,
    ):
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels/out_channels must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if dilation <= 0:
            raise ValueError("dilation must be > 0")
        if groups <= 0:
            raise ValueError("groups must be > 0")
        if not kernels:
            raise ValueError("kernels must be non-empty")

        kernels = tuple(int(k) for k in kernels)
        if any(k <= 0 for k in kernels):
            raise ValueError(f"invalid kernels: {kernels}")
        if any(k % 2 == 0 for k in kernels):
            # keep it simple + predictable padding
            raise ValueError(f"kernels must be odd for same-padding, got {kernels}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernels = kernels
        self.num_branches = len(kernels)

        # 默认激活函数使用 SiLU（与 YOLO 系列常用一致）
        if act is None:
            act = nn.SiLU(inplace=True)

        # 1) 多分支卷积：每个分支 kernel size 不同
        branches: list[nn.Module] = []
        for k in kernels:
            padding = _autopad(k, dilation=dilation)
            layers: list[nn.Module] = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=not norm,
                )
            ]
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(act)
            branches.append(nn.Sequential(*layers))
        self.branches = nn.ModuleList(branches)

        # 2) 门控网络：从融合特征 u 生成每个分支的权重
        # hidden 为门控 MLP 的隐藏维度
        hidden = max(min_hidden, out_channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.num_branches, kernel_size=1, bias=True),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C_in, H, W)
        # 分支输出：每个分支都是 (N, C_out, H', W')
        outs = [b(x) for b in self.branches]

        # fused: (N, B, C, H', W')
        fused = torch.stack(outs, dim=1)
        # u: 作为生成权重的“融合特征” (N, C, H', W')
        u = fused.sum(dim=1)

        # 全局平均池化：s = (N, C, 1, 1)
        s = self.pool(u)
        # gate 输出分支 logits：(N, B, 1, 1)，softmax 后得到权重
        a = self.softmax(self.gate(s))
        # 变形方便与 fused 广播相乘： (N, B, 1, 1, 1)
        a = a.unsqueeze(2)

        # 3) 分支加权求和：y = Σ a_b * out_b
        y = (fused * a).sum(dim=1)
        return y


if __name__ == "__main__":
    # Minimal smoke test
    x = torch.randn(2, 64, 32, 32)
    m = AKConv(64, 64, kernels=(3, 5, 7))
    y = m(x)
    print("OK", tuple(y.shape))
