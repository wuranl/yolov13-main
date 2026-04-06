"""IA-Net: 轻量图像自适应增强模块（Image-Adaptive Enhancement Network）。

设计目标：
1) 面向雨/雾/夜等退化图像，做“对检测友好”的自适应增强
2) 计算开销小，便于插入 YOLO 系列 backbone 前端
3) 端到端可训练（增强参数由检测损失反向驱动）

核心思路：
- 用全局上下文预测每通道增强参数（alpha/beta/gamma）
- 对输入做曲线映射：y = alpha * x^gamma + beta
- 再接一组深度可分离卷积做细节修复，并与输入残差融合

注意：
- 本实现是独立模块，不自动注册到 ultralytics YAML 解析器
- 默认输入范围假设为 [0,1]；若你的输入是 [0,255]，请先归一化
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DWSeparableConv(nn.Module):
    """Depthwise Separable Conv: DW 3x3 + PW 1x1."""

    def __init__(self, channels: int, act: nn.Module | None = None) -> None:
        super().__init__()
        if act is None:
            act = nn.SiLU(inplace=True)

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            act,
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IANet(nn.Module):
    """Image-Adaptive Enhancement Network.

    Args:
        channels: 输入/输出通道数
        hidden_ratio: 参数预测分支的缩减比例
        min_hidden: 参数预测分支最小隐藏通道
        refine_layers: 细节修复深度可分离卷积层数
        clamp_output: 是否将输出钳制到 [0,1]

    Input/Output:
        x: (B, C, H, W) in [0,1]
        y: (B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        hidden_ratio: int = 8,
        min_hidden: int = 16,
        refine_layers: int = 2,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be > 0")
        if refine_layers <= 0:
            raise ValueError("refine_layers must be > 0")

        hidden = max(min_hidden, channels // hidden_ratio)
        self.clamp_output = clamp_output

        # 全局上下文 -> 每通道增强参数(alpha, beta, gamma)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.param_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 3, 1, bias=True),
        )

        # 细节修复分支
        refine = [DWSeparableConv(channels) for _ in range(refine_layers)]
        self.refine = nn.Sequential(*refine)
        self.fuse = nn.Conv2d(channels, channels, 1, bias=False)

    @staticmethod
    def _split_params(params: torch.Tensor, c: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # params: (B, 3C, 1, 1)
        alpha, beta, gamma = torch.split(params, c, dim=1)
        # 约束范围：
        # alpha in [0.5, 1.5]，beta in [-0.1, 0.1]，gamma in [0.6, 1.8]
        alpha = 0.5 + torch.sigmoid(alpha)
        beta = 0.2 * torch.tanh(beta)
        gamma = 0.6 + 1.2 * torch.sigmoid(gamma)
        return alpha, beta, gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected 4D input, got shape={tuple(x.shape)}")

        b, c, _, _ = x.shape
        # 保障幂运算稳定性
        x01 = x.clamp(0.0, 1.0)

        params = self.param_mlp(self.pool(x01))
        alpha, beta, gamma = self._split_params(params, c)

        # 曲线增强
        enhanced = alpha * torch.pow(x01 + 1e-6, gamma) + beta
        enhanced = enhanced.clamp(0.0, 1.0)

        # 细节修复 + 残差融合
        detail = self.refine(enhanced)
        y = enhanced + self.fuse(detail)

        # 与原图做轻量残差混合，避免过增强
        y = 0.8 * y + 0.2 * x01

        if self.clamp_output:
            y = y.clamp(0.0, 1.0)
        return y


if __name__ == "__main__":
    x = torch.rand(2, 3, 320, 320)
    m = IANet(channels=3, refine_layers=2)
    y = m(x)
    print("IA-Net OK:", tuple(y.shape), float(y.min()), float(y.max()))
