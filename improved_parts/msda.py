"""MSDA: 多尺度可变形注意力（Multi-Scale Deformable Attention）2D 实现。

用途：
- 适合插在 Neck 的多尺度特征融合阶段
- 输入 query 特征 + 多尺度 value 特征，输出同分辨率融合特征

机制要点：
1) 对每个 query 位置预测 K 个可学习采样偏移（offset）
2) 在每个尺度特征图上按偏移位置采样（grid_sample）
3) 用注意力权重对多尺度多点采样结果加权汇聚

注意：
- 本实现偏工程可读性，便于你复试讲解与二次改造
- 非官方 Deformable DETR 原版实现，接口更贴近 YOLO neck 融合
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDeformableAttention2D(nn.Module):
    """2D Multi-Scale Deformable Attention.

    Args:
        channels: query/value 通道数
        num_levels: 多尺度层数（例如 P2~P5 => 4）
        num_heads: 注意力头数
        num_points: 每个 head 在每个 level 的采样点数
        offset_scale: 偏移缩放（归一化坐标尺度）

    Inputs:
        query: (B, C, H, W)
        feats: list of length num_levels, each (B, C, H_l, W_l)

    Output:
        out: (B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        num_levels: int = 4,
        num_heads: int = 8,
        num_points: int = 4,
        offset_scale: float = 0.25,
    ) -> None:
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if num_levels <= 0 or num_heads <= 0 or num_points <= 0:
            raise ValueError("num_levels/num_heads/num_points must be > 0")

        self.channels = channels
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = channels // num_heads
        self.offset_scale = offset_scale

        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj = nn.ModuleList([nn.Conv2d(channels, channels, 1, bias=False) for _ in range(num_levels)])
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)

        # 从 query 预测偏移与注意力 logits
        self.offset_conv = nn.Conv2d(
            channels,
            num_heads * num_levels * num_points * 2,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.attn_conv = nn.Conv2d(
            channels,
            num_heads * num_levels * num_points,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    @staticmethod
    def _reference_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # 归一化网格，范围 [-1,1]，shape=(1,1,H,W,2)
        ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).unsqueeze(0)
        return grid

    def forward(self, query: torch.Tensor, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feats) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} feature levels, got {len(feats)}")

        b, c, hq, wq = query.shape
        q = self.q_proj(query)

        # offsets: (B, heads, levels, points, 2, H, W)
        offsets = self.offset_conv(q)
        offsets = offsets.view(
            b,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
            hq,
            wq,
        )
        offsets = torch.tanh(offsets) * self.offset_scale

        # attn: (B, heads, levels, points, H, W)
        attn_logits = self.attn_conv(q)
        attn_logits = attn_logits.view(
            b,
            self.num_heads,
            self.num_levels,
            self.num_points,
            hq,
            wq,
        )
        attn = F.softmax(attn_logits.flatten(2, 3), dim=2).view_as(attn_logits)

        ref = self._reference_grid(hq, wq, query.device, query.dtype)  # (1,1,H,W,2)

        out = torch.zeros(b, self.num_heads, self.head_dim, hq, wq, device=query.device, dtype=query.dtype)

        for lvl, feat in enumerate(feats):
            if feat.shape[0] != b or feat.shape[1] != c:
                raise ValueError(
                    f"feature level {lvl} shape mismatch: expected (B={b},C={c},H,W), got {tuple(feat.shape)}"
                )

            v = self.v_proj[lvl](feat)
            # (B, heads, head_dim, H_l, W_l)
            v = v.view(b, self.num_heads, self.head_dim, feat.shape[2], feat.shape[3])

            # 变成 grid_sample 需要的批次维度
            v = v.reshape(b * self.num_heads, self.head_dim, feat.shape[2], feat.shape[3])

            for p in range(self.num_points):
                # (B, heads, H, W, 2)
                off = offsets[:, :, lvl, p].permute(0, 1, 3, 4, 2)
                grid = ref + off

                grid = grid.reshape(b * self.num_heads, hq, wq, 2)
                sampled = F.grid_sample(
                    v,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                sampled = sampled.view(b, self.num_heads, self.head_dim, hq, wq)

                w = attn[:, :, lvl, p].unsqueeze(2)  # (B, heads, 1, H, W)
                out = out + sampled * w

        out = out.reshape(b, c, hq, wq)
        out = self.out_proj(out)
        out = self.act(self.norm(out + query))
        return out


if __name__ == "__main__":
    x = torch.randn(2, 256, 80, 80)
    feats = [
        torch.randn(2, 256, 80, 80),
        torch.randn(2, 256, 40, 40),
        torch.randn(2, 256, 20, 20),
        torch.randn(2, 256, 10, 10),
    ]
    m = MultiScaleDeformableAttention2D(channels=256, num_levels=4, num_heads=8, num_points=4)
    y = m(x, feats)
    print("MSDA OK:", tuple(y.shape))
