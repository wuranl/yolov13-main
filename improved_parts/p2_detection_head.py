"""P2 Detection Head: 含 P2 小目标检测层的 YOLO 风格检测头。

用途：
- 输入四层特征 P2/P3/P4/P5（高到低分辨率）
- 每层输出 anchor-free 风格预测：(bbox 4 + obj 1 + cls nc)
- 适合远距离小目标场景（P2 对小目标更友好）

说明：
- 该实现是独立模块，不直接替换 ultralytics 内置 Detect
- 你可先用于实验验证，再按需要接入模型构建流程
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, g: int = 1) -> None:
        super().__init__()
        p = (k - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoupledHead(nn.Module):
    """单层 decoupled head：分类支路 + 回归支路。"""

    def __init__(self, in_ch: int, num_classes: int, hidden_ch: int = 128) -> None:
        super().__init__()
        self.cls_stem = nn.Sequential(
            ConvBNAct(in_ch, hidden_ch, 3),
            ConvBNAct(hidden_ch, hidden_ch, 3),
        )
        self.reg_stem = nn.Sequential(
            ConvBNAct(in_ch, hidden_ch, 3),
            ConvBNAct(hidden_ch, hidden_ch, 3),
        )

        self.cls_pred = nn.Conv2d(hidden_ch, num_classes, 1)
        self.obj_pred = nn.Conv2d(hidden_ch, 1, 1)
        self.box_pred = nn.Conv2d(hidden_ch, 4, 1)  # l,t,r,b 或 dx,dy,dw,dh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_feat = self.cls_stem(x)
        reg_feat = self.reg_stem(x)

        cls_logit = self.cls_pred(cls_feat)
        obj_logit = self.obj_pred(reg_feat)
        box = self.box_pred(reg_feat)

        # 输出通道格式: [box4, obj1, cls_nc]
        out = torch.cat([box, obj_logit, cls_logit], dim=1)
        return out


class P2DetectionHead(nn.Module):
    """P2/P3/P4/P5 四层检测头。

    Args:
        in_channels: 四层输入通道，例如 (128, 256, 512, 512)
        num_classes: 类别数（你的项目可设为2：car/truck）
        hidden_ch: decoupled head 中间通道

    Inputs:
        feats: [P2, P3, P4, P5]，每个形状 (B, C_i, H_i, W_i)

    Returns:
        list[Tensor]: 每层预测，形状 (B, 5+nc, H_i, W_i)
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 256, 512, 512),
        num_classes: int = 2,
        hidden_ch: int = 128,
    ) -> None:
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError("in_channels must have 4 items for [P2,P3,P4,P5]")
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")

        self.num_classes = num_classes
        self.out_channels = 5 + num_classes

        self.heads = nn.ModuleList([
            DecoupledHead(c, num_classes=num_classes, hidden_ch=hidden_ch) for c in in_channels
        ])

    def forward(self, feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(feats) != 4:
            raise ValueError(f"expected 4 feature maps [P2,P3,P4,P5], got {len(feats)}")

        outs = [head(x) for head, x in zip(self.heads, feats)]
        return outs


if __name__ == "__main__":
    p2 = torch.randn(2, 128, 160, 160)
    p3 = torch.randn(2, 256, 80, 80)
    p4 = torch.randn(2, 512, 40, 40)
    p5 = torch.randn(2, 512, 20, 20)

    m = P2DetectionHead(in_channels=(128, 256, 512, 512), num_classes=2, hidden_ch=128)
    ys = m([p2, p3, p4, p5])
    print("P2 Head OK:", [tuple(y.shape) for y in ys])
