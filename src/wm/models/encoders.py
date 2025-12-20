"""観測と自己受容感覚のエンコーダを定義する。"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ImageEncoderCNN(nn.Module):
    """RGB画像列から埋め込みを計算する小型CNN。"""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgbを入力として (B, T, E) の埋め込みを返す。"""
        rgb, has_batch = _ensure_batch_time(rgb)
        b, t, c, h, w = rgb.shape
        x = rgb.reshape(b * t, c, h, w)
        x = self.conv(x).reshape(b * t, -1)
        x = self.proj(x)
        x = x.reshape(b, t, -1)
        if not has_batch:
            x = x.squeeze(0)
        return x


class ProprioEncoderMLP(nn.Module):
    """関節状態とブロック姿勢から埋め込みを計算するMLP。"""

    def __init__(self, input_dim: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, q: torch.Tensor, dq: torch.Tensor, block_pose: torch.Tensor) -> torch.Tensor:
        """q, dq, block_poseを連結して埋め込みを返す。"""
        q, has_batch = _ensure_batch_time(q)
        dq, _ = _ensure_batch_time(dq)
        block_pose, _ = _ensure_batch_time(block_pose)
        x = torch.cat([q, dq, block_pose], dim=-1)
        b, t, d = x.shape
        x = x.reshape(b * t, d)
        x = self.mlp(x).reshape(b, t, -1)
        if not has_batch:
            x = x.squeeze(0)
        return x

class ForceEncoderMLP(nn.Module):
    """力覚（関節力など）を埋め込みに変換する。"""

    def __init__(self, input_dim: int, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: (B, T, J)
        b, t, j = f.shape
        f = f.view(b * t, j)
        out = self.net(f)
        return out.view(b, t, -1)

def concat_embeddings(*embeddings: torch.Tensor) -> torch.Tensor:
    """
    時系列埋め込みを最後の次元で結合する。
    すべて (B, T, D_i) を想定。
    """
    if len(embeddings) < 2:
        raise ValueError("concat_embeddings requires at least two tensors")
    return torch.cat(list(embeddings), dim=-1)

def _ensure_batch_time(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """(T, ...) を (1, T, ...) に揃える。"""
    if x.dim() in {2, 4}:
        return x.unsqueeze(0), False
    return x, True
