"""隠れ状態から状態を復元するデコーダを定義する。"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class StateDecoder(nn.Module):
    """隠れ状態から q, dq, block_pose を予測するデコーダ。"""

    def __init__(self, hidden_dim: int, j_dim: int, block_dim: int = 7) -> None:
        super().__init__()
        self.j_dim = j_dim
        self.block_dim = block_dim
        out_dim = j_dim * 2 + block_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """隠れ状態から (q, dq, block_pose) を出力する。"""
        b, t, d = h.shape
        x = h.reshape(b * t, d)
        x = self.mlp(x)
        x = x.reshape(b, t, -1)
        q_hat = x[..., : self.j_dim]
        dq_hat = x[..., self.j_dim : self.j_dim * 2]
        block_hat = x[..., self.j_dim * 2 :]
        return q_hat, dq_hat, block_hat
