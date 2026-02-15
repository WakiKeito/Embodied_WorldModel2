"""GRUベースのダイナミクスモデルを定義する。"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class GRUDynamics(nn.Module):
    """埋め込みと行動から隠れ状態を更新するGRUダイナミクス。"""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

    def forward_step(self, x_t: torch.Tensor, h_prev: Optional[torch.Tensor]) -> torch.Tensor:
        """1ステップ分の隠れ状態更新を行う。"""
        if h_prev is None:
            h_prev = torch.zeros(x_t.shape[0], self.hidden_dim, device=x_t.device)
        return self.gru_cell(x_t, h_prev)
