"""
latent系列 h_seq から固定次元の表現 h_rep を作る（probe用）。
"""

from __future__ import annotations
from typing import Literal
import torch


def summarize_latent(
    h_seq: torch.Tensor,
    mode: Literal["last", "mean"] = "mean",
) -> torch.Tensor:
    """
    Args:
        h_seq: (B, T, H) もしくは (T, H)
        mode:
          - "last": 最終ステップ h_{T-1}
          - "mean": 時間平均

    Returns:
        h_rep: (B, H)
    """
    if h_seq.dim() == 2:
        h_seq = h_seq.unsqueeze(0)  # (1,T,H)

    if mode == "last":
        return h_seq[:, -1]
    if mode == "mean":
        return h_seq.mean(dim=1)

    raise ValueError(f"Unknown mode: {mode}")
