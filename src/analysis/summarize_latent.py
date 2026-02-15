# src/analysis/summarize_latent.py
from __future__ import annotations
import torch

@torch.no_grad()
def summarize_latent(
    h_seq: torch.Tensor,
    mode: str = "mean",
    mask_bt: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Summarize latent sequence (B,T,H) -> representative (B,H).

    Args:
      h_seq: (B,T,H)
      mode: "mean" or "last"
      mask_bt: (B,T) bool or 0/1. If provided, summarize only masked timesteps.
               If a batch item has no True in mask, fallback to unmasked mode.
    """
    assert h_seq.dim() == 3, f"h_seq must be (B,T,H), got {tuple(h_seq.shape)}"
    B, T, H = h_seq.shape

    if mask_bt is None:
        if mode == "mean":
            return h_seq.mean(dim=1)
        if mode == "last":
            return h_seq[:, -1, :]
        raise ValueError(f"unknown mode: {mode}")

    m = mask_bt
    if m.dtype != torch.bool:
        m = m > 0.5
    has_any = m.any(dim=1)  # (B,)

    if mode == "mean":
        mf = m.to(h_seq.dtype).unsqueeze(-1)  # (B,T,1)
        denom = mf.sum(dim=1).clamp_min(1.0)  # (B,1)
        out = (h_seq * mf).sum(dim=1) / denom
        fallback = h_seq.mean(dim=1)
        out = torch.where(has_any.unsqueeze(-1), out, fallback)
        return out

    if mode == "last":
        # last timestep where mask is True; fallback to T-1 if empty
        idx = torch.arange(T, device=h_seq.device).unsqueeze(0).expand(B, T)  # (B,T)
        idx = idx.masked_fill(~m, -1)
        last_idx = idx.max(dim=1).values
        last_idx = torch.where(has_any, last_idx, torch.full_like(last_idx, T - 1))
        return h_seq[torch.arange(B, device=h_seq.device), last_idx, :]

    raise ValueError(f"unknown mode: {mode}")
