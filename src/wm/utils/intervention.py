"""
Latent Intervention utilities.

- probe (linear) の重みから mass / friction 方向ベクトルを取り出す
- GRU hidden h に方向ベクトルを加算して介入する
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Literal, Optional

import torch


Axis = Literal["mass", "friction"]


@dataclass
class ProbeDirections:
    w_mass: torch.Tensor      # (H,)
    w_friction: torch.Tensor  # (H,)
    rep_mode: str
    use_force: bool
    in_dim: int


def load_probe_directions(probe_path: str | Path, device: torch.device) -> ProbeDirections:
    """
    outputs/probe_vpf.pt のような保存形式を想定:
      {
        "model": state_dict,
        "in_dim": 128,
        "out_dim": 2,
        "rep_mode": "mean",
        "use_force": True,
      }

    state_dict は {"lin.weight": (2,H), "lin.bias": (2,)} のはず。
    """
    probe_path = Path(probe_path)
    ck = torch.load(probe_path, map_location=device, weights_only=False)

    state: Dict[str, torch.Tensor] = ck["model"]
    if "lin.weight" not in state:
        raise KeyError(f"probe state_dict に lin.weight が見つかりません: keys={list(state.keys())}")

    W = state["lin.weight"].to(device)  # (2,H)
    if W.ndim != 2 or W.shape[0] != 2:
        raise ValueError(f"lin.weight の shape が想定外です: {tuple(W.shape)} (期待: (2,H))")

    w_mass = W[0]
    w_friction = W[1]

    in_dim = int(ck.get("in_dim", W.shape[1]))
    rep_mode = str(ck.get("rep_mode", "mean"))
    use_force = bool(ck.get("use_force", False))

    if W.shape[1] != in_dim:
        raise ValueError(f"in_dim が不一致です: ckpt={in_dim}, weight={W.shape[1]}")

    return ProbeDirections(
        w_mass=w_mass,
        w_friction=w_friction,
        rep_mode=rep_mode,
        use_force=use_force,
        in_dim=in_dim,
    )


def normalize_direction(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """方向ベクトルをL2正規化してスケール解釈しやすくする。"""
    return v / (torch.linalg.norm(v) + eps)


def apply_intervention(
    h: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    h: (B,H) or (H,)
    direction: (H,)
    """
    if h.ndim == 1:
        return h + alpha * direction
    return h + alpha * direction.view(1, -1)
