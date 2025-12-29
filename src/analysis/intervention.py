"""
Latent Intervention utilities.

- probe（線形）の重み W を direction として使う
- rollout中の hidden h_t に h_t += alpha * direction を入れる
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class InterventionSpec:
    kind: str               # "mass" or "friction"
    alpha: float            # intervention strength
    normalize: bool = True  # directionを正規化するか
    apply_each_step: bool = True  # 毎step適用か（True推奨）


def extract_probe_directions(probe_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    LinearProbe(nn.Linear) の state_dict から direction を取り出す。
    想定:
      probe.lin.weight: (2, H)  0->mass, 1->friction
      probe.lin.bias:   (2,)
    """
    # どちらのキー形式でも対応
    w_key = None
    for k in ["lin.weight", "probe.lin.weight"]:
        if k in probe_state:
            w_key = k
            break
    if w_key is None:
        # state_dict が {"probe": {...}} の場合
        if "probe" in probe_state and isinstance(probe_state["probe"], dict):
            return extract_probe_directions(probe_state["probe"])
        raise KeyError("probe state_dict に lin.weight が見つかりません")

    W = probe_state[w_key]  # (2,H)
    if W.dim() != 2 or W.size(0) != 2:
        raise ValueError(f"lin.weight shape expected (2,H), got {tuple(W.shape)}")

    return {
        "mass": W[0].detach().clone(),
        "friction": W[1].detach().clone(),
    }


def normalize_direction(d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return d / (torch.norm(d) + eps)


@torch.no_grad()
def apply_intervention(
    h: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
    normalize: bool = True,
) -> torch.Tensor:
    """
    h: (B,H)
    direction: (H,)
    """
    d = direction
    if normalize:
        d = normalize_direction(d)
    return h + alpha * d.view(1, -1).to(h.device)


@torch.no_grad()
def rollout_with_intervention(
    model,
    batch: Dict[str, torch.Tensor],
    horizon: int,
    direction: torch.Tensor,
    spec: InterventionSpec,
) -> Dict[str, torch.Tensor]:
    """
    rollout(model, batch, horizon) と同等のインタフェースで、
    hidden に intervention を入れる版。

    前提: model は WorldModelVP / VPF で
      - image_encoder / proprio_encoder / (force_encoder)
      - dynamics.forward_step(x_t, h_prev)
      - decoder(h_seq) -> (q_hat, dq_hat, block_hat)
    が存在すること。
    """
    device = next(model.parameters()).device

    # batchは (B,T,...) 前提に揃える
    def _bt(x):
        return x if x.dim() >= 3 else x.unsqueeze(0)

    rgb = _bt(batch["rgb"]).to(device)
    q = _bt(batch["q"]).to(device)
    dq = _bt(batch["dq"]).to(device)
    block_pose = _bt(batch["block_pose"]).to(device)
    action = _bt(batch["action"]).to(device)

    use_force = hasattr(model, "force_encoder") and ("f" in batch)
    if use_force:
        f = _bt(batch["f"]).to(device)
    else:
        f = None

    # 入力の embedding（t=0 の観測から）
    img_emb = model.image_encoder(rgb)  # (B,T,Ei)
    prop_emb = model.proprio_encoder(q, dq, block_pose)  # (B,T,Ep)

    if use_force:
        force_emb = model.force_encoder(f)  # (B,T,Ef)
        emb = torch.cat([img_emb, prop_emb, force_emb], dim=-1)
    else:
        emb = torch.cat([img_emb, prop_emb], dim=-1)

    B, T, E = emb.shape
    H = model.dynamics.hidden_dim if hasattr(model.dynamics, "hidden_dim") else None

    # rollout 長が T-1 を超える場合、action の範囲に注意
    # ここでは「データにあるactionを使って horizon分回す」前提（horizon <= T-1 推奨）
    horizon = min(horizon, T - 1)

    h_list = []
    h_t: Optional[torch.Tensor] = None

    # step0: x_0 = [emb_0, a_0] で h_0 を作る（あなたのrollout実装に合わせてここは調整可）
    for t in range(horizon):
        a_t = action[:, t]  # (B,A)
        x_t = torch.cat([emb[:, t], a_t], dim=-1)
        h_t = model.dynamics.forward_step(x_t, h_t)  # (B,H)

        if spec.apply_each_step:
            h_t = apply_intervention(h_t, direction, spec.alpha, normalize=spec.normalize)

        h_list.append(h_t)

    h_seq = torch.stack(h_list, dim=1)  # (B,horizon,H)
    q_hat, dq_hat, block_hat = model.decoder(h_seq)

    return {
        "h_seq": h_seq,
        "q_hat": q_hat,
        "dq_hat": dq_hat,
        "block_pose_hat": block_hat,
    }
