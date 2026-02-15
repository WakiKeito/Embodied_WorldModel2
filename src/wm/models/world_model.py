# src/wm/models/world_model.py
"""世界モデルの最小構成（VP / VPF(in) / VPF-aux(strict)）を提供する。

VP:
  - dynamics input = img_emb + prop_emb + action

VPF(in):
  - dynamics input = img_emb + prop_emb + force_emb + action
  - optional aux: force_hat head + losses

VPF-aux(strict):
  - dynamics input = img_emb + prop_emb + action  （VPと同じ：forceは入れない）
  - aux: force_hat head + losses（教師として f を使うだけ）
"""

from __future__ import annotations
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .decoders import StateDecoder
from .dynamics import GRUDynamics
from .encoders import ImageEncoderCNN, ProprioEncoderMLP, ForceEncoderMLP, concat_embeddings


def _ensure_batch_time(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """(T, ...) を (1, T, ...) に揃える。"""
    if x.dim() in {2, 4}:
        return x.unsqueeze(0), False
    return x, True


class WorldModelVP(nn.Module):
    """VP用の最小世界モデル。"""

    force_in_dynamics: bool = False  # ★判定用フラグ

    def __init__(
        self,
        j_dim: int,
        action_dim: int,
        img_embed_dim: int = 64,
        prop_embed_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoderCNN(embed_dim=img_embed_dim)
        self.proprio_encoder = ProprioEncoderMLP(
            input_dim=j_dim * 2 + 7,
            embed_dim=prop_embed_dim,
        )
        self.dynamics = GRUDynamics(
            input_dim=img_embed_dim + prop_embed_dim + action_dim,
            hidden_dim=hidden_dim,
        )
        self.decoder = StateDecoder(hidden_dim=hidden_dim, j_dim=j_dim, block_dim=7)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rgb = batch["rgb"]
        q = batch["q"]
        dq = batch["dq"]
        block_pose = batch["block_pose"]
        action = batch["action"]

        rgb, has_batch = _ensure_batch_time(rgb)
        q, _ = _ensure_batch_time(q)
        dq, _ = _ensure_batch_time(dq)
        block_pose, _ = _ensure_batch_time(block_pose)
        action, _ = _ensure_batch_time(action)

        img_emb = self.image_encoder(rgb)
        prop_emb = self.proprio_encoder(q, dq, block_pose)
        emb = concat_embeddings(img_emb, prop_emb)

        b, t, _ = emb.shape
        device = emb.device

        h_list = []
        h_t = None
        zero_action = torch.zeros(b, self.action_dim, device=device)

        for step in range(t - 1):
            action_prev = zero_action if step == 0 else action[:, step - 1]
            x_t = torch.cat([emb[:, step], action_prev], dim=-1)
            h_t = self.dynamics.forward_step(x_t, h_t)
            h_list.append(h_t)

        h_seq = torch.stack(h_list, dim=1)  # (B, T-1, H)
        q_hat, dq_hat, block_hat = self.decoder(h_seq)

        outputs = {
            "q_hat": q_hat,
            "dq_hat": dq_hat,
            "block_pose_hat": block_hat,
            "h_seq": h_seq,
        }
        if not has_batch:
            for k in outputs:
                outputs[k] = outputs[k].squeeze(0)
        return outputs

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        has_batch = batch["q"].dim() == 3
        batch_in: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() in {2, 4}:
                batch_in[key] = value.unsqueeze(0)
            else:
                batch_in[key] = value

        outputs = self.forward(batch_in)

        q = batch_in["q"]
        dq = batch_in["dq"]
        block_pose = batch_in["block_pose"]

        q_target = q[:, 1:]
        dq_target = dq[:, 1:]
        block_target = block_pose[:, 1:]

        loss_q = F.mse_loss(outputs["q_hat"], q_target)
        loss_dq = F.mse_loss(outputs["dq_hat"], dq_target)
        loss_block = F.mse_loss(outputs["block_pose_hat"], block_target)

        loss = loss_q + loss_dq + loss_block

        metrics = {
            "loss_total": loss,
            "loss_q": loss_q,
            "loss_dq": loss_dq,
            "loss_block_pose": loss_block,
        }
        if not has_batch:
            for k in metrics:
                metrics[k] = metrics[k].squeeze()
        return loss, metrics


class WorldModelVPF(WorldModelVP):
    """
    VPF(in): Force is part of the dynamics input (the "real" VPF).
    Optional aux: force_hat head + losses.

    dynamics input = img + prop + force_emb + action
    """

    force_in_dynamics: bool = True  # ★判定用フラグ

    def __init__(
        self,
        j_dim: int,
        action_dim: int,
        force_dim: int,
        img_embed_dim: int = 64,
        prop_embed_dim: int = 64,
        force_embed_dim: int = 32,
        hidden_dim: int = 128,
        # --- stabilization knobs (TRAIN time only) ---
        force_lambda_train: float = 1.0,
        force_dropout_p: float = 0.0,
        # --- aux loss knobs (default OFF) ---
        lambda_force: float = 0.0,
        lambda_zero: float = 0.0,
        w_contact: float = 1.0,
        w_coast: float = 1.0,
        contact_key: str = "is_contact",
        force_th: float = 1.0,
    ) -> None:
        super().__init__(
            j_dim=j_dim,
            action_dim=action_dim,
            img_embed_dim=img_embed_dim,
            prop_embed_dim=prop_embed_dim,
            hidden_dim=hidden_dim,
        )

        self.force_dim = int(force_dim)
        self.force_encoder = ForceEncoderMLP(input_dim=self.force_dim, embed_dim=force_embed_dim)

        self.dynamics = GRUDynamics(
            input_dim=img_embed_dim + prop_embed_dim + force_embed_dim + action_dim,
            hidden_dim=hidden_dim,
        )

        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.force_dim),
        )

        self.force_lambda_train = float(force_lambda_train)
        self.force_dropout_p = float(force_dropout_p)

        self.lambda_force = float(lambda_force)
        self.lambda_zero = float(lambda_zero)
        self.w_contact = float(w_contact)
        self.w_coast = float(w_coast)
        self.contact_key = str(contact_key)
        self.force_th = float(force_th)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rgb = batch["rgb"]
        q = batch["q"]
        dq = batch["dq"]
        block_pose = batch["block_pose"]
        f = batch["f"]
        action = batch["action"]

        rgb, has_batch = _ensure_batch_time(rgb)
        q, _ = _ensure_batch_time(q)
        dq, _ = _ensure_batch_time(dq)
        block_pose, _ = _ensure_batch_time(block_pose)
        f, _ = _ensure_batch_time(f)
        action, _ = _ensure_batch_time(action)

        img_emb = self.image_encoder(rgb)
        prop_emb = self.proprio_encoder(q, dq, block_pose)
        force_emb = self.force_encoder(f)

        if self.training:
            lam = self.force_lambda_train
            if lam != 1.0:
                force_emb = force_emb * lam
            p = self.force_dropout_p
            if p > 0.0:
                force_emb = F.dropout(force_emb, p=p, training=True)

        emb = concat_embeddings(img_emb, prop_emb, force_emb)

        b, t, _ = emb.shape
        device = emb.device

        h_list = []
        h_t = None
        zero_action = torch.zeros(b, self.action_dim, device=device)

        for step in range(t - 1):
            action_prev = zero_action if step == 0 else action[:, step - 1]
            x_t = torch.cat([emb[:, step], action_prev], dim=-1)
            h_t = self.dynamics.forward_step(x_t, h_t)
            h_list.append(h_t)

        h_seq = torch.stack(h_list, dim=1)
        q_hat, dq_hat, block_hat = self.decoder(h_seq)
        force_hat = self.force_head(h_seq)

        outputs = {
            "q_hat": q_hat,
            "dq_hat": dq_hat,
            "block_pose_hat": block_hat,
            "force_hat": force_hat,
            "h_seq": h_seq,
        }
        if not has_batch:
            for k in outputs:
                outputs[k] = outputs[k].squeeze(0)
        return outputs

    def _get_contact_mask(self, batch_in: Dict[str, torch.Tensor], Tm1: int) -> torch.Tensor:
        device = batch_in["q"].device

        def fallback_from_force() -> torch.Tensor:
            f = batch_in["f"]
            if f.dim() == 2:
                f = f.unsqueeze(0)
            fn = torch.linalg.norm(f[:, 1:1 + Tm1], dim=-1)
            return (fn > float(self.force_th)).float()

        if self.contact_key not in batch_in:
            return fallback_from_force()

        c = batch_in[self.contact_key]
        if c.dim() == 1:
            c = c.unsqueeze(0)
        elif c.dim() == 0:
            return fallback_from_force()

        while c.dim() >= 3 and c.shape[-1] == 1:
            c = c[..., 0]
        while c.dim() >= 3 and c.shape[1] == 1:
            c = c[:, 0, :]

        if c.dim() == 2:
            B = batch_in["q"].shape[0]
            if c.shape[0] != B and c.shape[1] == B:
                c = c.transpose(0, 1)
            if c.shape[1] < (Tm1 + 1):
                return fallback_from_force()
            c = c[:, 1:1 + Tm1].float()
            return (c > 0.5).float()

        if c.dim() == 3:
            B = batch_in["q"].shape[0]
            if c.shape[0] != B:
                return fallback_from_force()
            sizes = list(c.shape)
            candidates = []
            for dim in (1, 2):
                if sizes[dim] >= (Tm1 + 1):
                    candidates.append(dim)
            if not candidates:
                return fallback_from_force()
            time_dim = candidates[0]
            if time_dim == 2:
                c = c.transpose(1, 2)
            if c.shape[2] > 1:
                c = c[:, :, 0]
            if c.shape[1] < (Tm1 + 1):
                return fallback_from_force()
            c = c[:, 1:1 + Tm1].float()
            return (c > 0.5).float()

        return fallback_from_force()

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        has_batch = batch["q"].dim() == 3
        batch_in: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() in {2, 4}:
                batch_in[key] = value.unsqueeze(0)
            else:
                batch_in[key] = value

        outputs = self.forward(batch_in)

        q = batch_in["q"]
        dq = batch_in["dq"]
        block_pose = batch_in["block_pose"]

        q_target = q[:, 1:]
        dq_target = dq[:, 1:]
        block_target = block_pose[:, 1:]

        loss_q = F.mse_loss(outputs["q_hat"], q_target)
        loss_dq = F.mse_loss(outputs["dq_hat"], dq_target)
        loss_block = F.mse_loss(outputs["block_pose_hat"], block_target)

        base_loss = loss_q + loss_dq + loss_block

        if self.lambda_force <= 0.0 and self.lambda_zero <= 0.0:
            metrics = {
                "loss_total": base_loss,
                "loss_q": loss_q,
                "loss_dq": loss_dq,
                "loss_block_pose": loss_block,
            }
            if not has_batch:
                for k in metrics:
                    metrics[k] = metrics[k].squeeze()
            return base_loss, metrics

        f = batch_in["f"]
        force_hat = outputs["force_hat"]
        B, Tm1, Fdim = force_hat.shape

        if f.dim() == 2:
            f = f.unsqueeze(0)
        f_target = f[:, 1:1 + Tm1, :]

        contact = self._get_contact_mask(batch_in, Tm1)
        coast = 1.0 - contact

        mse_f = ((force_hat - f_target) ** 2).mean(dim=-1)
        w = self.w_contact * contact + self.w_coast * coast
        denom = w.sum().clamp(min=1.0)
        loss_force = (mse_f * w).sum() / denom

        mse_zero = (force_hat ** 2).mean(dim=-1)
        denom0 = coast.sum().clamp(min=1.0)
        loss_zero = (mse_zero * coast).sum() / denom0

        loss = base_loss + self.lambda_force * loss_force + self.lambda_zero * loss_zero

        metrics = {
            "loss_total": loss,
            "loss_q": loss_q,
            "loss_dq": loss_dq,
            "loss_block_pose": loss_block,
            "loss_force": loss_force,
            "loss_force_zero_coast": loss_zero,
            "contact_ratio": contact.mean(),
        }
        if not has_batch:
            for k in metrics:
                metrics[k] = metrics[k].squeeze()
        return loss, metrics


class WorldModelVPFAux(WorldModelVP):
    """
    VPF-aux(strict):
      - dynamics input = img + prop + action (VPと同じ)
      - force は入力しない
      - hidden から force_hat を予測し、aux loss を掛ける（任意）
    """

    force_in_dynamics: bool = False  # ★厳密に False

    def __init__(
        self,
        j_dim: int,
        action_dim: int,
        force_dim: int,
        img_embed_dim: int = 64,
        prop_embed_dim: int = 64,
        hidden_dim: int = 128,
        # --- aux knobs (default OFF) ---
        lambda_force: float = 0.0,
        lambda_zero: float = 0.0,
        w_contact: float = 1.0,
        w_coast: float = 1.0,
        contact_key: str = "is_contact",
        force_th: float = 1.0,
    ) -> None:
        super().__init__(
            j_dim=j_dim,
            action_dim=action_dim,
            img_embed_dim=img_embed_dim,
            prop_embed_dim=prop_embed_dim,
            hidden_dim=hidden_dim,
        )

        self.force_dim = int(force_dim)

        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.force_dim),
        )

        self.lambda_force = float(lambda_force)
        self.lambda_zero = float(lambda_zero)
        self.w_contact = float(w_contact)
        self.w_coast = float(w_coast)
        self.contact_key = str(contact_key)
        self.force_th = float(force_th)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = super().forward(batch)  # q_hat,dq_hat,block_pose_hat,h_seq
        h_seq = out["h_seq"]
        force_hat = self.force_head(h_seq)
        out["force_hat"] = force_hat
        return out

    def _get_contact_mask(self, batch_in: Dict[str, torch.Tensor], Tm1: int) -> torch.Tensor:
        device = batch_in["q"].device

        def fallback_from_force() -> torch.Tensor:
            f = batch_in["f"]
            if f.dim() == 2:
                f = f.unsqueeze(0)
            fn = torch.linalg.norm(f[:, 1:1 + Tm1], dim=-1)
            return (fn > float(self.force_th)).float()

        if "f" not in batch_in:
            # aux trainingできないので0（=全部coast扱い）にする
            return torch.zeros(batch_in["q"].shape[0], Tm1, device=device)

        if self.contact_key not in batch_in:
            return fallback_from_force()

        c = batch_in[self.contact_key]
        if c.dim() == 1:
            c = c.unsqueeze(0)
        elif c.dim() == 0:
            return fallback_from_force()

        while c.dim() >= 3 and c.shape[-1] == 1:
            c = c[..., 0]
        while c.dim() >= 3 and c.shape[1] == 1:
            c = c[:, 0, :]

        if c.dim() == 2:
            B = batch_in["q"].shape[0]
            if c.shape[0] != B and c.shape[1] == B:
                c = c.transpose(0, 1)
            if c.shape[1] < (Tm1 + 1):
                return fallback_from_force()
            c = c[:, 1:1 + Tm1].float()
            return (c > 0.5).float()

        return fallback_from_force()

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        # baseはVPと同じ
        loss_base, metrics = super().compute_loss(batch)

        # aux OFF
        if self.lambda_force <= 0.0 and self.lambda_zero <= 0.0:
            return loss_base, metrics

        # aux ON には f が必要
        if "f" not in batch:
            # 落とさずbaseのみ
            return loss_base, metrics

        has_batch = batch["q"].dim() == 3
        batch_in: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() in {2, 4}:
                batch_in[key] = value.unsqueeze(0)
            else:
                batch_in[key] = value

        out = self.forward(batch_in)
        force_hat = out["force_hat"]  # (B,Tm1,F)
        B, Tm1, Fdim = force_hat.shape

        f = batch_in["f"]
        if f.dim() == 2:
            f = f.unsqueeze(0)
        f_target = f[:, 1:1 + Tm1, :]

        contact = self._get_contact_mask(batch_in, Tm1)
        coast = 1.0 - contact

        mse_f = ((force_hat - f_target) ** 2).mean(dim=-1)
        w = self.w_contact * contact + self.w_coast * coast
        denom = w.sum().clamp(min=1.0)
        loss_force = (mse_f * w).sum() / denom

        mse_zero = (force_hat ** 2).mean(dim=-1)
        denom0 = coast.sum().clamp(min=1.0)
        loss_zero = (mse_zero * coast).sum() / denom0

        loss = metrics["loss_total"] + self.lambda_force * loss_force + self.lambda_zero * loss_zero

        metrics = dict(metrics)
        metrics["loss_total"] = loss
        metrics["loss_force"] = loss_force
        metrics["loss_force_zero_coast"] = loss_zero
        metrics["contact_ratio"] = contact.mean()

        if not has_batch:
            for k in list(metrics.keys()):
                if torch.is_tensor(metrics[k]):
                    metrics[k] = metrics[k].squeeze()
        return loss, metrics
