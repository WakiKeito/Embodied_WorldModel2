"""世界モデルの最小構成（VP）を提供する。"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .decoders import StateDecoder
from .dynamics import GRUDynamics
from .encoders import ImageEncoderCNN, ProprioEncoderMLP, ForceEncoderMLP, concat_embeddings

class WorldModelVP(nn.Module):
    """VP用の最小世界モデル。"""

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
        self.action_dim = action_dim

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

        # ★ t-1 まで回す（t -> t+1 用）
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
    """VP + Force の世界モデル。"""

    def __init__(
        self,
        j_dim: int,
        action_dim: int,
        img_embed_dim: int = 64,
        prop_embed_dim: int = 64,
        force_embed_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__(
            j_dim=j_dim,
            action_dim=action_dim,
            img_embed_dim=img_embed_dim,
            prop_embed_dim=prop_embed_dim,
            hidden_dim=hidden_dim,
        )
        self.force_encoder = ForceEncoderMLP(
            input_dim=j_dim,
            embed_dim=force_embed_dim,
        )

        # dynamics の input_dim を拡張
        self.dynamics = GRUDynamics(
            input_dim=img_embed_dim + prop_embed_dim + force_embed_dim + action_dim,
            hidden_dim=hidden_dim,
        )

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

def _ensure_batch_time(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """(T, ...) を (1, T, ...) に揃える。"""
    if x.dim() in {2, 4}:
        return x.unsqueeze(0), False
    return x, True
