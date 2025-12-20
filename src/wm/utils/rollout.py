import torch
from typing import Dict

@torch.no_grad()
def rollout(model, batch: Dict[str, torch.Tensor], horizon: int) -> Dict[str, torch.Tensor]:
    """
    Open-loop rollout (teacher forcing なし).
    VP/VPF 両対応。
    - VPF の場合、force は予測していないので 0 ベクトルで固定（最小プロトタイプ）。
    """

    device = batch["q"].device

    # 初期観測（t=0）
    rgb0 = batch["rgb"][:, :1]          # (B,1,C,H,W)
    q0 = batch["q"][:, :1]              # (B,1,J)
    dq0 = batch["dq"][:, :1]            # (B,1,J)
    block0 = batch["block_pose"][:, :1] # (B,1,7)
    action = batch["action"]            # (B,T,A)

    # 画像埋め込み（最小：初期画像を固定で使う）
    img_emb = model.image_encoder(rgb0)   # (B,1,Di)

    # 初期状態埋め込み
    prop_emb = model.proprio_encoder(q0, dq0, block0)  # (B,1,Dp)

    use_force = hasattr(model, "force_encoder")
    if use_force:
        # rollout中は force を予測していないので 0 で固定
        B, _, J = q0.shape
        f0 = torch.zeros(B, 1, J, device=device)
        force_emb = model.force_encoder(f0)            # (B,1,Df)
        emb = torch.cat([img_emb, prop_emb, force_emb], dim=-1)  # (B,1,Di+Dp+Df)
    else:
        emb = torch.cat([img_emb, prop_emb], dim=-1)   # (B,1,Di+Dp)

    h_t = None
    preds_q, preds_dq, preds_block = [], [], []

    for k in range(horizon):
        # action を取り出す（足りなければ 0）
        if k < action.shape[1]:
            a_t = action[:, k]
        else:
            a_t = torch.zeros_like(action[:, 0])

        # 1-step dynamics
        x_t = torch.cat([emb[:, 0], a_t], dim=-1)  # (B, embed + A)
        h_t = model.dynamics.forward_step(x_t, h_t)

        # decode: (B,1,*) -> 取り出して (B,*)
        q_hat, dq_hat, block_hat = model.decoder(h_t.unsqueeze(1))
        q1 = q_hat[:, 0]
        dq1 = dq_hat[:, 0]
        bl1 = block_hat[:, 0]

        preds_q.append(q1)
        preds_dq.append(dq1)
        preds_block.append(bl1)

        # 次ステップ用に「予測状態」を埋め込みに戻す
        prop_emb = model.proprio_encoder(
            q1.unsqueeze(1),
            dq1.unsqueeze(1),
            bl1.unsqueeze(1),
        )  # (B,1,Dp)

        if use_force:
            # force は固定 0 のまま
            B, J = q1.shape
            fz = torch.zeros(B, 1, J, device=device)
            force_emb = model.force_encoder(fz)  # (B,1,Df)
            emb = torch.cat([img_emb, prop_emb, force_emb], dim=-1)
        else:
            emb = torch.cat([img_emb, prop_emb], dim=-1)

    return {
        "q_hat": torch.stack(preds_q, dim=1),                 # (B,H,J)
        "dq_hat": torch.stack(preds_dq, dim=1),               # (B,H,J)
        "block_pose_hat": torch.stack(preds_block, dim=1),    # (B,H,7)
    }
