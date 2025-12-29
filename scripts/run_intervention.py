"""
Run Latent Intervention experiment.

やること：
1) 1 episode をロード
2) t=0 の埋め込み emb0 を作って h0 を作る（GRU step）
3) h0 に probe 方向を alpha 倍して介入
4) 介入後の h から horizon 分 rollout
5) 介入なし(α=0) との RMSE を比較して CSV 出力

注意：
- ここでは「rollout が改善/悪化するか」を見るための最小実装
- “massが上がるとこうなるべき” の主張は本データで検証すること
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv

import torch

from wm.data.dataset_npz import EpisodeNPZDataset
from wm.data.collate import collate_fixed_length
from wm.models.world_model import WorldModelVP, WorldModelVPF
from wm.utils.rollout import rollout
from wm.utils.intervention import load_probe_directions, normalize_direction, apply_intervention


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).sqrt().item()


@torch.no_grad()
def build_initial_hidden(model, batch: dict, use_force: bool) -> torch.Tensor:
    """
    t=0 の観測から埋め込みを作り、GRU 1 step で h0 を作る。
    WorldModelVP/VPF の実装に合わせる。

    Returns:
        h0: (B,H)
    """
    # batch は (B,T,...) 前提
    rgb = batch["rgb"]
    q = batch["q"]
    dq = batch["dq"]
    block_pose = batch["block_pose"]
    action = batch["action"]

    # encoders は時系列入力を受けて (B,T,E) を返す実装
    img_emb = model.image_encoder(rgb)  # (B,T,Ei)
    prop_emb = model.proprio_encoder(q, dq, block_pose)  # (B,T,Ep)

    if use_force:
        force = batch["f"]
        force_emb = model.force_encoder(force)  # (B,T,Ef)
        emb0 = torch.cat([img_emb[:, 0], prop_emb[:, 0], force_emb[:, 0]], dim=-1)
    else:
        emb0 = torch.cat([img_emb[:, 0], prop_emb[:, 0]], dim=-1)

    # t=0 は a_{-1}=0 を使う（world_model.py と同じ）
    B = emb0.shape[0]
    a0 = torch.zeros(B, model.action_dim, device=emb0.device)

    x0 = torch.cat([emb0, a0], dim=-1)  # (B, E + A)
    h0 = model.dynamics.forward_step(x0, None)  # (B,H)
    return h0


@torch.no_grad()
def rollout_with_intervened_h0(model, batch: dict, h0: torch.Tensor, horizon: int, use_force: bool):
    """
    rollout.py を “h0 指定” で回す最小版。
    rollout.py を壊さないために、ここでだけ実装する。
    """
    # batch は (B,T,...) 前提
    rgb = batch["rgb"]
    q = batch["q"]
    dq = batch["dq"]
    block_pose = batch["block_pose"]
    action = batch["action"]

    img_emb = model.image_encoder(rgb)              # (B,T,Ei)
    prop_emb = model.proprio_encoder(q, dq, block_pose)  # (B,T,Ep)

    if use_force:
        force = batch["f"]
        force_emb = model.force_encoder(force)      # (B,T,Ef)
        emb = torch.cat([img_emb, prop_emb, force_emb], dim=-1)  # (B,T,E)
    else:
        emb = torch.cat([img_emb, prop_emb], dim=-1)  # (B,T,E)

    # 予測を格納
    q_hats = []
    dq_hats = []
    block_hats = []

    h_t = h0

    # ここは簡易に「teacher forcing 的に action は batch の先頭から使う」
    # 予測の比較が目的なので、行動は固定でOK
    for step in range(horizon):
        # action_prev は step==0 のとき a_{-1}=0、それ以外は action[:, step-1]
        if step == 0:
            a_prev = torch.zeros(batch["action"].shape[0], model.action_dim, device=emb.device)
        else:
            a_prev = action[:, step - 1]

        x_t = torch.cat([emb[:, step], a_prev], dim=-1)  # (B,E+A)
        h_t = model.dynamics.forward_step(x_t, h_t)      # (B,H)

        # decoder: (B,?,J) を返す実装だが、h_t は (B,H) なので unsqueeze
        q_hat, dq_hat, block_hat = model.decoder(h_t.unsqueeze(1))
        q_hats.append(q_hat[:, 0])
        dq_hats.append(dq_hat[:, 0])
        block_hats.append(block_hat[:, 0])

    preds = {
        "q_hat": torch.stack(q_hats, dim=1),                # (B,H,J)
        "dq_hat": torch.stack(dq_hats, dim=1),              # (B,H,J)
        "block_pose_hat": torch.stack(block_hats, dim=1),   # (B,H,7)
    }
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=30)

    parser.add_argument("--use-force", action="store_true", help="VPF を使う")
    parser.add_argument("--ckpt", type=Path, required=True, help="world model ckpt (best.pt)")
    parser.add_argument("--probe", type=Path, required=True, help="probe ckpt (probe_vpf.pt など)")

    parser.add_argument("--axis", type=str, default="mass", choices=["mass", "friction"])
    parser.add_argument("--alpha", type=float, default=0.0, help="介入強度（0=介入なし）")
    parser.add_argument("--normalize", action="store_true", help="direction を L2 正規化する")

    parser.add_argument("--out", type=Path, default=Path("outputs/intervention.csv"))
    args = parser.parse_args()

    npz_files = sorted(args.dataset_root.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No npz in {args.dataset_root}")

    p = npz_files[args.episode_idx]

    # dataset -> batch (B=1)
    cfg = {"sequence_length": 64, "frame_skip": 1, "keys": {}}
    ds = EpisodeNPZDataset([p], config=cfg)
    batch = collate_fixed_length([ds[0]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    j_dim = batch["q"].shape[-1]
    action_dim = batch["action"].shape[-1]

    if args.use_force:
        model = WorldModelVPF(j_dim=j_dim, action_dim=action_dim).to(device)
        print("[INFO] Using VPF")
    else:
        model = WorldModelVP(j_dim=j_dim, action_dim=action_dim).to(device)
        print("[INFO] Using VP")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    dirs = load_probe_directions(args.probe, device=device)
    if dirs.in_dim != 128:
        print(f"[WARN] probe in_dim={dirs.in_dim}")

    direction = dirs.w_mass if args.axis == "mass" else dirs.w_friction
    if args.normalize:
        direction = normalize_direction(direction)

    # baseline rollout（既存 rollout.py）
    preds_base = rollout(model, batch, horizon=args.horizon)

    # intervention rollout
    h0 = build_initial_hidden(model, batch, use_force=args.use_force)
    h0_i = apply_intervention(h0, direction, alpha=args.alpha)
    preds_int = rollout_with_intervened_h0(model, batch, h0_i, horizon=args.horizon, use_force=args.use_force)

    # GT
    q_gt = batch["q"][:, 1:args.horizon + 1]
    block_gt = batch["block_pose"][:, 1:args.horizon + 1]

    rmse_q_base = rmse(preds_base["q_hat"], q_gt)
    rmse_block_base = rmse(preds_base["block_pose_hat"], block_gt)

    rmse_q_int = rmse(preds_int["q_hat"], q_gt)
    rmse_block_int = rmse(preds_int["block_pose_hat"], block_gt)

    # output CSV (1行)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "model", "axis", "alpha", "normalize",
            "mass", "friction",
            "rmse_q_base", "rmse_block_base",
            "rmse_q_int", "rmse_block_int",
        ])
        w.writerow([
            p.name,
            "VPF" if args.use_force else "VP",
            args.axis,
            args.alpha,
            int(args.normalize),
            float(batch["mass"].item()),
            float(batch["friction"].item()),
            rmse_q_base, rmse_block_base,
            rmse_q_int, rmse_block_int,
        ])

    print(f"[OK] saved: {args.out}")
    print(f"[BASE] rmse_q={rmse_q_base:.6f} rmse_block={rmse_block_base:.6f}")
    print(f"[INT ] rmse_q={rmse_q_int:.6f} rmse_block={rmse_block_int:.6f}")


if __name__ == "__main__":
    main()
