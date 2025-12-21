from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from wm.data.collate import collate_fixed_length
from wm.data.dataset_npz import EpisodeNPZDataset
from wm.models.world_model import WorldModelVP, WorldModelVPF
from wm.utils.rollout import rollout


def to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def load_ckpt(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("datasets/raw"))
    p.add_argument("--use-force", action="store_true")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()

    npz_files = sorted(args.dataset_root.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"NPZが見つかりません: {args.dataset_root}")

    cfg = {"sequence_length": 64, "frame_skip": 1, "keys": {}}
    ds = EpisodeNPZDataset(npz_files, config=cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch0 = next(iter(loader))
    batch0 = to_device(batch0, device)
    j_dim = batch0["q"].shape[-1]
    action_dim = batch0["action"].shape[-1]

    if args.use_force:
        model = WorldModelVPF(j_dim=j_dim, action_dim=action_dim).to(device)
        print("[INFO] Using VPF")
    else:
        model = WorldModelVP(j_dim=j_dim, action_dim=action_dim).to(device)
        print("[INFO] Using VP")

    load_ckpt(model, args.ckpt, device)
    model.eval()

    # 1バッチで評価（必要なら全体平均にしてもOK）
    with torch.no_grad():
        preds = rollout(model, batch0, horizon=args.horizon)

    # rollout.py が返すフォーマットに合わせる：最低限 q のRMSEを出す
    # preds["q_hat_roll"] がある想定（なければあなたのrolloutのkeyに合わせて調整）
    q_true = batch0["q"][:, 1 : 1 + args.horizon]  # (B,H,J)
    q_hat = preds["q_hat"]  # (B,H,J) を想定

    rmse_q = torch.mean((q_hat - q_true) ** 2).sqrt().item()
    print(f"Rollout RMSE (q): {rmse_q:.6f}")


if __name__ == "__main__":
    main()
