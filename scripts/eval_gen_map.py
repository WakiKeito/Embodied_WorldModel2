import argparse
from pathlib import Path
import csv
from collections import defaultdict

import torch
import numpy as np

from wm.data.dataset_npz import EpisodeNPZDataset
from wm.data.collate import collate_fixed_length
from wm.models.world_model import WorldModelVP, WorldModelVPF
from wm.utils.rollout import rollout


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).sqrt().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--use-force", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/genmap_rollout.csv"))
    args = parser.parse_args()

    npz_files = sorted(args.dataset_root.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No npz in {args.dataset_root}")

    # まず1本ロードして次元決定
    tmp_ds = EpisodeNPZDataset([npz_files[0]], config={"sequence_length": 64, "frame_skip": 1, "keys": {}})
    tmp_batch = collate_fixed_length([tmp_ds[0]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_batch = {k: v.to(device) for k, v in tmp_batch.items()}

    j_dim = tmp_batch["q"].shape[-1]
    action_dim = tmp_batch["action"].shape[-1]

    if args.use_force:
        model = WorldModelVPF(j_dim=j_dim, action_dim=action_dim).to(device)
        print("[INFO] Using VPF")
    else:
        model = WorldModelVP(j_dim=j_dim, action_dim=action_dim).to(device)
        print("[INFO] Using VP")

    model.eval()

    # (mass, friction) -> list of metrics
    bucket = defaultdict(list)

    # 全episodeを回す
    for p in npz_files:
        ds = EpisodeNPZDataset([p], config={"sequence_length": 64, "frame_skip": 1, "keys": {}})
        batch = collate_fixed_length([ds[0]])
        batch = {k: v.to(device) for k, v in batch.items()}

        mass = float(batch["mass"].item())
        friction = float(batch["friction"].item())

        preds = rollout(model, batch, horizon=args.horizon)

        # GT: t=1..H
        q_gt = batch["q"][:, 1:args.horizon + 1]
        block_gt = batch["block_pose"][:, 1:args.horizon + 1]

        rmse_q = rmse(preds["q_hat"], q_gt)
        rmse_block = rmse(preds["block_pose_hat"], block_gt)

        bucket[(mass, friction)].append((rmse_q, rmse_block))

    # 出力
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mass", "friction", "rmse_q", "rmse_block_pose", "n_episodes"])
        for (m, mu), vals in sorted(bucket.items()):
            vals = np.array(vals, dtype=np.float32)
            writer.writerow([m, mu, float(vals[:, 0].mean()), float(vals[:, 1].mean()), int(len(vals))])

    print(f"[OK] saved: {args.out}")


if __name__ == "__main__":
    main()
