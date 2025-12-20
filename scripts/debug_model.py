"""
世界モデル最小構成（VP / VPF）の動作確認スクリプト。

- Dataset → DataLoader → Model → forward → loss
が一通り動くかを確認するための最小デバッグ用。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from wm.data.collate import collate_fixed_length
from wm.data.dataset_npz import EpisodeNPZDataset
from wm.models.world_model import WorldModelVP, WorldModelVPF


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="世界モデル（VP / VPF）の簡易デバッグ")
    parser.add_argument("--default-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/push_block_npz.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--use-force",
        action="store_true",
        help="Force encoder を使う（VPF）",
    )
    args = parser.parse_args()

    # -------------------------
    # config
    # -------------------------
    default_cfg = load_yaml(args.default_config)
    data_cfg = load_yaml(args.data_config)

    # -------------------------
    # dataset
    # -------------------------
    dataset_root = args.dataset_root or Path(default_cfg.get("dataset_root", "datasets/raw"))
    if dataset_root.is_dir() and dataset_root.name != "raw":
        candidate = dataset_root / "raw"
        if candidate.exists():
            dataset_root = candidate

    npz_files = sorted(dataset_root.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"NPZファイルが見つかりません: {dataset_root}")

    print(f"[INFO] dataset_root = {dataset_root}")
    print(f"[INFO] num_episodes = {len(npz_files)}")

    dataset = EpisodeNPZDataset(npz_files, config=data_cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fixed_length,
    )
    batch = next(iter(loader))

    # -------------------------
    # device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = move_batch_to_device(batch, device)

    # -------------------------
    # model
    # -------------------------
    j_dim = batch["q"].shape[-1]
    action_dim = batch["action"].shape[-1]

    if args.use_force:
        print("[INFO] Using VPF model")
        model = WorldModelVPF(j_dim=j_dim, action_dim=action_dim)
    else:
        print("[INFO] Using VP model")
        model = WorldModelVP(j_dim=j_dim, action_dim=action_dim)

    model = model.to(device)
    model.eval()

    # -------------------------
    # forward & loss
    # -------------------------
    with torch.no_grad():
        loss, metrics = model.compute_loss(batch)
        outputs = model(batch)

    # -------------------------
    # print results
    # -------------------------
    print("\n出力テンソルの shape")
    for key in ["q_hat", "dq_hat", "block_pose_hat", "h_seq"]:
        print(f"- {key}: {tuple(outputs[key].shape)}")

    print("\n損失値")
    print(f"- loss_total: {metrics['loss_total'].item():.6f}")


if __name__ == "__main__":
    main()
