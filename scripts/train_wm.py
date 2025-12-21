"""
World Model 学習スクリプト（最小版）

- EpisodeNPZDataset -> DataLoader -> WorldModel(VP/VPF) -> compute_loss -> optimizer step
- ckpt保存（best / last）
- ログはprintのみ（最小）

使い方例:
  # VP
  PYTHONPATH=src python -m scripts.train_wm --dataset-root datasets/raw_synth --out-dir outputs/ckpt_vp

  # VPF
  PYTHONPATH=src python -m scripts.train_wm --dataset-root datasets/raw_synth --use-force --out-dir outputs/ckpt_vpf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from wm.data.dataset_npz import EpisodeNPZDataset
from wm.data.collate import collate_fixed_length
from wm.models.world_model import WorldModelVP, WorldModelVPF


def split_paths(paths: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = max(1, int(len(paths) * val_ratio))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    tr_paths = [paths[i] for i in tr_idx]
    va_paths = [paths[i] for i in val_idx]
    return tr_paths, va_paths


def to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def save_ckpt(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_val: float,
    args_dict: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_val": best_val,
            "args": args_dict,
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("datasets/raw"))
    p.add_argument("--use-force", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/ckpt"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.2)

    # data slicing
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--frame-skip", type=int, default=1)

    # train
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    npz_files = sorted(args.dataset_root.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"NPZが見つかりません: {args.dataset_root}")

    tr_paths, va_paths = split_paths(npz_files, args.val_ratio, args.seed)
    if len(tr_paths) == 0:
        # データが少なすぎる場合の保険
        tr_paths = npz_files
        va_paths = npz_files
        print("[WARN] データが少ないため train=val で学習します（本番では避けてください）。")

    cfg = {
        "sequence_length": int(args.sequence_length),
        "frame_skip": int(args.frame_skip),
        "keys": {},  # 形状検証をyamlに寄せたい場合は後で入れる
    }

    tr_ds = EpisodeNPZDataset(tr_paths, config=cfg)
    va_ds = EpisodeNPZDataset(va_paths, config=cfg)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fixed_length)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] train={len(tr_ds)} val={len(va_ds)}")
    print(f"[INFO] model={'VPF' if args.use_force else 'VP'}")

    # 次元推定用に1バッチ
    batch0 = next(iter(tr_loader))
    batch0 = to_device(batch0, device)
    j_dim = batch0["q"].shape[-1]
    action_dim = batch0["action"].shape[-1]

    if args.use_force:
        model = WorldModelVPF(j_dim=j_dim, action_dim=action_dim).to(device)
    else:
        model = WorldModelVP(j_dim=j_dim, action_dim=action_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    global_step = 0

    args_dict = vars(args).copy()
    args_dict["j_dim"] = int(j_dim)
    args_dict["action_dim"] = int(action_dim)

    for epoch in range(1, args.epochs + 1):
        # -------- train --------
        model.train()
        running = 0.0
        n = 0

        for batch in tr_loader:
            global_step += 1
            batch = to_device(batch, device)

            loss, metrics = model.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running += float(metrics["loss_total"].item())
            n += 1

            if global_step % args.log_every == 0:
                print(f"[train] epoch={epoch:03d} step={global_step:06d} loss={running/max(n,1):.6f}")

        train_loss = running / max(n, 1)

        # -------- val --------
        model.eval()
        v_running = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in va_loader:
                batch = to_device(batch, device)
                loss, metrics = model.compute_loss(batch)
                v_running += float(metrics["loss_total"].item())
                v_n += 1

        val_loss = v_running / max(v_n, 1)

        print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # save last
        save_ckpt(
            args.out_dir / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            step=global_step,
            best_val=best_val,
            args_dict=args_dict,
        )

        # save best
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(
                args.out_dir / "best.pt",
                model,
                optimizer,
                epoch=epoch,
                step=global_step,
                best_val=best_val,
                args_dict=args_dict,
            )
            print(f"[OK] best updated: val_loss={best_val:.6f}")

    print(f"[DONE] best_val={best_val:.6f}")
    print(f"[OK] saved ckpt dir: {args.out_dir}")


if __name__ == "__main__":
    main()
