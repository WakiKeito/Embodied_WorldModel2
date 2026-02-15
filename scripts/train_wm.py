#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_wm.py (fixed + normalized)

What this version does
- Compute normalization stats (q/dq/f) from TRAIN split (or load from --norm-cfg / existing out-dir/norm_cfg.json)
- Save norm_cfg.json under out-dir (only when --save-norm-cfg)
- Apply the SAME normalization to train/val datasets (EpisodeNPZDataset config["normalization"])
- Save norm_cfg into checkpoint (best.pt) so run_intervention can always resolve it

Notes / assumptions
- Each episode npz has keys: q, dq, (optionally f)
- EpisodeNPZDataset reads config["normalization"] and applies it consistently.
- If your dataset may NOT have f (VP training), this code handles it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from wm.data.dataset_npz import EpisodeNPZDataset
from wm.data.collate import collate_fixed_length
from wm.models.world_model import WorldModelVP, WorldModelVPF


# -------------------------
# Normalization utilities
# -------------------------

@dataclass
class RunningMoments:
    n: int
    sum: np.ndarray
    sumsq: np.ndarray

    @classmethod
    def create(cls, dim: int) -> "RunningMoments":
        return cls(
            n=0,
            sum=np.zeros((dim,), dtype=np.float64),
            sumsq=np.zeros((dim,), dtype=np.float64),
        )

    def update(self, x: np.ndarray) -> None:
        """
        x: (..., D)
        """
        x2 = x.reshape(-1, x.shape[-1]).astype(np.float64)
        self.n += int(x2.shape[0])
        self.sum += x2.sum(axis=0)
        self.sumsq += (x2 * x2).sum(axis=0)

    def mean_std(self, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
        if self.n <= 0:
            raise ValueError("RunningMoments is empty")
        mean = self.sum / float(self.n)
        var = self.sumsq / float(self.n) - mean * mean
        var = np.maximum(var, 0.0)
        std = np.sqrt(var + eps)
        return mean.astype(np.float32), std.astype(np.float32)


def load_norm_cfg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_norm_cfg(norm_cfg: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(norm_cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def compute_norm_cfg_from_npz_dir(dataset_root: Path, require_force: bool) -> Dict:
    """
    Compute mean/std for q, dq, and (optionally) f over ALL time steps and episodes.

    require_force:
      - True  : fail if any episode misses 'f'
      - False : compute q/dq only (and compute f only if present in the first file AND in each file)
    """
    npz_paths = sorted(dataset_root.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No npz found under: {dataset_root}")

    # infer dims from first file
    with np.load(npz_paths[0], allow_pickle=True) as d0:
        if "q" not in d0.files or "dq" not in d0.files:
            raise KeyError(f"{npz_paths[0]} must contain 'q' and 'dq'")
        q_dim = int(d0["q"].shape[-1])
        dq_dim = int(d0["dq"].shape[-1])
        has_f0 = ("f" in d0.files)
        f_dim = int(d0["f"].shape[-1]) if has_f0 else 0

    mq = RunningMoments.create(q_dim)
    mdq = RunningMoments.create(dq_dim)
    mf = RunningMoments.create(f_dim) if (has_f0 and f_dim > 0) else None

    for p in npz_paths:
        with np.load(p, allow_pickle=True) as d:
            mq.update(d["q"])
            mdq.update(d["dq"])

            has_f = ("f" in d.files)
            if require_force and (not has_f):
                raise KeyError(f"require_force=True but missing 'f' in {p}")
            if mf is not None:
                if not has_f:
                    raise KeyError(f"Normalization expects 'f' (from first file) but missing in {p}")
                mf.update(d["f"])

    q_mean, q_std = mq.mean_std()
    dq_mean, dq_std = mdq.mean_std()

    norm_cfg: Dict = {
        "enabled": True,
        "q_mean": q_mean.tolist(),
        "q_std": q_std.tolist(),
        "dq_mean": dq_mean.tolist(),
        "dq_std": dq_std.tolist(),
    }

    if mf is not None:
        f_mean, f_std = mf.mean_std()
        norm_cfg["f_mean"] = f_mean.tolist()
        norm_cfg["f_std"] = f_std.tolist()

    return norm_cfg


def build_dataset_cfg(sequence_length: int, frame_skip: int, norm_cfg: Optional[Dict]) -> Dict:
    cfg = {
        "sequence_length": int(sequence_length),
        "frame_skip": int(frame_skip),
        "keys": {},  # keep your current behavior
    }
    if norm_cfg is not None:
        cfg["normalization"] = norm_cfg
    return cfg


# -------------------------
# Training helpers
# -------------------------

@torch.no_grad()
def evaluate(model, dl, device) -> float:
    model.eval()
    losses = []
    for batch in dl:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        loss, _ = model.compute_loss(batch)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def resolve_norm_cfg(
    train_root: Path,
    out_dir: Path,
    use_force: bool,
    norm_cfg_arg: Optional[Path],
    save_norm_cfg_flag: bool,
) -> Tuple[Dict, str]:
    """
    Priority:
    1) --norm-cfg path
    2) out_dir/norm_cfg.json (reuse for resume / consistency)
    3) compute from train_root

    If save_norm_cfg_flag=True, save the resolved norm_cfg to out_dir/norm_cfg.json
    """
    if norm_cfg_arg is not None:
        norm_cfg = load_norm_cfg(norm_cfg_arg)
        src = f"args:{norm_cfg_arg}"
    else:
        cand = out_dir / "norm_cfg.json"
        if cand.is_file():
            norm_cfg = load_norm_cfg(cand)
            src = f"file:{cand}"
        else:
            norm_cfg = compute_norm_cfg_from_npz_dir(train_root, require_force=bool(use_force))
            src = f"computed:{train_root}"

    if save_norm_cfg_flag:
        save_norm_cfg(norm_cfg, out_dir / "norm_cfg.json")

    return norm_cfg, src


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--val-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)

    ap.add_argument("--use-force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--sequence-length", type=int, default=64)
    ap.add_argument("--frame-skip", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)

    # --- VPF knobs (TRAIN time only) ---
    ap.add_argument("--force-lambda-train", type=float, default=1.0)
    ap.add_argument("--force-dropout", type=float, default=0.0)

    # --- aux loss knobs ---
    ap.add_argument("--lambda-force", type=float, default=0.0)
    ap.add_argument("--lambda-zero", type=float, default=0.0)
    ap.add_argument("--w-contact", type=float, default=1.0)
    ap.add_argument("--w-coast", type=float, default=1.0)
    ap.add_argument("--force-th", type=float, default=1.0)

    # NOTE: you had --force-loss-norm 200 in command line
    ap.add_argument("--force-loss-norm", type=float, default=1.0)

    # --- normalization I/O ---
    ap.add_argument("--norm-cfg", type=Path, default=None,
                    help="Optional normalization config (json). If omitted, reuse out-dir/norm_cfg.json or compute from train split.")
    ap.add_argument("--save-norm-cfg", action="store_true",
                    help="Save resolved norm_cfg.json to out-dir and embed it into checkpoints.")

    args = ap.parse_args()

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_root = Path(args.dataset_root)
    val_root = Path(args.val_root)

    # -------------------------
    # resolve norm_cfg once
    # -------------------------
    norm_cfg, norm_src = resolve_norm_cfg(
        train_root=train_root,
        out_dir=out_dir,
        use_force=bool(args.use_force),
        norm_cfg_arg=args.norm_cfg,
        save_norm_cfg_flag=bool(args.save_norm_cfg),
    )
    if bool(args.save_norm_cfg):
        print(f"[OK] saved norm_cfg.json -> {out_dir/'norm_cfg.json'} (src={norm_src})")
    else:
        print(f"[INFO] norm_cfg resolved (src={norm_src}) but not saved (use --save-norm-cfg to save)")

    # -------------------------
    # datasets / loaders
    # -------------------------
    train_npz = sorted(train_root.glob("*.npz"))
    val_npz = sorted(val_root.glob("*.npz"))
    if not train_npz:
        raise FileNotFoundError(f"no train npz: {train_root}")
    if not val_npz:
        raise FileNotFoundError(f"no val npz: {val_root}")

    ds_cfg = build_dataset_cfg(args.sequence_length, args.frame_skip, norm_cfg)

    ds_tr = EpisodeNPZDataset(train_npz, config=ds_cfg)
    ds_va = EpisodeNPZDataset(val_npz, config=ds_cfg)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fixed_length)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fixed_length)

    # infer dims from one batch (AFTER normalization is applied)
    b0 = next(iter(dl_tr))
    j_dim = int(b0["q"].shape[-1])
    a_dim = int(b0["action"].shape[-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # build model
    # -------------------------
    if args.use_force:
        if "f" not in b0:
            raise KeyError("use_force=True but batch has no 'f' (check dataset / normalization).")
        f_dim = int(b0["f"].shape[-1])

        model = WorldModelVPF(
            j_dim=j_dim,
            action_dim=a_dim,
            force_dim=f_dim,
            force_lambda_train=float(args.force_lambda_train),
            force_dropout_p=float(args.force_dropout),
            lambda_force=float(args.lambda_force),
            lambda_zero=float(args.lambda_zero),
            w_contact=float(args.w_contact),
            w_coast=float(args.w_coast),
            force_th=float(args.force_th),
        ).to(device)
        print(f"[INFO] model=VPF(in) force_dim={f_dim}")
    else:
        model = WorldModelVP(j_dim=j_dim, action_dim=a_dim).to(device)
        print("[INFO] model=VP")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------------------------
    # train loop
    # -------------------------
    best_val = float("inf")
    best_path = out_dir / "best.pt"

    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in dl_tr:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            loss, metrics = model.compute_loss(batch)

            # optional: scale loss externally (keep if you used it before; otherwise set --force-loss-norm 1.0)
            if args.use_force and args.force_loss_norm and float(args.force_loss_norm) != 1.0:
                loss = loss / float(args.force_loss_norm)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            opt.step()

            if step % int(args.log_every) == 0:
                print(f"[train] epoch={ep:03d} step={step:06d} loss={float(loss.detach().cpu()):.6f}")
            step += 1

        val_loss = evaluate(model, dl_va, device)
        print(f"[val] epoch={ep:03d} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "best_val": float(best_val),
                "epoch": int(ep),

                # IMPORTANT: embed norm cfg so run_intervention can always resolve it
                "norm_cfg": norm_cfg,
                "norm_cfg_source": norm_src,
            }
            torch.save(ckpt, best_path)
            print(f"[OK] saved best: {best_path} (val={best_val:.6f})")

    print("[DONE] best_val =", best_val)


if __name__ == "__main__":
    main()
