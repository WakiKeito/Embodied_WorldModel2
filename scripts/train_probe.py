"""
Probe: world model の latent から mass / friction を回帰する。

- VP / VPF を切り替え可能
- h_seq -> h_rep (mean/last)
- 線形回帰（ridgeなしの最小）を PyTorch で実装
- 指標: RMSE / MAE / R2
- まず「動く」こと優先（世界モデルの学習済み重みが無くても動く）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from wm.data.dataset_npz import EpisodeNPZDataset
from wm.data.collate import collate_fixed_length
from wm.models.world_model import WorldModelVP, WorldModelVPF
from analysis.summarize_latent import summarize_latent


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val].tolist()
    tr_idx = idx[n_val:].tolist()
    return tr_idx, val_idx


def to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def rmse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_hat - y) ** 2).sqrt()


def mae(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_hat - y))


def r2_score(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # y, y_hat: (N, D)
    y_mean = y.mean(dim=0, keepdim=True)
    ss_res = torch.sum((y - y_hat) ** 2, dim=0)
    ss_tot = torch.sum((y - y_mean) ** 2, dim=0) + 1e-12
    return 1.0 - ss_res / ss_tot


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


@torch.no_grad()
def extract_latent_and_targets(
    model,
    loader: DataLoader,
    device: torch.device,
    rep_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        X: (N, H) latent reps
        Y: (N, 2) targets [mass, friction]
    """
    X_list, Y_list = [], []

    model.eval()

    for batch in loader:
        batch = to_device(batch, device)

        outputs = model(batch)  # outputs["h_seq"] exists
        h_seq = outputs["h_seq"]  # (B,T,H)

        h_rep = summarize_latent(h_seq, mode=rep_mode)  # (B,H)

        mass = batch["mass"].view(-1, 1).float()
        friction = batch["friction"].view(-1, 1).float()
        y = torch.cat([mass, friction], dim=1)  # (B,2)

        X_list.append(h_rep.detach().cpu())
        Y_list.append(y.detach().cpu())

    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    return X, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--use-force", action="store_true")
    parser.add_argument("--rep-mode", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/probe_metrics.txt"))
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()

    npz_files = sorted(args.dataset_root.glob("*.npz"))
    if len(npz_files) < 2:
        raise ValueError("probeには少なくとも2エピソード必要です（できれば複数(m,μ)）。")

    # Dataset（EpisodeNPZDataset は npz_paths を渡す形式）
    cfg = {"sequence_length": 64, "frame_skip": 1, "keys": {}}
    dataset = EpisodeNPZDataset(npz_files, config=cfg)

    tr_idx, val_idx = split_indices(len(dataset), args.val_ratio, args.seed)

    # Subset相当（最小実装）
    tr_paths = [npz_files[i] for i in tr_idx]
    va_paths = [npz_files[i] for i in val_idx]
    tr_ds = EpisodeNPZDataset(tr_paths, config=cfg)
    va_ds = EpisodeNPZDataset(va_paths, config=cfg)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fixed_length)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed_length)

    # 1バッチで次元推定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch0 = next(iter(tr_loader))
    batch0 = to_device(batch0, device)
    j_dim = batch0["q"].shape[-1]
    action_dim = batch0["action"].shape[-1]

    if args.use_force:
        wm = WorldModelVPF(j_dim=j_dim, action_dim=action_dim).to(device)
        print("[INFO] Probe on VPF latent")
    else:
        wm = WorldModelVP(j_dim=j_dim, action_dim=action_dim).to(device)
        print("[INFO] Probe on VP latent")

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    wm.load_state_dict(state, strict=True)
    wm.eval()

    # latent抽出（世界モデルは固定）
    X_tr, Y_tr = extract_latent_and_targets(wm, tr_loader, device, args.rep_mode)
    X_va, Y_va = extract_latent_and_targets(wm, va_loader, device, args.rep_mode)

    X_tr = X_tr.to(device)
    Y_tr = Y_tr.to(device)
    X_va = X_va.to(device)
    Y_va = Y_va.to(device)

    probe = LinearProbe(in_dim=X_tr.shape[1], out_dim=2).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_va = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        probe.train()
        pred = probe(X_tr)
        loss = torch.mean((pred - Y_tr) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 20 == 0 or ep == 1:
            probe.eval()
            with torch.no_grad():
                pred_va = probe(X_va)
                rmse_va = rmse(pred_va, Y_va).item()
            if rmse_va < best_va:
                best_va = rmse_va
                best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            print(f"[epoch {ep:4d}] train_mse={loss.item():.6f}  val_rmse={rmse_va:.6f}")

    if best_state is not None:
        probe.load_state_dict(best_state)

    probe.eval()
    with torch.no_grad():
        pred_va = probe(X_va)
        rmse_all = rmse(pred_va, Y_va)
        mae_all = mae(pred_va, Y_va)
        r2 = r2_score(pred_va, Y_va)

        # 次元ごと（mass, friction）
        rmse_dim = torch.mean((pred_va - Y_va) ** 2, dim=0).sqrt()
        mae_dim = torch.mean(torch.abs(pred_va - Y_va), dim=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    txt = []
    txt.append(f"model: {'VPF' if args.use_force else 'VP'}")
    txt.append(f"rep_mode: {args.rep_mode}")
    txt.append(f"N_train={X_tr.shape[0]} N_val={X_va.shape[0]}")
    txt.append(f"RMSE(all): {rmse_all.item():.6f}")
    txt.append(f"MAE(all):  {mae_all.item():.6f}")
    txt.append(f"R2(mass):     {r2[0].item():.6f}")
    txt.append(f"R2(friction): {r2[1].item():.6f}")
    txt.append(f"RMSE(mass):     {rmse_dim[0].item():.6f}")
    txt.append(f"RMSE(friction): {rmse_dim[1].item():.6f}")
    txt.append(f"MAE(mass):     {mae_dim[0].item():.6f}")
    txt.append(f"MAE(friction): {mae_dim[1].item():.6f}")

    out_str = "\n".join(txt)
    args.out.write_text(out_str + "\n", encoding="utf-8")
    print("\n" + out_str)
    print(f"[OK] saved: {args.out}")


if __name__ == "__main__":
    main()
