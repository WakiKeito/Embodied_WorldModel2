#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/eval_probe.py

Evaluate a trained probe (linear or ridge) on a dataset split (train / intrap / extrap).

Features:
- Correct EpisodeNPZDataset config:
    cfg = {"sequence_length":..., "frame_skip":..., "keys":..., "normalization": norm_stats}
- Ridge-aware evaluation:
    If probe metadata indicates y-transform (standardize_y / log_mass), we invert it for metrics & plots.
- Robust probe state loading:
    Accepts keys:
      A) {"model":{"lin.weight","lin.bias"}} (wrapped + prefixed)
      B) {"model":{"weight","bias"}}
      C) direct {"lin.weight","lin.bias"} or {"weight","bias"}
- PyTorch 2.6 weights_only workaround:
    Uses torch.load(..., weights_only=False) for trusted local artifacts.

NEW (requested):
- --level-plots : per-level RMSE plots for mass levels and mu levels (1D)
- --plot-2d     : 2D heatmaps (mass x mu) of per-cell RMSE (mass / mu)
  + optional annotation options: --annotate --annotate-outline --annotate-fmt "{:.3f}"

Usage:
  # VP intrap
  python tools/eval_probe.py \
    --dataset-root datasets/test_intrap \
    --ckpt outputs/ckpt_vp/best.pt \
    --probe outputs/probes/probe_vp.pt \
    --outdir outputs/probe_eval/vp_intrap \
    --level-plots \
    --plot-2d --clip --annotate --annotate-outline --annotate-fmt "{:.3f}"

  # VPF extrap
  python tools/eval_probe.py \
    --dataset-root datasets/test_extrap \
    --ckpt outputs/ckpt_vpf/best.pt \
    --probe outputs/probes/probe_vpf.pt \
    --use-force \
    --outdir outputs/probe_eval/vpf_extrap \
    --level-plots \
    --plot-2d --clip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import matplotlib.patheffects as pe

from wm.data.dataset_npz import EpisodeNPZDataset
from wm.data.collate import collate_fixed_length
from wm.models.world_model import WorldModelVP, WorldModelVPF
from analysis.summarize_latent import summarize_latent


# -------------------------
# util
# -------------------------
def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def rmse_dim(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((yhat - y) ** 2, dim=0).sqrt()


def mae_dim(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(yhat - y), dim=0)


def r2_dim(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_mean = y.mean(dim=0, keepdim=True)
    ss_res = torch.sum((y - yhat) ** 2, dim=0)
    ss_tot = torch.sum((y - y_mean) ** 2, dim=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def torch_load_trusted(path: Path, map_location="cpu"):
    """
    PyTorch 2.6 workaround: default weights_only=True can break when probe contains numpy arrays, etc.
    We set weights_only=False. Do this only for trusted local artifacts.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


# -------------------------
# norm cfg
# -------------------------
def load_norm_stats_from_ckpt_dir(ckpt_path: Path) -> Dict[str, Any]:
    """
    Priority:
      1) ckpt["norm_cfg"] (embedded)
      2) <ckpt_dir>/norm_cfg.json
      3) <ckpt_dir>/*norm*.json (fallback)
    """
    # 1) embedded
    try:
        ck = torch_load_trusted(ckpt_path, map_location="cpu")
        if isinstance(ck, dict) and isinstance(ck.get("norm_cfg", None), dict):
            return ck["norm_cfg"]
    except Exception:
        pass

    # 2) file
    d = ckpt_path.parent
    cand = d / "norm_cfg.json"
    if cand.exists():
        return _load_json(cand)

    # 3) fallback glob
    for p in d.glob("*norm*.json"):
        return _load_json(p)

    raise FileNotFoundError(f"norm_cfg not found (neither embedded nor file) under: {d}")



def build_dataset_cfg(seq_len: int, frame_skip: int, norm_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "sequence_length": int(seq_len),
        "frame_skip": int(frame_skip),
        "keys": {},
    }
    if norm_stats is not None:
        cfg["normalization"] = norm_stats
    return cfg


# -------------------------
# probe loading / transforms
# -------------------------
def extract_probe_state(probe_obj: Any) -> Dict[str, torch.Tensor]:
    """
    Return a state_dict compatible with torch.nn.Linear(in_dim, 2) -> keys: weight, bias
    """
    if isinstance(probe_obj, dict) and "model" in probe_obj and isinstance(probe_obj["model"], dict):
        st = probe_obj["model"]
    elif isinstance(probe_obj, dict):
        st = probe_obj
    else:
        raise TypeError("probe must be a dict-like object")

    if "lin.weight" in st and "lin.bias" in st:
        return {"weight": st["lin.weight"], "bias": st["lin.bias"]}
    if "weight" in st and "bias" in st:
        return {"weight": st["weight"], "bias": st["bias"]}

    raise KeyError(f"cannot find probe weights in keys={list(st.keys())[:50]}")


def _as_tensor_1d(x, device: torch.device) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().to(device).float()
    return torch.tensor(x, device=device, dtype=torch.float32)


def infer_transform_metadata(probe_obj: Dict[str, Any]) -> Dict[str, Any]:
    meta = {}
    meta["log_mass"] = bool(probe_obj.get("log_mass", False))
    meta["standardize_y"] = bool(probe_obj.get("standardize_y", False))
    meta["y_mean"] = probe_obj.get("y_mean", probe_obj.get("target_mean", None))
    meta["y_std"] = probe_obj.get("y_std", probe_obj.get("target_std", None))
    meta["target_transform"] = probe_obj.get("target_transform", None)
    meta["y_transform"] = probe_obj.get("y_transform", None)
    return meta


def invert_predictions(P_raw: torch.Tensor, meta: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """
    P_raw: (N,2) in probe output space.
    Returns P_phys: (N,2) in physical space [mass, mu].
    """
    y = P_raw

    if meta.get("standardize_y", False):
        y_mean = _as_tensor_1d(meta.get("y_mean", None), device)
        y_std = _as_tensor_1d(meta.get("y_std", None), device)
        if y_mean is not None and y_std is not None and y_mean.numel() == 2 and y_std.numel() == 2:
            y = y * y_std.view(1, 2) + y_mean.view(1, 2)

    mass = y[:, 0]
    mu = y[:, 1]

    if meta.get("log_mass", False):
        mass = torch.exp(torch.clamp(mass, min=-30.0, max=30.0))

    return torch.stack([mass, mu], dim=1)


# -------------------------
# latent extraction
# -------------------------
@torch.no_grad()
def extract_latents(
    wm: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    rep_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: (N,H) latent reps
      Y: (N,2) physical GT [mass, mu]
    """
    wm.eval()
    Xs, Ys = [], []

    for batch in loader:
        batch = _to_device(batch, device)
        out = wm(batch)
        if "h_seq" not in out:
            raise KeyError(f"world model output has no h_seq. keys={list(out.keys())}")

        h_seq = out["h_seq"]  # (B,T-1,H)
        h_rep = summarize_latent(h_seq, mode=rep_mode)  # (B,H)

        y = torch.stack([batch["mass"].float(), batch["friction"].float()], dim=1)  # (B,2)

        Xs.append(h_rep.detach().cpu())
        Ys.append(y.detach().cpu())

    return torch.cat(Xs, dim=0), torch.cat(Ys, dim=0)


# -------------------------
# plotting helpers
# -------------------------
def _robust_clip(grid: np.ndarray, qlo=2, qhi=98):
    v = grid[np.isfinite(grid)]
    if v.size == 0:
        return None, None
    vmin = float(np.percentile(v, qlo))
    vmax = float(np.percentile(v, qhi))
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax):
        return None, None
    return vmin, vmax


def _ticks(vals, max_ticks=10):
    n = len(vals)
    if n <= max_ticks:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_ticks).round().astype(int)
    lab = [f"{vals[i]:g}" for i in idx]
    return idx, lab


def _annotate_grid(ax, grid: np.ndarray, fmt: str, outline: bool, max_cells: int = 400):
    H, W = grid.shape
    total = H * W
    if total <= max_cells:
        step_i = step_j = 1
    else:
        ratio = np.sqrt(total / max_cells)
        step = int(np.ceil(ratio))
        step_i = step_j = max(1, step)

    effects = None
    if outline:
        effects = [pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()]

    for i in range(0, H, step_i):
        for j in range(0, W, step_j):
            v = grid[i, j]
            if not np.isfinite(v):
                continue
            try:
                s = fmt.format(float(v))
            except Exception:
                s = f"{float(v):.3f}"
            t = ax.text(j, i, s, ha="center", va="center", color="black", fontsize=8)
            if effects is not None:
                t.set_path_effects(effects)


def _plot_heatmap(
    masses: np.ndarray,
    mus: np.ndarray,
    grid: np.ndarray,
    title: str,
    out_path: Path,
    clip: bool,
    annotate: bool,
    annotate_fmt: str,
    annotate_outline: bool,
    annotate_max_cells: int,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmin = vmax = None
    if clip:
        vmin, vmax = _robust_clip(grid)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("friction (mu)")
    ax.set_ylabel("mass (m)")

    j_idx, j_lab = _ticks(mus, 10)
    i_idx, i_lab = _ticks(masses, 10)
    ax.set_xticks(j_idx); ax.set_xticklabels(j_lab, rotation=45, ha="right")
    ax.set_yticks(i_idx); ax.set_yticklabels(i_lab)

    fig.colorbar(im, ax=ax, shrink=0.9)

    if annotate:
        _annotate_grid(ax, grid, fmt=annotate_fmt, outline=annotate_outline, max_cells=annotate_max_cells)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("[saved]", out_path)


def _make_cell_grid(df: pd.DataFrame, value_col: str):
    masses = np.sort(pd.to_numeric(df["mass_gt"], errors="coerce").dropna().unique())
    mus    = np.sort(pd.to_numeric(df["mu_gt"], errors="coerce").dropna().unique())

    grid = np.full((len(masses), len(mus)), np.nan, dtype=np.float64)
    g = df.groupby(["mass_gt", "mu_gt"], as_index=False)[value_col].mean(numeric_only=True)

    mi = {float(m): i for i, m in enumerate(masses)}
    mj = {float(mu): j for j, mu in enumerate(mus)}
    for r in g.itertuples(index=False):
        m = float(getattr(r, "mass_gt"))
        mu = float(getattr(r, "mu_gt"))
        grid[mi[m], mj[mu]] = float(getattr(r, value_col))

    return masses, mus, grid


def _plot_level_rmse(df: pd.DataFrame, outdir: Path, model_name: str, rep_mode: str):
    """
    Per-level RMSE:
      - for each unique mass_gt: RMSE over (mass_pred - mass_gt), also mu_rmse at that mass
      - for each unique mu_gt:   RMSE over (mu_pred - mu_gt), also mass_rmse at that mu
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- by mass level ----
    gM = df.groupby("mass_gt", as_index=False).agg(
        rmse_mass=("mass_err", lambda s: float(np.sqrt(np.mean(np.square(s.to_numpy(dtype=float)))))),
        rmse_mu  =("mu_err",   lambda s: float(np.sqrt(np.mean(np.square(s.to_numpy(dtype=float)))))),
        count=("mass_err", "size"),
    ).sort_values("mass_gt")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(gM["mass_gt"].to_numpy(), gM["rmse_mass"].to_numpy(), marker="o")
    ax.set_xlabel("mass_gt level")
    ax.set_ylabel("RMSE(mass)")
    ax.set_title(f"{model_name} probe level-RMSE by mass (rep={rep_mode})")
    fig.tight_layout()
    fig.savefig(outdir / "level_rmse_by_mass.png", dpi=200)
    plt.close(fig)
    print("[saved]", outdir / "level_rmse_by_mass.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(gM["mass_gt"].to_numpy(), gM["rmse_mu"].to_numpy(), marker="o")
    ax.set_xlabel("mass_gt level")
    ax.set_ylabel("RMSE(mu)")
    ax.set_title(f"{model_name} probe level-RMSE(mu) by mass (rep={rep_mode})")
    fig.tight_layout()
    fig.savefig(outdir / "level_rmse_mu_by_mass.png", dpi=200)
    plt.close(fig)
    print("[saved]", outdir / "level_rmse_mu_by_mass.png")

    gM.to_csv(outdir / "level_metrics_by_mass.csv", index=False)

    # ---- by mu level ----
    gU = df.groupby("mu_gt", as_index=False).agg(
        rmse_mu  =("mu_err",   lambda s: float(np.sqrt(np.mean(np.square(s.to_numpy(dtype=float)))))),
        rmse_mass=("mass_err", lambda s: float(np.sqrt(np.mean(np.square(s.to_numpy(dtype=float)))))),
        count=("mu_err", "size"),
    ).sort_values("mu_gt")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(gU["mu_gt"].to_numpy(), gU["rmse_mu"].to_numpy(), marker="o")
    ax.set_xlabel("mu_gt level")
    ax.set_ylabel("RMSE(mu)")
    ax.set_title(f"{model_name} probe level-RMSE by mu (rep={rep_mode})")
    fig.tight_layout()
    fig.savefig(outdir / "level_rmse_by_mu.png", dpi=200)
    plt.close(fig)
    print("[saved]", outdir / "level_rmse_by_mu.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(gU["mu_gt"].to_numpy(), gU["rmse_mass"].to_numpy(), marker="o")
    ax.set_xlabel("mu_gt level")
    ax.set_ylabel("RMSE(mass)")
    ax.set_title(f"{model_name} probe level-RMSE(mass) by mu (rep={rep_mode})")
    fig.tight_layout()
    fig.savefig(outdir / "level_rmse_mass_by_mu.png", dpi=200)
    plt.close(fig)
    print("[saved]", outdir / "level_rmse_mass_by_mu.png")

    gU.to_csv(outdir / "level_metrics_by_mu.csv", index=False)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--use-force", action="store_true")

    ap.add_argument("--rep-mode", choices=["mean", "last"], default=None,
                    help="Override rep_mode. If omitted, use probe's saved rep_mode if present.")
    ap.add_argument("--sequence-length", type=int, default=64)
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)

    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--dump-debug", action="store_true")

    # NEW: plots
    ap.add_argument("--level-plots", action="store_true", help="save per-level RMSE plots (mass levels / mu levels)")
    ap.add_argument("--plot-2d", action="store_true", help="save 2D heatmaps (mass x mu) of per-cell RMSE")

    # NEW: 2D options
    ap.add_argument("--clip", action="store_true", help="robust clip heatmap color scale (2-98 percentile)")
    ap.add_argument("--annotate", action="store_true", help="annotate heatmap cells with values")
    ap.add_argument("--annotate-outline", action="store_true", help="white outline around black text")
    ap.add_argument("--annotate-fmt", type=str, default="{:.3f}", help='format string, e.g. "{:.3f}"')
    ap.add_argument("--annotate-max-cells", type=int, default=400,
                    help="max cells to annotate; if larger, downsample")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # load probe
    # -------------------------
    probe_obj = torch_load_trusted(args.probe, map_location="cpu")
    if not isinstance(probe_obj, dict):
        raise TypeError("probe file must contain a dict")

    rep_mode_saved = probe_obj.get("rep_mode", "mean")
    rep_mode = args.rep_mode if args.rep_mode is not None else rep_mode_saved

    probe_state = extract_probe_state(probe_obj)
    meta = infer_transform_metadata(probe_obj)

    # -------------------------
    # dataset cfg
    # -------------------------
    norm_stats = load_norm_stats_from_ckpt_dir(args.ckpt)
    ds_cfg = build_dataset_cfg(args.sequence_length, args.frame_skip, norm_stats)

    npz = sorted(args.dataset_root.glob("*.npz"))
    if not npz:
        raise FileNotFoundError(f"no npz in {args.dataset_root}")

    print(f"[INFO] dataset_root={args.dataset_root}")
    print(f"[INFO] ckpt={args.ckpt}")
    print(f"[INFO] probe={args.probe}")
    print(f"[INFO] rep_mode={rep_mode}")
    print(f"[INFO] use_force={args.use_force}")
    print(f"[INFO] norm_stats={args.ckpt.parent/'norm_cfg.json'}")
    print(f"[INFO] probe_meta: log_mass={meta.get('log_mass')} standardize_y={meta.get('standardize_y')}"
          f" has_y_mean={meta.get('y_mean') is not None} has_y_std={meta.get('y_std') is not None}")

    ds = EpisodeNPZDataset(npz, config=ds_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed_length)

    # -------------------------
    # build world model
    # -------------------------
    b0 = next(iter(loader))
    b0 = _to_device(b0, device)
    j_dim = int(b0["q"].shape[-1])
    action_dim = int(b0["action"].shape[-1])

    if args.use_force:
        if "f" not in b0:
            raise KeyError("--use-force but batch has no 'f' (check dataset and normalization).")
        force_dim = int(b0["f"].shape[-1])
        wm = WorldModelVPF(j_dim=j_dim, action_dim=action_dim, force_dim=force_dim).to(device)
        model_name = "VPF"
    else:
        wm = WorldModelVP(j_dim=j_dim, action_dim=action_dim).to(device)
        model_name = "VP"

    ckpt = torch_load_trusted(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    wm.load_state_dict(state, strict=True)
    wm.eval()

    # -------------------------
    # build linear layer from probe
    # -------------------------
    in_dim = int(probe_state["weight"].shape[1])
    lin = torch.nn.Linear(in_dim, 2).to(device)
    lin.load_state_dict(probe_state, strict=True)
    lin.eval()

    # -------------------------
    # extract latents + predict
    # -------------------------
    X_cpu, Y_cpu = extract_latents(wm, loader, device, rep_mode)
    X = X_cpu.to(device)
    Y = Y_cpu.to(device)  # physical GT

    with torch.no_grad():
        P_raw = lin(X)

    P_phys = invert_predictions(P_raw, meta=meta, device=device)

    # metrics in physical space
    rmse = rmse_dim(P_phys, Y).detach().cpu().numpy()
    mae = mae_dim(P_phys, Y).detach().cpu().numpy()
    r2 = r2_dim(P_phys, Y).detach().cpu().numpy()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    txt = []
    txt.append(f"model={model_name}")
    txt.append(f"dataset={args.dataset_root}")
    txt.append(f"N={Y.shape[0]}")
    txt.append(f"rep_mode={rep_mode}")
    txt.append(f"log_mass={meta.get('log_mass', False)}")
    txt.append(f"standardize_y={meta.get('standardize_y', False)}")
    txt.append(f"has_y_mean={meta.get('y_mean') is not None}")
    txt.append(f"has_y_std={meta.get('y_std') is not None}")
    txt.append(f"RMSE(mass)={rmse[0]:.6f} RMSE(mu)={rmse[1]:.6f}")
    txt.append(f"MAE(mass) ={mae[0]:.6f} MAE(mu) ={mae[1]:.6f}")
    txt.append(f"R2(mass)  ={r2[0]:.6f}")
    txt.append(f"R2(mu)    ={r2[1]:.6f}")

    (outdir / "metrics.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n".join(txt))

    # -------------------------
    # plots + tables
    # -------------------------
    Pcpu = P_phys.detach().cpu().numpy()
    Ycpu = Y.detach().cpu().numpy()

    def scatter(idx: int, name: str):
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.scatter(Ycpu[:, idx], Pcpu[:, idx], s=12, alpha=0.7)
        lo = float(min(Ycpu[:, idx].min(), Pcpu[:, idx].min()))
        hi = float(max(Ycpu[:, idx].max(), Pcpu[:, idx].max()))
        ax.plot([lo, hi], [lo, hi])
        ax.set_xlabel("GT")
        ax.set_ylabel("Pred")
        ax.set_title(f"{model_name} probe: {name} (rep={rep_mode})")
        fig.tight_layout()
        fig.savefig(outdir / f"scatter_{name}.png", dpi=200)
        plt.close(fig)

    scatter(0, "mass")
    scatter(1, "mu")

    df = pd.DataFrame({
        "mass_gt": Ycpu[:, 0], "mu_gt": Ycpu[:, 1],
        "mass_pred": Pcpu[:, 0], "mu_pred": Pcpu[:, 1],
        "mass_err": (Pcpu[:, 0] - Ycpu[:, 0]),
        "mu_err": (Pcpu[:, 1] - Ycpu[:, 1]),
        "abs_mass_err": np.abs(Pcpu[:, 0] - Ycpu[:, 0]),
        "abs_mu_err": np.abs(Pcpu[:, 1] - Ycpu[:, 1]),
        "sq_mass_err": np.square(Pcpu[:, 0] - Ycpu[:, 0]),
        "sq_mu_err": np.square(Pcpu[:, 1] - Ycpu[:, 1]),
    })
    df.to_csv(outdir / "pred_table.csv", index=False)

    # -------------------------
    # NEW: level plots
    # -------------------------
    if args.level_plots:
        _plot_level_rmse(df, outdir / "level_plots", model_name=model_name, rep_mode=rep_mode)

    # -------------------------
    # NEW: 2D heatmaps (mass x mu)
    # -------------------------
    if args.plot_2d:
        # per-cell RMSE
        cell = df.groupby(["mass_gt", "mu_gt"], as_index=False).agg(
            rmse_mass=("sq_mass_err", lambda s: float(np.sqrt(np.mean(s.to_numpy(dtype=float))))),
            rmse_mu=("sq_mu_err", lambda s: float(np.sqrt(np.mean(s.to_numpy(dtype=float))))),
            count=("sq_mass_err", "size"),
        )
        cell.to_csv(outdir / "grid_metrics_probe.csv", index=False)

        masses, mus, grid_mass = _make_cell_grid(cell.rename(columns={"rmse_mass": "val"}), "val")
        masses2, mus2, grid_mu = _make_cell_grid(cell.rename(columns={"rmse_mu": "val"}), "val")

        figdir = outdir / "grid_plots"
        figdir.mkdir(parents=True, exist_ok=True)

        _plot_heatmap(
            masses, mus, grid_mass,
            title=f"{model_name} probe per-cell RMSE(mass) (rep={rep_mode})",
            out_path=figdir / "heatmap_rmse_mass.png",
            clip=bool(args.clip),
            annotate=bool(args.annotate),
            annotate_fmt=str(args.annotate_fmt),
            annotate_outline=bool(args.annotate_outline),
            annotate_max_cells=int(args.annotate_max_cells),
        )
        _plot_heatmap(
            masses2, mus2, grid_mu,
            title=f"{model_name} probe per-cell RMSE(mu) (rep={rep_mode})",
            out_path=figdir / "heatmap_rmse_mu.png",
            clip=bool(args.clip),
            annotate=bool(args.annotate),
            annotate_fmt=str(args.annotate_fmt),
            annotate_outline=bool(args.annotate_outline),
            annotate_max_cells=int(args.annotate_max_cells),
        )

    # -------------------------
    # optional debug dump (raw/transformed) — kept minimal
    # -------------------------
    if args.dump_debug:
        Praw_cpu = P_raw.detach().cpu().numpy()
        df2 = pd.DataFrame({"y0_pred_raw": Praw_cpu[:, 0], "y1_pred_raw": Praw_cpu[:, 1]})
        df2.to_csv(outdir / "debug_pred_raw.csv", index=False)
        print("[OK] saved debug_pred_raw.csv")

    print("[OK] saved:", outdir)


if __name__ == "__main__":
    main()
