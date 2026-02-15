#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tools/genmap_split_plot.py
"""
Plot heatmaps from grid_metrics.csv produced by genmap_split_agg.py.

- Default: plot 4 heatmaps:
    rmse_block_full_contact / coast
    rmse_block_open_contact / coast
- Optional: if --csv2 is provided, also plot DIFF heatmaps:
    (csv2 - csv1) for the same four metrics.

Annotate (optional):
  --annotate
  --annotate-outline
  --annotate-fmt "{:.3f}"

Usage:
  python tools/genmap_split_plot.py --csv A/grid_metrics.csv --outdir A/figs --clip
  python tools/genmap_split_plot.py --csv VP/grid_metrics.csv --csv2 VPF/grid_metrics.csv --outdir diff/figs --clip --diff
  python tools/genmap_split_plot.py --csv A/grid_metrics.csv --outdir A/figs --clip --annotate --annotate-outline --annotate-fmt "{:.3f}"
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for outline text
import matplotlib.patheffects as pe


TARGETS = [
    ("rmse_block_full_contact", "heatmap_full_contact.png", "GenMap: FULL / contact (RMSE block xy)"),
    ("rmse_block_full_coast",   "heatmap_full_coast.png",   "GenMap: FULL / coast (RMSE block xy)"),
    ("rmse_block_open_contact", "heatmap_open_contact.png", "GenMap: OPEN / contact (RMSE block xy)"),
    ("rmse_block_open_coast",   "heatmap_open_coast.png",   "GenMap: OPEN / coast (RMSE block xy)"),
]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def pivot_grid(df: pd.DataFrame, value_col: str):
    if "mass" not in df.columns or "friction" not in df.columns:
        raise KeyError(f"need columns mass, friction in csv. columns={list(df.columns)}")
    if value_col not in df.columns:
        raise KeyError(f"missing {value_col}. columns={list(df.columns)}")

    masses = np.sort(pd.to_numeric(df["mass"], errors="coerce").dropna().unique())
    mus    = np.sort(pd.to_numeric(df["friction"], errors="coerce").dropna().unique())

    grid = np.full((len(masses), len(mus)), np.nan, dtype=np.float64)

    # mean over duplicates (if any)
    g = df.groupby(["mass", "friction"], as_index=False)[value_col].mean(numeric_only=True)
    mi = {float(m): i for i, m in enumerate(masses)}
    mj = {float(mu): j for j, mu in enumerate(mus)}

    for r in g.itertuples(index=False):
        m = float(getattr(r, "mass"))
        mu = float(getattr(r, "friction"))
        i = mi[m]; j = mj[mu]
        grid[i, j] = float(getattr(r, value_col))

    return masses, mus, grid


def robust_clip(grid: np.ndarray, qlo=2, qhi=98):
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


def _annotate_grid(
    ax,
    grid: np.ndarray,
    fmt: str,
    outline: bool,
    max_cells: int = 400,
):
    """
    Put text on each cell. If the grid is too large, downsample indices to keep total <= max_cells.
    - black text
    - optional white outline
    """
    H, W = grid.shape
    total = H * W

    # decide sampling step if too many cells
    if total <= max_cells:
        step_i = step_j = 1
    else:
        # choose steps so that (H/si)*(W/sj) <= max_cells
        # simple symmetric strategy
        ratio = np.sqrt(total / max_cells)
        step = int(np.ceil(ratio))
        step_i = step_j = max(1, step)

    # path effects for white outline
    if outline:
        effects = [pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()]
    else:
        effects = None

    for i in range(0, H, step_i):
        for j in range(0, W, step_j):
            v = grid[i, j]
            if not np.isfinite(v):
                continue
            try:
                s = fmt.format(float(v))
            except Exception:
                s = f"{float(v):.3f}"

            t = ax.text(
                j, i, s,
                ha="center", va="center",
                color="black",
                fontsize=8,
            )
            if effects is not None:
                t.set_path_effects(effects)


def plot_one(
    masses,
    mus,
    grid,
    title,
    out_path: Path,
    clip: bool,
    cmap: str = None,
    annotate: bool = False,
    annotate_fmt: str = "{:.3f}",
    annotate_outline: bool = True,
    annotate_max_cells: int = 400,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vmin = vmax = None
    if clip:
        vmin, vmax = robust_clip(grid)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("friction (mu)")
    ax.set_ylabel("mass (m)")

    j_idx, j_lab = _ticks(mus, 10)
    i_idx, i_lab = _ticks(masses, 10)
    ax.set_xticks(j_idx); ax.set_xticklabels(j_lab, rotation=45, ha="right")
    ax.set_yticks(i_idx); ax.set_yticklabels(i_lab)

    fig.colorbar(im, ax=ax, shrink=0.9)

    if annotate:
        _annotate_grid(
            ax=ax,
            grid=grid,
            fmt=annotate_fmt,
            outline=annotate_outline,
            max_cells=int(annotate_max_cells),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("[saved]", out_path)


def align_grids(m1, mu1, g1, m2, mu2, g2):
    # Align by intersection to avoid mismatched axes
    m_common = np.array(sorted(set(map(float, m1)) & set(map(float, m2))), dtype=float)
    mu_common = np.array(sorted(set(map(float, mu1)) & set(map(float, mu2))), dtype=float)
    if m_common.size == 0 or mu_common.size == 0:
        raise RuntimeError("No common (mass, friction) axis between csv and csv2.")

    def subgrid(masses, mus, grid, m_common, mu_common):
        mi = {float(m): i for i, m in enumerate(masses)}
        mj = {float(mu): j for j, mu in enumerate(mus)}
        out = np.full((len(m_common), len(mu_common)), np.nan, dtype=np.float64)
        for i, m in enumerate(m_common):
            for j, mu in enumerate(mu_common):
                out[i, j] = grid[mi[m], mj[mu]]
        return out

    sg1 = subgrid(m1, mu1, g1, m_common, mu_common)
    sg2 = subgrid(m2, mu2, g2, m_common, mu_common)
    return m_common, mu_common, sg1, sg2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str, help="grid_metrics.csv (baseline)")
    ap.add_argument("--csv2", default=None, type=str, help="optional second grid_metrics.csv to plot DIFF (csv2-csv)")
    ap.add_argument("--outdir", default="outputs/genmap_figs_split", type=str)
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--diff", action="store_true", help="if set with --csv2, also plot diff heatmaps (csv2-csv)")

    # ---- NEW: annotation options ----
    ap.add_argument("--annotate", action="store_true", help="annotate each cell with value")
    ap.add_argument("--annotate-outline", action="store_true", help="white outline around black text")
    ap.add_argument("--annotate-fmt", type=str, default="{:.3f}", help='format string, e.g. "{:.3f}"')
    ap.add_argument("--annotate-max-cells", type=int, default=400,
                    help="max cells to annotate; if larger, downsample to keep it readable/fast")

    args = ap.parse_args()

    df1 = _coerce_numeric(pd.read_csv(args.csv))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) base plots
    for col, fname, title in TARGETS:
        m, mu, grid = pivot_grid(df1, col)
        plot_one(
            m, mu, grid,
            title,
            outdir / fname,
            clip=bool(args.clip),
            cmap=None,
            annotate=bool(args.annotate),
            annotate_fmt=str(args.annotate_fmt),
            annotate_outline=bool(args.annotate_outline),
            annotate_max_cells=int(args.annotate_max_cells),
        )

    # 2) diff plots (optional)
    if args.csv2 is not None and args.diff:
        df2 = _coerce_numeric(pd.read_csv(args.csv2))
        diffdir = outdir / "diff_csv2_minus_csv"
        diffdir.mkdir(parents=True, exist_ok=True)

        for col, fname, title in TARGETS:
            m1, mu1, g1 = pivot_grid(df1, col)
            m2, mu2, g2 = pivot_grid(df2, col)
            mC, muC, sg1, sg2 = align_grids(m1, mu1, g1, m2, mu2, g2)
            d = sg2 - sg1
            plot_one(
                mC, muC, d,
                f"DIFF (csv2-csv): {title}",
                diffdir / fname,
                clip=bool(args.clip),
                cmap=None,
                annotate=bool(args.annotate),
                annotate_fmt=str(args.annotate_fmt),
                annotate_outline=bool(args.annotate_outline),
                annotate_max_cells=int(args.annotate_max_cells),
            )

    print("[DONE]")


if __name__ == "__main__":
    main()
