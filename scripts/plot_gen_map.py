"""
Generalization Map (mass, friction) のCSVを読み、heatmap画像を保存する。

入力:
  outputs/genmap_vpf.csv (または vp.csv など)
  columns: mass, friction, rmse_q, rmse_block_pose, n_episodes

出力:
  outputs/genmap_vpf_rmse_q.png
  outputs/genmap_vpf_rmse_block_pose.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"CSVが空です: {path}")
    return rows


def build_grid(
    rows: List[Dict[str, str]],
    value_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    rows から mass/friction の格子を作り、value_key の2D配列を返す。
    戻り値:
      masses: (M,)
      frictions: (N,)
      grid: (N, M)  ※ y=friction, x=mass になるように配置
    """
    masses = sorted({float(r["mass"]) for r in rows})
    frictions = sorted({float(r["friction"]) for r in rows})

    m_index = {m: i for i, m in enumerate(masses)}
    mu_index = {mu: i for i, mu in enumerate(frictions)}

    grid = np.full((len(frictions), len(masses)), np.nan, dtype=np.float32)

    for r in rows:
        m = float(r["mass"])
        mu = float(r["friction"])
        v = float(r[value_key])
        grid[mu_index[mu], m_index[m]] = v

    return np.array(masses), np.array(frictions), grid


def plot_heatmap(
    masses: np.ndarray,
    frictions: np.ndarray,
    grid: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # imshow の extent は (xmin, xmax, ymin, ymax)
    # friction を縦軸、mass を横軸にする
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = plt.gca()

    # 値が欠けている場合があるので、NaNは表示から除外
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[masses.min(), masses.max(), frictions.min(), frictions.max()],
    )

    ax.set_xlabel("mass")
    ax.set_ylabel("friction")
    ax.set_title(title)

    # 目盛り：値が多すぎると見づらいので最大10個程度に間引き
    def choose_ticks(vals: np.ndarray, max_ticks: int = 10) -> np.ndarray:
        if len(vals) <= max_ticks:
            return vals
        idx = np.linspace(0, len(vals) - 1, max_ticks).round().astype(int)
        return vals[idx]

    ax.set_xticks(choose_ticks(masses))
    ax.set_yticks(choose_ticks(frictions))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("value")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/genmap.csv"),
        help="入力CSV（mass, friction, rmse_* を含む）",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="出力ファイルのprefix（未指定ならCSV名から自動生成）",
    )
    args = parser.parse_args()

    rows = read_csv(args.csv)

    # 出力prefix
    if args.out_prefix is None:
        # outputs/genmap_vpf.csv -> outputs/genmap_vpf
        out_prefix = args.csv.with_suffix("")
    else:
        out_prefix = args.out_prefix

    # rmse_q
    masses, frictions, grid_q = build_grid(rows, value_key="rmse_q")
    plot_heatmap(
        masses,
        frictions,
        grid_q,
        title=f"Generalization Map: RMSE(q)\n{args.csv.name}",
        out_path=Path(str(out_prefix) + "_rmse_q.png"),
    )

    # rmse_block_pose
    if "rmse_block_pose" in rows[0]:
        masses, frictions, grid_b = build_grid(rows, value_key="rmse_block_pose")
        plot_heatmap(
            masses,
            frictions,
            grid_b,
            title=f"Generalization Map: RMSE(block_pose)\n{args.csv.name}",
            out_path=Path(str(out_prefix) + "_rmse_block_pose.png"),
        )

    print(f"[OK] saved: {str(out_prefix)}_rmse_q.png")
    if "rmse_block_pose" in rows[0]:
        print(f"[OK] saved: {str(out_prefix)}_rmse_block_pose.png")


if __name__ == "__main__":
    main()
