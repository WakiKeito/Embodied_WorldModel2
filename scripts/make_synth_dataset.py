"""
疑似データ生成（パイプライン動作確認用）

既存の1 episode npz をテンプレとして、mass/frictionだけ変えた npz を量産する。
必要に応じて連続値ノイズも少量追加できる（デフォルトは0）。

使い方例:
  PYTHONPATH=src python -m scripts.make_synth_dataset \
    --template datasets/raw/episode_000000.npz \
    --out-dir datasets/raw_synth \
    --masses 0.5 0.8 1.0 1.2 1.5 \
    --frictions 0.1 0.3 0.5 0.7 0.9 \
    --rep 1 \
    --noise-q 0.0 --noise-dq 0.0 --noise-f 0.0 --noise-pose 0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np


REQUIRED_KEYS = ["rgb", "q", "dq", "f", "action", "block_pose", "mass", "friction"]


def load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    missing = [k for k in REQUIRED_KEYS if k not in out]
    if missing:
        raise KeyError(f"templateに必須キーが足りません: {missing}")
    return out


def add_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return x
    # float32 にして加算（dtype維持したいので最後に戻す）
    y = x.astype(np.float32) + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return y.astype(x.dtype, copy=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--template", type=Path, required=True, help="元にするnpz（1本）")
    p.add_argument("--out-dir", type=Path, default=Path("datasets/raw_synth"))
    p.add_argument("--masses", type=float, nargs="+", required=True)
    p.add_argument("--frictions", type=float, nargs="+", required=True)
    p.add_argument("--rep", type=int, default=1, help="各(m,μ)あたり何本複製するか")
    p.add_argument("--seed", type=int, default=0)

    # 任意：見た目だけ少し変えたい場合（デフォ0）
    p.add_argument("--noise-q", type=float, default=0.0)
    p.add_argument("--noise-dq", type=float, default=0.0)
    p.add_argument("--noise-f", type=float, default=0.0)
    p.add_argument("--noise-pose", type=float, default=0.0)

    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    tmpl = load_npz(args.template)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # rgb は uint8 のままにする（仕様維持）
    rgb = tmpl["rgb"]
    q = tmpl["q"]
    dq = tmpl["dq"]
    f = tmpl["f"]
    action = tmpl["action"]
    block_pose = tmpl["block_pose"]

    saved = 0
    for m in args.masses:
        for mu in args.frictions:
            for k in range(args.rep):
                out = {}
                out["rgb"] = rgb  # そのまま
                out["q"] = add_noise(q, args.noise_q, rng)
                out["dq"] = add_noise(dq, args.noise_dq, rng)
                out["f"] = add_noise(f, args.noise_f, rng)
                out["action"] = action  # そのまま
                out["block_pose"] = add_noise(block_pose, args.noise_pose, rng)

                # スカラーはfloat32で保存（仕様）
                out["mass"] = np.array(m, dtype=np.float32)
                out["friction"] = np.array(mu, dtype=np.float32)

                fname = f"episode_m{m:.3f}_mu{mu:.3f}_{k:03d}.npz"
                path = args.out_dir / fname
                np.savez(path, **out)
                saved += 1

    print(f"[OK] saved {saved} files to: {args.out_dir}")
    print("[NOTE] これはパイプライン検証用の疑似データです（研究主張には使用しないでください）。")


if __name__ == "__main__":
    main()
