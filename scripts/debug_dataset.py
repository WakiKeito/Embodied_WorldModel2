"""データセット読み込みの簡易デバッグスクリプト。"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from wm.data.dataset_npz import EpisodeNPZDataset


def load_yaml(path: Path) -> dict:
    """YAMLファイルを読み込む。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="NPZデータセットの読み込み確認")
    parser.add_argument(
        "--default-config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="共通設定ファイルのパス",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/push_block_npz.yaml"),
        help="データセット設定ファイルのパス",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="NPZが置かれたディレクトリ（未指定ならdefault.yamlを使用）",
    )
    args = parser.parse_args()

    default_cfg = load_yaml(args.default_config)
    data_cfg = load_yaml(args.data_config)

    dataset_root = args.dataset_root or Path(default_cfg.get("dataset_root", "datasets"))
    npz_files = sorted(dataset_root.glob("*.npz"))

    if not npz_files:
        raise FileNotFoundError(f"NPZファイルが見つかりません: {dataset_root}")

    dataset = EpisodeNPZDataset(npz_files[:1], data_cfg)
    episode = dataset[0]

    print("読み込んだエピソードのテンソルshape")
    for key, value in episode.items():
        print(f"- {key}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
