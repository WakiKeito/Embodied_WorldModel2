"""NPZデータの正規化・前処理を行う関数群。"""

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import torch


# 追加で出力に載せたい時系列キー（あれば使う / なければゼロ埋め）
# run_intervention の air mask がここを参照するので、最低限 is_air_force/phase_id は欲しい。
OPTIONAL_TIME_KEYS = [
    "is_contact",     # 0/1
    "is_air_force",   # 0/1
    "phase_id",       # int
]


def apply_transforms(sample: Dict[str, np.ndarray], config: Dict) -> Dict[str, torch.Tensor]:
    """サンプル辞書に対して前処理を適用する。"""
    norm_cfg = config.get("normalization", {}) or {}
    keys_cfg = config.get("keys", {}) or {}
    enabled = bool(norm_cfg.get("enabled", False))

    output: Dict[str, torch.Tensor] = {}

    # ---- rgb ----
    rgb = sample["rgb"]
    output["rgb"] = _transform_rgb(rgb)

    # ---- vectors (q/dq/f are common; f is optional) ----
    for key in ["q", "dq", "f"]:
        if key not in sample:
            # f が無い VP データでも動くようにする
            continue

        vec = sample[key]
        mean = norm_cfg.get(f"{key}_mean", 0.0)
        std = norm_cfg.get(f"{key}_std", 1.0)
        do_norm = enabled and bool(keys_cfg.get(key, {}).get("normalize", True))
        output[key] = _transform_vector(vec, mean, std, do_norm)

    # ---- other arrays ----
    for key in ["action", "block_pose"]:
        if key not in sample:
            raise KeyError(f"missing required key in npz: {key}")
        vec = sample[key]
        output[key] = torch.from_numpy(np.asarray(vec, dtype=np.float32))

    # ---- scalars ----
    for key in ["mass", "friction"]:
        if key not in sample:
            raise KeyError(f"missing required key in npz: {key}")
        value = sample[key]
        # value が numpy scalar / (1,) / float どれでもOKにする
        value = float(np.asarray(value).reshape(-1)[0])
        output[key] = torch.tensor(value, dtype=torch.float32)

    # ---- optional time keys: always return to keep collate stable ----
    # ここが重要：存在しないエピソードでもゼロ埋めして常にキーを返す。
    T = int(output["q"].shape[0])  # sequence_length after slicing
    for k in OPTIONAL_TIME_KEYS:
        output[k] = _get_time_key(sample, k, T=T, default=0)

    return output


def _transform_rgb(rgb: np.ndarray) -> torch.Tensor:
    """RGBをfloat32化し、[0,1]正規化とCHW変換を行う。"""
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    # (T,H,W,C) -> (T,C,H,W)
    rgb = np.transpose(rgb, (0, 3, 1, 2))
    return torch.from_numpy(rgb)


def _transform_vector(
    vec: np.ndarray,
    mean: float | np.ndarray,
    std: float | np.ndarray,
    normalize: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """ベクトル系列をfloat32化し、必要に応じて正規化する。std=0対策込み。"""
    vec = np.asarray(vec, dtype=np.float32)
    if normalize:
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)
        # std=0 で壊れないように clamp
        std_arr = np.maximum(std_arr, eps).astype(np.float32)
        vec = (vec - mean_arr) / std_arr
    return torch.from_numpy(vec)


def _get_time_key(sample: Dict[str, Any], name: str, *, T: int, default: int = 0) -> torch.Tensor:
    """
    時系列キーを (T,) float32 tensor で返す。
    無ければ default でゼロ埋め。
    """
    if name not in sample:
        return torch.full((T,), float(default), dtype=torch.float32)

    arr = np.asarray(sample[name])
    # (T,1) -> (T,)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    # scalar -> (T,) にブロードキャスト（念のため）
    if arr.ndim == 0:
        arr = np.full((T,), float(arr), dtype=np.float32)

    arr = arr.astype(np.float32)

    # 長さが違う場合は切る/パディング（データが変でも落とさない）
    if arr.shape[0] < T:
        pad = np.full((T - arr.shape[0],), float(default), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > T:
        arr = arr[:T]

    return torch.from_numpy(arr)
