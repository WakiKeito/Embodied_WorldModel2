"""NPZデータの正規化・前処理を行う関数群。"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def apply_transforms(sample: Dict[str, np.ndarray], config: Dict) -> Dict[str, torch.Tensor]:
    """サンプル辞書に対して前処理を適用する。"""
    norm_cfg = config.get("normalization", {})
    keys_cfg = config.get("keys", {})
    enabled = bool(norm_cfg.get("enabled", False))

    output: Dict[str, torch.Tensor] = {}

    rgb = sample["rgb"]
    output["rgb"] = _transform_rgb(rgb)

    for key in ["q", "dq", "f"]:
        vec = sample[key]
        mean = norm_cfg.get(f"{key}_mean", 0.0)
        std = norm_cfg.get(f"{key}_std", 1.0)
        do_norm = enabled and bool(keys_cfg.get(key, {}).get("normalize", True))
        output[key] = _transform_vector(vec, mean, std, do_norm)

    for key in ["action", "block_pose"]:
        vec = sample[key]
        output[key] = torch.from_numpy(vec.astype(np.float32))

    for key in ["mass", "friction"]:
        value = sample[key]
        output[key] = torch.tensor(value, dtype=torch.float32)

    return output


def _transform_rgb(rgb: np.ndarray) -> torch.Tensor:
    """RGBをfloat32化し、[0,1]正規化とCHW変換を行う。"""
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.transpose(rgb, (0, 3, 1, 2))
    return torch.from_numpy(rgb)


def _transform_vector(
    vec: np.ndarray,
    mean: float | np.ndarray,
    std: float | np.ndarray,
    normalize: bool,
) -> torch.Tensor:
    """ベクトル系列をfloat32化し、必要に応じて正規化する。"""
    vec = vec.astype(np.float32)
    if normalize:
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)
        vec = (vec - mean_arr) / std_arr
    return torch.from_numpy(vec)
