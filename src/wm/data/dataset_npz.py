# dataset_npz.py
"""NPZエピソードデータセットを読み込むためのDataset実装。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import apply_transforms


REQUIRED_KEYS = [
    "rgb",
    "q",
    "dq",
    "f",
    "action",
    "block_pose",
    "mass",
    "friction",
]

# 返したい “任意” の時系列キー（NPZにあれば一緒に返す）
OPTIONAL_TIME_KEYS = [
    "is_contact",
    "phase_id",
    "action_mode_id",
    "block_vel",
    "ee_pos",
    "target_pos",
    "ee_err",
    "contacts_table",
    "contacts_block",
    "max_nf_table",
    "max_nf_block",
    "q_jump",
    "is_kick",
    "ee_speed",
    "f_push",
    "v_cmd",
    "f_ext",
    "is_air_force",
    "air_lifted",
]


@dataclass
class EpisodeSpec:
    """エピソード内で共有されるshape情報。"""

    t: int
    j: Optional[int] = None
    a: Optional[int] = None
    f_dim: Optional[int] = None


class EpisodeNPZDataset(Dataset):
    """1エピソードNPZを読み込むDataset。"""

    def __init__(
        self,
        npz_paths: Iterable[str | Path],
        config: Dict,
        transform: Optional[Callable[[Dict[str, np.ndarray], Dict], Dict]] = None,
    ) -> None:
        self.config = config
        self.sequence_length = int(config.get("sequence_length", 1))
        self.frame_skip = int(config.get("frame_skip", 1))
        self.transform = transform or apply_transforms

        paths: List[Path] = []
        for p in npz_paths:
            path = Path(p)
            if path.is_dir():
                paths.extend(sorted(path.glob("*.npz")))
            else:
                paths.append(path)
        self.npz_paths = paths

        if not self.npz_paths:
            raise ValueError("NPZファイルが見つかりません。")

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        npz_path = self.npz_paths[index]
        with np.load(npz_path, allow_pickle=True) as data:
            self._validate_required_keys(data, npz_path)
            spec = self._build_spec(data, npz_path)
            self._validate_shapes_and_dtypes(data, spec, npz_path)
            episode = self._slice_episode(data, spec, npz_path)

        return self.transform(episode, self.config)

    def _validate_required_keys(self, data: np.lib.npyio.NpzFile, npz_path: Path) -> None:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(f"必須キーが不足しています: {missing} ({npz_path})")

    def _build_spec(self, data: np.lib.npyio.NpzFile, npz_path: Path) -> EpisodeSpec:
        t: Optional[int] = None
        j: Optional[int] = None
        a: Optional[int] = None
        f_dim: Optional[int] = None

        for key in ["rgb", "q", "dq", "f", "action", "block_pose"]:
            arr = data[key]
            if arr.ndim < 1:
                raise ValueError(f"{key} のshapeが不正です: {arr.shape} ({npz_path})")

            if t is None:
                t = arr.shape[0]
            elif arr.shape[0] != t:
                raise ValueError(f"時系列長Tが一致しません: {key}={arr.shape[0]} != {t} ({npz_path})")

            if key in {"q", "dq"}:
                j = self._check_dim(key, arr, 1, j, "J", npz_path)

            if key == "f":
                f_dim = self._check_dim(key, arr, 1, f_dim, "F", npz_path)
                if f_dim != 3:
                    raise ValueError(f"force次元は3を想定: f={f_dim} ({npz_path})")

            if key == "action":
                a = self._check_dim(key, arr, 1, a, "A", npz_path)

        if t is None:
            raise ValueError(f"Tが取得できません ({npz_path})")

        return EpisodeSpec(t=t, j=j, a=a, f_dim=f_dim)

    def _normalize_key_cfg(self, key_cfg: Any) -> Dict[str, Any]:
        """
        keys_cfg[key] が dict ではなく str の shorthand で入ってきても落ちないようにする。
        例:
          "rgb": "uint8_or_float32"  -> {"dtype":"uint8_or_float32"}
          "q": "float32"             -> {"dtype":"float32"}
        """
        if key_cfg is None:
            return {}
        if isinstance(key_cfg, dict):
            return key_cfg
        if isinstance(key_cfg, str):
            return {"dtype": key_cfg}
        # 予期しない型は無視（安全側）
        return {}

    def _validate_shapes_and_dtypes(
        self,
        data: np.lib.npyio.NpzFile,
        spec: EpisodeSpec,
        npz_path: Path,
    ) -> None:
        keys_cfg = self.config.get("keys", {})
        if not isinstance(keys_cfg, dict):
            # 壊れていても落とさない（評価スクリプトを動かすのが優先）
            return

        for key in REQUIRED_KEYS:
            arr = data[key]
            key_cfg = self._normalize_key_cfg(keys_cfg.get(key))

            expected = key_cfg.get("shape")
            if expected is not None:
                resolved = []
                for dim in expected:
                    if dim == "T":
                        resolved.append(spec.t)
                    elif dim == "J":
                        if spec.j is None:
                            raise ValueError(f"Jが未定義です ({npz_path})")
                        resolved.append(spec.j)
                    elif dim == "A":
                        if spec.a is None:
                            raise ValueError(f"Aが未定義です ({npz_path})")
                        resolved.append(spec.a)
                    elif dim == "F":
                        if spec.f_dim is None:
                            raise ValueError(f"Fが未定義です ({npz_path})")
                        resolved.append(spec.f_dim)
                    else:
                        resolved.append(dim)

                if tuple(arr.shape) != tuple(resolved):
                    raise ValueError(
                        f"{key} のshapeが不一致です: 期待={resolved}, 実際={arr.shape} ({npz_path})"
                    )

            expected_dtype = key_cfg.get("dtype")
            if expected_dtype:
                self._validate_dtype(key, arr.dtype, expected_dtype, npz_path)

    def _validate_dtype(
        self,
        key: str,
        dtype: np.dtype,
        expected: str,
        npz_path: Path,
    ) -> None:
        if expected == "uint8_or_float32":
            if dtype != np.uint8 and dtype != np.float32:
                raise ValueError(
                    f"{key} のdtypeが不正です: 期待=uint8またはfloat32, 実際={dtype} ({npz_path})"
                )
            return
        if expected == "float32" and dtype != np.float32:
            raise ValueError(f"{key} のdtypeが不正です: 期待=float32, 実際={dtype} ({npz_path})")

    def _check_dim(
        self,
        key: str,
        arr: np.ndarray,
        axis: int,
        current: Optional[int],
        name: str,
        npz_path: Path,
    ) -> int:
        if arr.ndim <= axis:
            raise ValueError(f"{key} のshapeが不正です: {arr.shape} ({npz_path})")
        size = arr.shape[axis]
        if current is None:
            return size
        if size != current:
            raise ValueError(f"{name}の次元が一致しません: {key}={size} != {current} ({npz_path})")
        return current

    def _slice_episode(
        self,
        data: np.lib.npyio.NpzFile,
        spec: EpisodeSpec,
        npz_path: Path,
    ) -> Dict[str, np.ndarray]:
        needed = self.sequence_length * self.frame_skip
        if spec.t < needed:
            raise ValueError(f"Tが不足しています: 必要={needed}, 実際={spec.t} ({npz_path})")
        indices = np.arange(0, needed, self.frame_skip)

        episode: Dict[str, np.ndarray] = {}

        # required time series
        for key in ["rgb", "q", "dq", "f", "action", "block_pose"]:
            episode[key] = data[key][indices]

        # optional time series (if exists)
        for key in OPTIONAL_TIME_KEYS:
            if key in data:
                arr = data[key]
                if arr.ndim >= 1 and arr.shape[0] == spec.t:
                    episode[key] = arr[indices]

        # scalars
        for key in ["mass", "friction"]:
            arr = data[key]
            if arr.shape != ():
                raise ValueError(f"{key} はスカラーである必要があります: {arr.shape} ({npz_path})")
            episode[key] = arr

        return episode


# 仕様名との互換性のためのエイリアス
NPZEpisodeDataset = EpisodeNPZDataset
