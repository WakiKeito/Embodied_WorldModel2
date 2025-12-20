"""固定長エピソードの単純なcollate関数。"""

from __future__ import annotations

from typing import Dict, List

import torch


def collate_fixed_length(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """同一shapeのテンソルをキーごとにstackする。"""
    if not batch:
        raise ValueError("空のバッチはcollateできません。")

    output: Dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        output[key] = torch.stack([item[key] for item in batch], dim=0)
    return output
