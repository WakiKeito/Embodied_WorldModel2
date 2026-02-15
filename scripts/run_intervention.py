#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/run_intervention.py
from __future__ import annotations

import argparse
import csv
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from wm.data.dataset_npz import EpisodeNPZDataset
from wm.data.collate import collate_fixed_length
from wm.models.world_model import WorldModelVP, WorldModelVPF
from wm.utils.rollout import rollout
from wm.utils.intervention import load_probe_directions, apply_intervention


# -----------------------------
# metrics
# -----------------------------
def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).sqrt().item()


def rmse_masked(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    diff2 = (a - b) ** 2

    if mask.ndim == 1:
        mask_bh = mask.view(1, -1)
    else:
        mask_bh = mask

    while mask_bh.ndim < diff2.ndim:
        mask_bh = mask_bh.unsqueeze(-1)

    m = mask_bh.to(diff2.device).to(diff2.dtype)

    denom = m.sum().item()
    if denom <= 0:
        return float("nan")

    mse = (diff2 * m).sum() / m.sum()
    return mse.sqrt().item()


def alpha_to_token(a: float, ndigits: int = 2) -> str:
    s = f"{a:.{ndigits}f}"
    return s.replace("-", "m").replace(".", "p")


# -----------------------------
# normalization helpers (NEW)
# -----------------------------
def _load_norm_cfg_from_json(path: Path) -> Optional[dict]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] failed to read norm_cfg.json: {path} ({e})")
    return None


def resolve_norm_cfg(ckpt_path: Path, ckpt_obj: Any, norm_cfg_arg: Optional[Path]) -> Tuple[Optional[dict], str]:
    """
    Priority:
      1) --norm-cfg (explicit)
      2) ckpt["norm_cfg"] (embedded)
      3) ckpt_dir / "norm_cfg.json"
    """
    # 1) explicit
    if norm_cfg_arg is not None:
        norm_cfg_arg = Path(norm_cfg_arg)
        nc = _load_norm_cfg_from_json(norm_cfg_arg)
        if nc is not None:
            return nc, f"args:{norm_cfg_arg}"
        raise FileNotFoundError(f"--norm-cfg specified but cannot load: {norm_cfg_arg}")

    # 2) embedded in ckpt
    if isinstance(ckpt_obj, dict) and ("norm_cfg" in ckpt_obj) and (ckpt_obj["norm_cfg"] is not None):
        nc = ckpt_obj["norm_cfg"]
        # nc might already be dict
        if isinstance(nc, dict):
            return nc, "ckpt:embedded"
        # or JSON string
        if isinstance(nc, str):
            try:
                return json.loads(nc), "ckpt:embedded_str"
            except Exception:
                pass
        print("[WARN] ckpt has norm_cfg but unrecognized type; ignore")

    # 3) ckpt_dir/norm_cfg.json
    cand = ckpt_path.parent / "norm_cfg.json"
    nc = _load_norm_cfg_from_json(cand)
    if nc is not None:
        return nc, f"file:{cand}"

    return None, "none"


# -----------------------------
# helpers
# -----------------------------
def _infer_force_embed_dim(model: torch.nn.Module, force_dim: int, device: torch.device) -> int:
    """
    force_encoder の出力次元を実測で推定する（確実に Df が得られる）。
    force_dim が 0 の場合でも、model.force_dim があればそれを使って推定する。
    """
    if not hasattr(model, "force_encoder"):
        return 0
    enc = getattr(model, "force_encoder")
    if not callable(enc):
        return 0

    fd = int(force_dim)
    if fd <= 0 and hasattr(model, "force_dim"):
        try:
            fd = int(getattr(model, "force_dim"))
        except Exception:
            fd = 0
    if fd <= 0:
        return 0

    z = torch.zeros(1, 1, fd, device=device)
    y = enc(z)  # (1,1,Df)
    return int(y.shape[-1])


@torch.no_grad()
def build_initial_hidden(
    model,
    batch: dict,
    use_force_model: bool,
    force_mode: str = "zero",        # "zero" or "gt"
    force_input_mode: str = "auto",  # "auto" | "on" | "off"
    gt_force_scale: float = 1.0,
) -> torch.Tensor:
    """
    rollout と同じルールで t=0 の embedding を作って h0 を生成する。

    重要: ここは「モデルの forward に合わせて次元整合が絶対に崩れない」実装にする。
    - VP:  img+prop
    - VPF: img+prop+force_emb（force_input_mode=off のときは force_emb をゼロ埋め）
    """
    device = next(model.parameters()).device

    rgb = batch["rgb"]
    q = batch["q"]
    dq = batch["dq"]
    block_pose = batch["block_pose"]

    rgb0 = rgb[:, :1]
    q0 = q[:, :1]
    dq0 = dq[:, :1]
    bp0 = block_pose[:, :1]

    img_emb0 = model.image_encoder(rgb0)            # (B,1,Di)
    prop_emb0 = model.proprio_encoder(q0, dq0, bp0) # (B,1,Dp)

    has_force_encoder = hasattr(model, "force_encoder") and callable(getattr(model, "force_encoder"))
    has_f = ("f" in batch)

    if force_input_mode == "on":
        use_force_input = bool(use_force_model and has_force_encoder and has_f)
    elif force_input_mode == "off":
        use_force_input = False
    elif force_input_mode == "auto":
        use_force_input = bool(use_force_model and has_force_encoder and has_f)
    else:
        raise ValueError(f"bad force_input_mode: {force_input_mode}")

    # ---- VP or "no-force-available" fallback ----
    if (not use_force_model) or (not has_force_encoder):
        emb0 = torch.cat([img_emb0[:, 0], prop_emb0[:, 0]], dim=-1)
        B = emb0.shape[0]
        a0 = torch.zeros(B, int(model.action_dim), device=emb0.device)
        x0 = torch.cat([emb0, a0], dim=-1)
        h0 = model.dynamics.forward_step(x0, None)
        return h0

    # ---- VPF ----
    Fdim = int(batch["f"].shape[-1]) if has_f else 0
    Df = _infer_force_embed_dim(model, Fdim, device=device)

    if use_force_input:
        if force_mode == "gt":
            f0 = batch["f"][:, :1] * float(gt_force_scale)
        elif force_mode == "zero":
            f0 = torch.zeros(batch["q"].shape[0], 1, Fdim, device=device)
        else:
            raise ValueError(f"Unknown force_mode: {force_mode} (expected 'zero' or 'gt')")
        force_emb0 = model.force_encoder(f0)  # (B,1,Df)
    else:
        # force_input_mode=off: 次元合わせのゼロ埋め
        force_emb0 = torch.zeros(batch["q"].shape[0], 1, Df, device=device)

    emb0 = torch.cat([img_emb0[:, 0], prop_emb0[:, 0], force_emb0[:, 0]], dim=-1)

    B = emb0.shape[0]
    a0 = torch.zeros(B, int(model.action_dim), device=emb0.device)
    x0 = torch.cat([emb0, a0], dim=-1)
    h0 = model.dynamics.forward_step(x0, None)
    return h0


def _extract_air_flag_from_batch(batch: dict, H: int, device: torch.device) -> torch.Tensor:
    if "is_air_force" in batch:
        air = batch["is_air_force"][0, 1:1 + H].to(device)
        return (air.to(torch.int32) == 1)
    if "phase_id" in batch:
        ph = batch["phase_id"][0, 1:1 + H].to(device)
        return (ph.to(torch.int32) == 3)
    return torch.zeros(H, device=device, dtype=torch.bool)


# -----------------------------
# force ablation utilities
# -----------------------------
def _force_apply_norm_only(f: torch.Tensor) -> torch.Tensor:
    # f: (B,T,F)
    fn = torch.linalg.norm(f, dim=-1, keepdim=True)  # (B,T,1)
    out = torch.zeros_like(f)
    out[..., 0:1] = fn
    return out


def _force_shuffle_time(
    f: torch.Tensor,
    start_t: int = 0,
    end_t: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Shuffle time indices in [start_t, end_t) per batch item.
    - f: (B,T,F)
    """
    B, T, _ = f.shape
    out = f.clone()

    s = int(max(0, min(start_t, T)))
    e = T if end_t is None else int(max(0, min(end_t, T)))
    if e <= s + 1:
        return out

    gen = None
    if seed is not None:
        gen = torch.Generator(device=f.device)
        gen.manual_seed(int(seed))

    for b in range(B):
        head = torch.arange(0, s, device=f.device)
        mid = torch.arange(s, e, device=f.device)
        tail = torch.arange(e, T, device=f.device)

        perm = mid[torch.randperm(mid.numel(), generator=gen, device=f.device)]
        idx = torch.cat([head, perm, tail], dim=0)
        out[b] = f[b, idx, :]
    return out


def _force_zero_slice(f: torch.Tensor, start_t: int, end_t: Optional[int] = None) -> torch.Tensor:
    out = f.clone()
    T = out.shape[1]
    s = int(max(0, min(start_t, T)))
    e = T if end_t is None else int(max(0, min(end_t, T)))
    if e > s:
        out[:, s:e, :] = 0.0
    return out


def _apply_force_ablation(
    f: torch.Tensor,
    release_start: int,
    mode_norm_only: bool,
    shuffle_mode: str,
    zero_mode: str,
    shuffle_seed: Optional[int],
) -> torch.Tensor:
    """
    Apply ablations to f (B,T,F) BEFORE rollout.

    Important:
      - rollout() uses f_gt[:, t] with t=0..H-1.
      - release_start is in the same time index space.
    """
    rs = int(release_start)
    rs = max(0, min(rs, f.shape[1]))

    if mode_norm_only:
        f = _force_apply_norm_only(f)

    if shuffle_mode == "all":
        f = _force_shuffle_time(f, start_t=0, end_t=None, seed=shuffle_seed)
    elif shuffle_mode == "contact":
        f = _force_shuffle_time(f, start_t=0, end_t=rs, seed=shuffle_seed)
    elif shuffle_mode in ("coast", "after_release"):
        f = _force_shuffle_time(f, start_t=rs, end_t=None, seed=shuffle_seed)
    elif shuffle_mode == "none":
        pass
    else:
        raise ValueError(f"bad force_shuffle: {shuffle_mode}")

    if zero_mode == "all":
        f = _force_zero_slice(f, 0, None)
    elif zero_mode == "contact":
        f = _force_zero_slice(f, 0, rs)
    elif zero_mode in ("coast", "after_release"):
        f = _force_zero_slice(f, rs, None)
    elif zero_mode == "none":
        pass
    else:
        raise ValueError(f"bad force_zero: {zero_mode}")

    return f


# -----------------------------
# CSV saving
# -----------------------------
def save_traj_csv(path: str, meta: dict, base: dict, inter: dict, batch: dict | None = None):
    bp_b = base["block_pose_hat"][0].detach().cpu().numpy()
    bp_i = inter["block_pose_hat"][0].detach().cpu().numpy()
    H = min(len(bp_b), len(bp_i))
    bp_b = bp_b[:H]
    bp_i = bp_i[:H]

    data = {
        "t": np.arange(H),
        "block_x_base": bp_b[:, 0],
        "block_y_base": bp_b[:, 1],
        "block_x_int":  bp_i[:, 0],
        "block_y_int":  bp_i[:, 1],
    }

    if batch is not None and ("block_pose" in batch):
        bp_gt = batch["block_pose"][0, 1:1 + H].detach().cpu().numpy()
        data["block_x_gt"] = bp_gt[:, 0]
        data["block_y_gt"] = bp_gt[:, 1]

    if batch is not None and ("f" in batch):
        f = batch["f"][0, 1:1 + H].detach().cpu().numpy()
        if f.ndim == 1:
            f = f.reshape(H, 1)

        if f.shape[1] >= 3:
            fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
            fn = np.sqrt(fx**2 + fy**2 + fz**2)
            data["force_x_gt"] = fx
            data["force_y_gt"] = fy
            data["force_z_gt"] = fz
            data["force_norm_gt"] = fn
        else:
            data["force_norm_gt"] = np.linalg.norm(f, axis=1)

    for pref, preds in [("base", base), ("int", inter)]:
        if "force_hat" in preds and torch.is_tensor(preds["force_hat"]):
            fh = preds["force_hat"][0].detach().cpu().numpy()[:H]
            if fh.ndim == 2 and fh.shape[1] >= 3:
                data[f"force_hat_x_{pref}"] = fh[:, 0]
                data[f"force_hat_y_{pref}"] = fh[:, 1]
                data[f"force_hat_z_{pref}"] = fh[:, 2]

        if "force_hat_scaled" in preds and torch.is_tensor(preds["force_hat_scaled"]):
            fhs = preds["force_hat_scaled"][0].detach().cpu().numpy()[:H]
            if fhs.ndim == 2 and fhs.shape[1] >= 3:
                data[f"force_hat_scaled_x_{pref}"] = fhs[:, 0]
                data[f"force_hat_scaled_y_{pref}"] = fhs[:, 1]
                data[f"force_hat_scaled_z_{pref}"] = fhs[:, 2]

    if "release_start" in meta:
        rs = int(meta["release_start"])
        rs = max(0, min(rs, H))
        data["is_contact_rs"] = (np.arange(H) < rs).astype(int)

    if "force_norm_gt" in data:
        th = float(meta.get("contact_th", 0.10))
        data["is_contact_force"] = (np.asarray(data["force_norm_gt"]) > th).astype(int)

    if "is_contact_rs" in data:
        data["is_contact"] = data["is_contact_rs"]
    elif "is_contact_force" in data:
        data["is_contact"] = data["is_contact_force"]

    if batch is not None:
        if "is_air_force" in batch:
            air = batch["is_air_force"][0, 1:1 + H].detach().cpu().numpy().astype(int)
            data["is_air_force"] = air
        if "phase_id" in batch:
            ph = batch["phase_id"][0, 1:1 + H].detach().cpu().numpy().astype(int)
            data["phase_id"] = ph

    for k, v in meta.items():
        data[k] = [v] * H

    def _add_dbg(pref: str, preds: dict):
        dbg = preds.get("debug", None)
        if not isinstance(dbg, dict):
            return
        for key in ["force_gt_norm", "force_used_norm", "force_hat_norm", "h_norm", "dh_norm"]:
            if key in dbg and torch.is_tensor(dbg[key]):
                data[f"{key}_{pref}"] = dbg[key][0].detach().cpu().numpy()[:H]

    _add_dbg("base", base)
    _add_dbg("int", inter)

    df_out = pd.DataFrame(data)
    tmp = str(path) + ".tmp"
    df_out.to_csv(tmp, index=False)
    Path(tmp).replace(path)
    print(f"[OK] saved traj: {path}")


# -----------------------------
# args
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    # (CHANGED default) raw -> train
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/train"))
    ap.add_argument("--episode-idx", type=int, default=0)
    ap.add_argument("--episode-path", type=Path, default=None, help="npzファイルを直接指定（episode-idxより優先）")

    ap.add_argument("--horizon", type=int, default=80)

    ap.add_argument("--use-force", action="store_true", help="VPF系（force関連を有効化）")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)

    # NEW: normalization config
    ap.add_argument("--norm-cfg", type=Path, default=None,
                    help="Optional norm_cfg.json path. If omitted, use ckpt['norm_cfg'] or ckpt_dir/norm_cfg.json.")

    ap.add_argument("--axis", type=str, default="mass", choices=["mass", "friction"])
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--normalize", action="store_true")

    ap.add_argument("--h0-force", type=str, default="zero", choices=["zero", "gt"])

    ap.add_argument("--rollout-gt-force", action="store_true")
    ap.add_argument("--gt-force-scale", type=float, default=1.0)

    ap.add_argument("--gt-force-pre-release", action="store_true")
    ap.add_argument("--gt-force-decay-lambda", type=float, default=1.0)

    ap.add_argument("--force-input-mode", type=str, default="auto",
                    choices=["auto", "on", "off"])

    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--tag", type=str, default="")

    ap.add_argument("--release-start", type=int, default=40)
    ap.add_argument("--prefer-args-release-start", action="store_true")

    ap.add_argument("--intervene-mode", type=str, default="h0",
                    choices=["h0", "all", "contact", "coast"])

    ap.add_argument("--force-cut-after-release", action="store_true")
    ap.add_argument("--force-decay-lambda", type=float, default=1.0)
    ap.add_argument("--force-hat-scale", type=float, default=1.0)
    ap.add_argument("--force-feedback", type=str, default="hat", choices=["hat", "zero"])
    ap.add_argument("--force-hat-sign-x", type=float, default=1.0)
    ap.add_argument("--force-hat-sign-y", type=float, default=1.0)
    ap.add_argument("--force-hat-sign-z", type=float, default=1.0)

    ap.add_argument("--contact-th", type=float, default=0.10)

    ap.add_argument("--force-norm-only", action="store_true",
                    help="Replace force vector with norm on x-axis (direction ablation).")

    ap.add_argument("--force-shuffle", type=str, default="none",
                    choices=["none", "all", "contact", "coast", "after_release"],
                    help="Shuffle force sequence along time dimension. 'after_release' is an alias of 'coast'.")

    ap.add_argument("--force-zero", type=str, default="none",
                    choices=["none", "all", "contact", "coast", "after_release"],
                    help="Zero force on a slice. 'after_release' is an alias of 'coast'.")

    ap.add_argument("--force-shuffle-seed", type=int, default=None,
                    help="Optional seed for force shuffling (for reproducibility).")

    ap.add_argument("--force-zero-after-release", action="store_true",
                    help="[DEPRECATED] Same as --force-zero after_release.")

    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


# -----------------------------
# main
# -----------------------------
def main():
    args = parse_args()

    # choose episode
    if args.episode_path is None:
        npz_files = sorted(args.dataset_root.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No npz in {args.dataset_root}")
        if args.episode_idx < 0 or args.episode_idx >= len(npz_files):
            raise IndexError(f"episode_idx out of range: {args.episode_idx} (0..{len(npz_files)-1})")
        ep_path = npz_files[args.episode_idx]
    else:
        ep_path = args.episode_path

    # read npz meta
    release_start_npz = None
    release_steps_npz = None
    with np.load(ep_path) as d:
        T_npz = int(d["q"].shape[0])
        if "release_steps" in d.files:
            release_steps_npz = int(np.asarray(d["release_steps"]).item())
        if release_steps_npz is not None:
            release_steps_npz = max(0, min(release_steps_npz, T_npz))
            release_start_npz = T_npz - release_steps_npz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load ckpt first (NEW: for norm_cfg resolution)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    norm_cfg, norm_src = resolve_norm_cfg(args.ckpt, ckpt, args.norm_cfg)
    if norm_cfg is not None:
        print(f"[INFO] norm_cfg resolved: {norm_src}")
    else:
        print(f"[WARN] norm_cfg not found (source={norm_src}). Dataset will be unnormalized. This may break comparability.")

    # dataset -> batch (B=1)
    seq_len = min(T_npz, 256)
    cfg = {"sequence_length": seq_len, "frame_skip": 1, "keys": {}}
    # NEW: pass normalization into dataset
    if norm_cfg is not None:
        cfg["normalization"] = norm_cfg

    ds = EpisodeNPZDataset([ep_path], config=cfg)
    batch = collate_fixed_length([ds[0]])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    j_dim = int(batch["q"].shape[-1])
    action_dim = int(batch["action"].shape[-1])

    if args.use_force:
        if "f" not in batch:
            raise KeyError("--use-force specified but this episode has no 'f'.")
        force_dim = int(batch["f"].shape[-1])
        model = WorldModelVPF(j_dim=j_dim, action_dim=action_dim, force_dim=force_dim).to(device)
        model_name = "VPF"
        print(f"[INFO] Using VPF (force_dim={force_dim})")
    else:
        model = WorldModelVP(j_dim=j_dim, action_dim=action_dim).to(device)
        model_name = "VP"
        print("[INFO] Using VP")

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    dirs = load_probe_directions(args.probe, device=device)
    direction = dirs.w_mass if args.axis == "mass" else dirs.w_friction

    # horizon H
    T = int(batch["q"].shape[1])
    H = min(int(args.horizon), T - 1)
    if H <= 0:
        raise ValueError(f"H must be >0, got H={H}, T={T}")

    # release_start
    if args.prefer_args_release_start or (release_start_npz is None):
        release_start = int(args.release_start)
        rs_src = "args"
    else:
        release_start = int(release_start_npz)
        rs_src = "npz"

    release_start = max(0, min(int(release_start), H))

    if release_start_npz is not None:
        print(
            f"[INFO] release_start in npz = {release_start_npz} "
            f"(release_steps={release_steps_npz}, T_npz={T_npz}) "
            f"(prefer_args={int(args.prefer_args_release_start)})"
        )
    print(f"[INFO] release_start used = {release_start} (source={rs_src})")
    print(f"[INFO] contact_th used = {float(args.contact_th)}")

    # -----------------------------
    # force ablation (batch["f"] pre-processing)
    # -----------------------------
    if args.force_zero_after_release:
        if args.force_zero == "none":
            args.force_zero = "after_release"

    if args.use_force:
        f = batch["f"]  # (B,T,F)
        f = _apply_force_ablation(
            f=f,
            release_start=release_start,
            mode_norm_only=bool(args.force_norm_only),
            shuffle_mode=str(args.force_shuffle),
            zero_mode=str(args.force_zero),
            shuffle_seed=args.force_shuffle_seed,
        )
        batch["f"] = f

    q_gt = batch["q"][:, 1:1 + H]
    block_gt = batch["block_pose"][:, 1:1 + H]

    def crop_preds(preds: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in list(preds.items()):
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] >= H:
                preds[k] = v[:, :H]
        return preds

    use_gt_force = bool(args.rollout_gt_force)

    # build h0 (safe)
    h0 = build_initial_hidden(
        model, batch,
        use_force_model=bool(args.use_force),
        force_mode=str(args.h0_force),
        force_input_mode=str(args.force_input_mode),
        gt_force_scale=float(args.gt_force_scale),
    )

    intervene_mode = str(args.intervene_mode)

    if intervene_mode == "h0":
        # --- normalize direction here (compat) ---
        direction_use = direction
        if bool(args.normalize):
            direction_use = direction_use / (direction_use.norm() + 1e-12)

        h0_i = apply_intervention(h0, direction_use, alpha=float(args.alpha))

        dh0 = (h0_i - h0).norm().item()
        print(f"[DBG] mode=h0 ||h0_i - h0||={dh0:.6e} alpha={args.alpha} normalize={args.normalize}")

        rollout_intervene_direction = None
        rollout_intervene_alpha = 0.0
        rollout_intervene_normalize = False
        rollout_intervene_mode = "none"
    else:
        h0_i = h0
        rollout_intervene_direction = direction
        rollout_intervene_alpha = float(args.alpha)
        rollout_intervene_normalize = bool(args.normalize)
        rollout_intervene_mode = intervene_mode
        print(f"[DBG] mode={intervene_mode} stepwise_alpha={rollout_intervene_alpha} "
              f"normalize={rollout_intervene_normalize} release_start={release_start}")

    base_h_for_rollout = h0

    rollout_force_kwargs = dict(
        release_start=release_start,
        force_input_mode=str(args.force_input_mode),

        use_gt_force=use_gt_force,
        gt_force_scale=float(args.gt_force_scale),
        gt_force_pre_release=bool(args.gt_force_pre_release),
        gt_force_decay_lambda=float(args.gt_force_decay_lambda),

        force_cut_after_release=bool(args.force_cut_after_release),
        force_decay_lambda=float(args.force_decay_lambda),
        force_hat_scale=float(args.force_hat_scale),
        force_feedback=str(args.force_feedback),
        force_hat_sign_x=float(args.force_hat_sign_x),
        force_hat_sign_y=float(args.force_hat_sign_y),
        force_hat_sign_z=float(args.force_hat_sign_z),

        return_debug=True,
    )

    # OPEN (vision t0)
    preds_open_base = crop_preds(rollout(
        model, batch, horizon=H, h0=base_h_for_rollout,
        vision_mode="t0",
        intervene_direction=None, intervene_alpha=0.0, intervene_normalize=False, intervene_mode="none",
        **rollout_force_kwargs,
    ))
    preds_open_int = crop_preds(rollout(
        model, batch, horizon=H, h0=h0_i,
        vision_mode="t0",
        intervene_direction=rollout_intervene_direction,
        intervene_alpha=rollout_intervene_alpha,
        intervene_normalize=rollout_intervene_normalize,
        intervene_mode=rollout_intervene_mode,
        **rollout_force_kwargs,
    ))

    # FULL (vision gt)
    preds_full_base = crop_preds(rollout(
        model, batch, horizon=H, h0=base_h_for_rollout,
        vision_mode="gt",
        intervene_direction=None, intervene_alpha=0.0, intervene_normalize=False, intervene_mode="none",
        **rollout_force_kwargs,
    ))
    preds_full_int = crop_preds(rollout(
        model, batch, horizon=H, h0=h0_i,
        vision_mode="gt",
        intervene_direction=rollout_intervene_direction,
        intervene_alpha=rollout_intervene_alpha,
        intervene_normalize=rollout_intervene_normalize,
        intervene_mode=rollout_intervene_mode,
        **rollout_force_kwargs,
    ))

    # RMSE
    rmse_q_open_base = rmse(preds_open_base["q_hat"], q_gt)
    rmse_block_open_base = rmse(preds_open_base["block_pose_hat"], block_gt)
    rmse_q_open_int = rmse(preds_open_int["q_hat"], q_gt)
    rmse_block_open_int = rmse(preds_open_int["block_pose_hat"], block_gt)

    rmse_q_full_base = rmse(preds_full_base["q_hat"], q_gt)
    rmse_block_full_base = rmse(preds_full_base["block_pose_hat"], block_gt)
    rmse_q_full_int = rmse(preds_full_int["q_hat"], q_gt)
    rmse_block_full_int = rmse(preds_full_int["block_pose_hat"], block_gt)

    # masks
    idx = torch.arange(H, device=device)
    mask_contact = (idx < release_start)
    mask_coast = ~mask_contact

    mask_air = _extract_air_flag_from_batch(batch, H=H, device=device)
    mask_contact_wo_air = mask_contact & (~mask_air)
    mask_coast_wo_air = mask_coast & (~mask_air)

    rmse_block_full_base_air = rmse_masked(preds_full_base["block_pose_hat"], block_gt, mask_air)
    rmse_block_full_int_air = rmse_masked(preds_full_int["block_pose_hat"], block_gt, mask_air)

    rmse_block_full_base_contact = rmse_masked(preds_full_base["block_pose_hat"], block_gt, mask_contact_wo_air)
    rmse_block_full_base_coast = rmse_masked(preds_full_base["block_pose_hat"], block_gt, mask_coast_wo_air)

    rmse_block_full_int_contact = rmse_masked(preds_full_int["block_pose_hat"], block_gt, mask_contact_wo_air)
    rmse_block_full_int_coast = rmse_masked(preds_full_int["block_pose_hat"], block_gt, mask_coast_wo_air)

    # outdir
    args.outdir.mkdir(parents=True, exist_ok=True)

    tag = args.tag
    tag = (tag + "_") if (tag and not tag.endswith("_")) else tag

    ep_id = args.episode_idx
    if args.episode_path is not None:
        m = re.search(r"episode_(\d+)", ep_path.name)
        if m:
            ep_id = int(m.group(1))

    alpha_str = alpha_to_token(args.alpha)
    base = f"{tag}int_{model_name.lower()}_ep{ep_id}_{args.axis}_a{alpha_str}"

    out_metrics = args.outdir / f"{base}_metrics.csv"
    out_open = args.outdir / f"{base}_traj_open.csv"
    out_full = args.outdir / f"{base}_traj_full.csv"

    for p in [out_metrics, out_open, out_full]:
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"File exists. Use --overwrite: {p}")

    with out_metrics.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "model", "axis", "alpha", "normalize",
            "mass", "friction",
            "force_input_mode",
            "h0_force", "rollout_gt_force",
            "gt_force_scale", "gt_force_pre_release", "gt_force_decay_lambda",
            "release_start", "release_start_source", "horizon",
            "force_cut_after_release", "force_decay_lambda",
            "force_hat_scale", "force_feedback",
            "force_hat_sign_x", "force_hat_sign_y", "force_hat_sign_z",
            "contact_th",
            # NEW: norm
            "norm_cfg_source",
            # force ablations
            "force_norm_only", "force_shuffle", "force_zero", "force_shuffle_seed",
            "rmse_q_open_base", "rmse_block_open_base", "rmse_q_open_int", "rmse_block_open_int",
            "rmse_q_full_base", "rmse_block_full_base", "rmse_q_full_int", "rmse_block_full_int",
            "rmse_block_full_base_air", "rmse_block_full_int_air",
            "rmse_block_full_base_contact", "rmse_block_full_base_coast",
            "rmse_block_full_int_contact", "rmse_block_full_int_coast",
            "n_air", "n_contact_wo_air", "n_coast_wo_air",
        ])
        w.writerow([
            ep_path.name, model_name, args.axis, float(args.alpha), int(args.normalize),
            float(batch["mass"].item()), float(batch["friction"].item()),
            str(args.force_input_mode),
            str(args.h0_force), int(use_gt_force),
            float(args.gt_force_scale), int(args.gt_force_pre_release), float(args.gt_force_decay_lambda),
            int(release_start), rs_src, int(H),
            int(args.force_cut_after_release), float(args.force_decay_lambda),
            float(args.force_hat_scale), str(args.force_feedback),
            float(args.force_hat_sign_x), float(args.force_hat_sign_y), float(args.force_hat_sign_z),
            float(args.contact_th),
            str(norm_src),
            int(bool(args.force_norm_only)), str(args.force_shuffle), str(args.force_zero),
            ("" if args.force_shuffle_seed is None else int(args.force_shuffle_seed)),
            rmse_q_open_base, rmse_block_open_base, rmse_q_open_int, rmse_block_open_int,
            rmse_q_full_base, rmse_block_full_base, rmse_q_full_int, rmse_block_full_int,
            rmse_block_full_base_air, rmse_block_full_int_air,
            rmse_block_full_base_contact, rmse_block_full_base_coast,
            rmse_block_full_int_contact, rmse_block_full_int_coast,
            int(mask_air.sum().item()),
            int(mask_contact_wo_air.sum().item()),
            int(mask_coast_wo_air.sum().item()),
        ])

    print(f"[OK] saved metrics: {out_metrics}")

    meta_common = dict(
        episode=ep_path.name,
        model=model_name,
        axis=args.axis,
        alpha=float(args.alpha),
        normalize=int(args.normalize),
        mass=float(batch["mass"].item()),
        friction=float(batch["friction"].item()),
        force_input_mode=str(args.force_input_mode),
        h0_force=str(args.h0_force),
        rollout_gt_force=int(use_gt_force),
        gt_force_scale=float(args.gt_force_scale),
        gt_force_pre_release=int(args.gt_force_pre_release),
        gt_force_decay_lambda=float(args.gt_force_decay_lambda),
        release_start=int(release_start),
        release_start_source=rs_src,
        intervene_mode=str(args.intervene_mode),
        stepwise_alpha=float(args.alpha) if str(args.intervene_mode) != "h0" else 0.0,
        force_cut_after_release=int(args.force_cut_after_release),
        force_decay_lambda=float(args.force_decay_lambda),
        force_hat_scale=float(args.force_hat_scale),
        force_feedback=str(args.force_feedback),
        force_hat_sign_x=float(args.force_hat_sign_x),
        force_hat_sign_y=float(args.force_hat_sign_y),
        force_hat_sign_z=float(args.force_hat_sign_z),
        contact_th=float(args.contact_th),
        force_norm_only=int(bool(args.force_norm_only)),
        force_shuffle=str(args.force_shuffle),
        force_zero=str(args.force_zero),
        force_shuffle_seed=(None if args.force_shuffle_seed is None else int(args.force_shuffle_seed)),
        # NEW
        norm_cfg_source=str(norm_src),
    )

    meta_open = dict(meta_common); meta_open["mode"] = "open"
    meta_full = dict(meta_common); meta_full["mode"] = "full"

    save_traj_csv(str(out_open), meta_open, preds_open_base, preds_open_int, batch=batch)
    save_traj_csv(str(out_full), meta_full, preds_full_base, preds_full_int, batch=batch)

    print(f"[OK] saved outputs under: {args.outdir}")

    if args.use_force:
        print(
            f"[INFO] rollout force settings: "
            f"force_input_mode={args.force_input_mode} "
            f"use_gt_force={int(use_gt_force)} "
            f"gt_force_scale={args.gt_force_scale} "
            f"gt_pre_release={int(args.gt_force_pre_release)} "
            f"gt_decay_lambda={args.gt_force_decay_lambda} "
            f"release_start={release_start} "
            f"force_feedback={args.force_feedback} "
            f"force_hat_scale={args.force_hat_scale} "
            f"cut_after_release={int(args.force_cut_after_release)} "
            f"decay_lambda={args.force_decay_lambda} "
            f"force_norm_only={int(bool(args.force_norm_only))} "
            f"force_shuffle={args.force_shuffle} "
            f"force_zero={args.force_zero} "
            f"force_shuffle_seed={args.force_shuffle_seed}"
        )

    print(f"[OPEN BASE] rmse_q={rmse_q_open_base:.6f} rmse_block={rmse_block_open_base:.6f}")
    print(f"[OPEN INT ] rmse_q={rmse_q_open_int:.6f} rmse_block={rmse_block_open_int:.6f}")
    print(f"[FULL BASE] rmse_q={rmse_q_full_base:.6f} rmse_block={rmse_block_full_base:.6f}")
    print(f"[FULL INT ] rmse_q={rmse_q_full_int:.6f} rmse_block={rmse_block_full_int:.6f}")
    print(f"[AIR FULL] block base air={rmse_block_full_base_air:.6f} int air={rmse_block_full_int_air:.6f} n_air={int(mask_air.sum().item())}")
    print(f"[SEG FULL] block base contact={rmse_block_full_base_contact:.6f} coast={rmse_block_full_base_coast:.6f}")
    print(f"[SEG FULL] block int  contact={rmse_block_full_int_contact:.6f}  coast={rmse_block_full_int_coast:.6f}")


if __name__ == "__main__":
    main()