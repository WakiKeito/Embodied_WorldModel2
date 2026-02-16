#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_probe.py (fast, accuracy-preserving; meta-safe + cache-safe)

Fixes:
1) Save probe_meta into probe checkpoint (so eval_probe can interpret targets consistently)
2) Add cache/probe signature to detect stale/mismatched latent caches
3) Store enough provenance into latent cache (dataset_root/ckpt/norm_cfg/use_force/etc.)
4) Make --pin-memory a real toggle
5) Fix a fatal typo: args.batch-size -> args.batch_size
6) Allow disabling persistent workers (previous version forced True)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset

from wm.data.dataset_npz import EpisodeNPZDataset
from wm.data.collate import collate_fixed_length
from wm.models.world_model import WorldModelVP, WorldModelVPF


# -------------------------
# norm cfg helpers
# -------------------------
def _load_json(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))


def load_norm_cfg(norm_cfg_path: Optional[str], ckpt_path: str) -> Optional[Dict]:
    ckpt_p = Path(ckpt_path)

    if norm_cfg_path:
        p = Path(norm_cfg_path)
        if not p.exists():
            raise FileNotFoundError(f"--norm-cfg not found: {p}")
        return _load_json(p)

    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "norm_cfg" in ck and isinstance(ck["norm_cfg"], dict):
        return ck["norm_cfg"]

    cand = ckpt_p.parent / "norm_cfg.json"
    if cand.exists():
        return _load_json(cand)

    return None


def build_dataset_cfg(seq_len: int, frame_skip: int, norm_cfg: Optional[Dict]) -> Dict:
    cfg = {
        "sequence_length": int(seq_len),
        "frame_skip": int(frame_skip),
        "keys": {},
    }
    if norm_cfg is not None:
        cfg["normalization"] = norm_cfg
    return cfg


# -------------------------
# provenance / signatures
# -------------------------
def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def make_signature(
    dataset_root: str,
    ckpt: str,
    use_force: bool,
    rep_mode: str,
    seq_len: int,
    frame_skip: int,
    norm_cfg: Optional[Dict],
) -> str:
    payload = {
        "dataset_root": str(dataset_root),
        "ckpt": str(ckpt),
        "use_force": bool(use_force),
        "rep_mode": str(rep_mode),
        "seq_len": int(seq_len),
        "frame_skip": int(frame_skip),
        "norm_cfg": norm_cfg,
    }
    s = _stable_json(payload)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def default_probe_meta() -> Dict[str, Any]:
    return {
        "log_mass": False,
        "standardize_y": False,
        "has_y_mean": False,
        "has_y_std": False,
        "target_names": ["mass", "friction"],
    }


# -------------------------
# model / probe
# -------------------------
def summarize_latent(h_seq: torch.Tensor, mode: str) -> torch.Tensor:
    """
    h_seq: (B, T-1, H)
    returns: (B, H)
    """
    if mode == "mean":
        return h_seq.mean(dim=1)
    if mode == "last":
        return h_seq[:, -1]
    raise ValueError(f"unknown rep-mode: {mode}")


class LinearProbe(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.lin = nn.Linear(h_dim, 2)  # [mass, friction]

    def forward(self, x):
        return self.lin(x)


# -------------------------
# latent cache
# -------------------------
@torch.no_grad()
def build_latent_cache(
    wm: nn.Module,
    dl: DataLoader,
    device: torch.device,
    rep_mode: str,
    save_path: Path,
    *,
    dataset_root: str,
    ckpt_path: str,
    use_force: bool,
    norm_cfg: Optional[Dict],
    signature: str,
    probe_meta: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    wm.eval()

    xs = []
    ys = []

    for batch in dl:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = wm(batch)
        h = summarize_latent(out["h_seq"], mode=rep_mode)  # (B,H)
        y = torch.stack([batch["mass"].float(), batch["friction"].float()], dim=1)  # (B,2)
        xs.append(h.detach().cpu())
        ys.append(y.detach().cpu())

    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "X": X,
            "Y": Y,
            "rep_mode": rep_mode,
            "dataset_root": str(dataset_root),
            "ckpt": str(ckpt_path),
            "use_force": bool(use_force),
            "norm_cfg": norm_cfg,
            "probe_meta": probe_meta,
            "signature": str(signature),
        },
        save_path,
    )
    print(f"[OK] saved latent cache: {save_path}  (N={X.shape[0]}, Hdim={X.shape[1]})")
    return X, Y


def load_latent_cache(path: Path) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["X"], d["Y"], d.get("rep_mode", "mean"), d.get("signature", "")


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--use-force", action="store_true")

    ap.add_argument("--rep-mode", type=str, default="mean", choices=["mean", "last"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)

    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--save-probe", required=True, type=str)
    ap.add_argument("--norm-cfg", type=str, default=None)

    # speed knobs
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    ap.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")

    # allow disabling persistent workers
    ap.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=True)
    ap.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")

    ap.add_argument("--prefetch-factor", type=int, default=4)

    # latent cache
    ap.add_argument("--cache-latents", action="store_true")
    ap.add_argument("--cache-path", type=str, default="")
    ap.add_argument("--rebuild-cache", action="store_true")

    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--patience", type=int, default=30)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe_meta = default_probe_meta()

    norm_cfg = load_norm_cfg(args.norm_cfg, args.ckpt)
    if norm_cfg is None:
        print("[WARN] norm_cfg not found. Probe training may be scale-mismatched (esp. for VPF).")
    else:
        print(f"[INFO] norm_cfg loaded (enabled={norm_cfg.get('enabled', True)})")

    npz = sorted(Path(args.dataset_root).glob("*.npz"))
    if not npz:
        raise FileNotFoundError(f"no npz under: {args.dataset_root}")

    SEQ_LEN = 64
    FRAME_SKIP = 1

    cfg = build_dataset_cfg(seq_len=SEQ_LEN, frame_skip=FRAME_SKIP, norm_cfg=norm_cfg)
    ds = EpisodeNPZDataset(npz, config=cfg)

    n_val = int(round(len(ds) * float(args.val_ratio)))
    n_tr = len(ds) - n_val
    ds_tr, ds_va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(args.seed))

    num_workers = int(args.num_workers)
    persistent = bool(args.persistent_workers) and (num_workers > 0)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=persistent,
        prefetch_factor=int(args.prefetch_factor) if num_workers > 0 else 2,
        collate_fn=collate_fixed_length,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=persistent,
        prefetch_factor=int(args.prefetch_factor) if num_workers > 0 else 2,
        collate_fn=collate_fixed_length,
    )

    b0 = next(iter(dl_tr))
    j_dim = int(b0["q"].shape[-1])
    a_dim = int(b0["action"].shape[-1])

    if args.use_force:
        if "f" not in b0:
            raise KeyError("use_force=True but batch has no 'f'. Check dataset/transforms.")
        f_dim = int(b0["f"].shape[-1])
        wm = WorldModelVPF(j_dim=j_dim, action_dim=a_dim, force_dim=f_dim).to(device)
        wm_kind = "vpf"
    else:
        wm = WorldModelVP(j_dim=j_dim, action_dim=a_dim).to(device)
        wm_kind = "vp"

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    st = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    wm.load_state_dict(st, strict=True)
    wm.eval()

    out_txt = Path(args.out)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    if args.cache_path:
        cache_path = Path(args.cache_path)
    else:
        cache_path = out_txt.parent / f"latent_cache_{args.rep_mode}_{wm_kind}.pt"

    signature = make_signature(
        dataset_root=args.dataset_root,
        ckpt=args.ckpt,
        use_force=bool(args.use_force),
        rep_mode=args.rep_mode,
        seq_len=SEQ_LEN,
        frame_skip=FRAME_SKIP,
        norm_cfg=norm_cfg,
    )

    # -------------------------
    # Option A: cache latents
    # -------------------------
    if args.cache_latents:
        need_rebuild = bool(args.rebuild_cache)

        if cache_path.exists() and not args.rebuild_cache:
            X, Y, rep_in, sig_in = load_latent_cache(cache_path)
            if rep_in != args.rep_mode:
                print(f"[WARN] cache rep_mode={rep_in} but args.rep_mode={args.rep_mode}. Rebuild recommended.")
                need_rebuild = True
            if sig_in != signature:
                print("[WARN] latent cache signature mismatch. Rebuild required.")
                need_rebuild = True
            if not need_rebuild:
                print(f"[OK] loaded latent cache: {cache_path} (N={X.shape[0]}, Hdim={X.shape[1]})")

        if (not cache_path.exists()) or need_rebuild:
            dl_all = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=bool(args.pin_memory),
                persistent_workers=persistent,
                prefetch_factor=int(args.prefetch_factor) if num_workers > 0 else 2,
                collate_fn=collate_fixed_length,
            )
            X, Y = build_latent_cache(
                wm,
                dl_all,
                device=device,
                rep_mode=args.rep_mode,
                save_path=cache_path,
                dataset_root=args.dataset_root,
                ckpt_path=args.ckpt,
                use_force=bool(args.use_force),
                norm_cfg=norm_cfg,
                signature=signature,
                probe_meta=probe_meta,
            )

        N = X.shape[0]
        idx = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
        n_val2 = int(round(N * float(args.val_ratio)))
        val_idx = idx[:n_val2]
        tr_idx = idx[n_val2:]

        Xtr, Ytr = X[tr_idx], Y[tr_idx]
        Xva, Yva = X[val_idx], Y[val_idx]

        ds_tr2 = TensorDataset(Xtr, Ytr)
        ds_va2 = TensorDataset(Xva, Yva)

        dl_tr2 = DataLoader(ds_tr2, batch_size=args.batch_size, shuffle=True, num_workers=0)
        dl_va2 = DataLoader(ds_va2, batch_size=args.batch_size, shuffle=False, num_workers=0)

        h_dim = int(X.shape[1])
        probe = LinearProbe(h_dim).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=args.lr)

        def eval_loss_cached(dl):
            probe.eval()
            losses = []
            with torch.no_grad():
                for xb, yb in dl:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    yhat = probe(xb)
                    loss = torch.mean((yhat - yb) ** 2)
                    losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else float("nan")

        best = float("inf")
        best_ep = -1
        bad = 0

        for ep in range(1, args.epochs + 1):
            probe.train()
            for xb, yb in dl_tr2:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat = probe(xb)
                loss = torch.mean((yhat - yb) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            va = eval_loss_cached(dl_va2)
            if va < best:
                best = va
                best_ep = ep
                bad = 0
                payload = {
                    "model": probe.state_dict(),
                    "rep_mode": args.rep_mode,
                    "use_force": bool(args.use_force),
                    "ckpt": args.ckpt,
                    "norm_cfg": norm_cfg,
                    "cache_path": str(cache_path),
                    "probe_meta": probe_meta,
                    "signature": signature,
                }
                Path(args.save_probe).parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, args.save_probe)
            else:
                bad += 1

            if ep % 20 == 0 or ep == 1:
                print(f"[probe-cached] epoch={ep:03d} val_mse={va:.6f} best={best:.6f} (best_ep={best_ep})")

            if args.early_stop and bad >= int(args.patience):
                print(f"[EARLY STOP] patience={args.patience} reached at epoch={ep} (best_ep={best_ep})")
                break

        out_txt.write_text(
            f"ckpt={args.ckpt}\nprobe={args.save_probe}\nrep_mode={args.rep_mode}\nuse_force={args.use_force}\n"
            f"cache_latents=True\ncache_path={cache_path}\n"
            f"signature={signature}\n"
            f"best_val_mse={best:.8f}\nbest_epoch={best_ep}\n",
            encoding="utf-8",
        )
        print("[OK] saved:", out_txt)
        print("[OK] saved probe:", args.save_probe)
        return

    # -------------------------
    # Option B: no cache
    # -------------------------
    with torch.no_grad():
        btmp = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b0.items()}
        out = wm(btmp)
        h_dim = int(out["h_seq"].shape[-1])

    probe = LinearProbe(h_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)

    def eval_loss(dl):
        probe.eval()
        losses = []
        with torch.no_grad():
            for batch in dl:
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = wm(batch)
                h = summarize_latent(out["h_seq"], mode=args.rep_mode)
                y = torch.stack([batch["mass"].float(), batch["friction"].float()], dim=1)
                yhat = probe(h)
                loss = torch.mean((yhat - y) ** 2)
                losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    best = float("inf")
    best_ep = -1
    bad = 0

    for ep in range(1, args.epochs + 1):
        probe.train()
        for batch in dl_tr:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                out = wm(batch)
                h = summarize_latent(out["h_seq"], mode=args.rep_mode)
            y = torch.stack([batch["mass"].float(), batch["friction"].float()], dim=1)

            yhat = probe(h)
            loss = torch.mean((yhat - y) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        va = eval_loss(dl_va)
        if va < best:
            best = va
            best_ep = ep
            bad = 0
            payload = {
                "model": probe.state_dict(),
                "rep_mode": args.rep_mode,
                "use_force": bool(args.use_force),
                "ckpt": args.ckpt,
                "norm_cfg": norm_cfg,
                "probe_meta": probe_meta,
                "signature": signature,
            }
            Path(args.save_probe).parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.save_probe)
        else:
            bad += 1

        if ep % 20 == 0 or ep == 1:
            print(f"[probe] epoch={ep:03d} val_mse={va:.6f} best={best:.6f} (best_ep={best_ep})")

        if args.early_stop and bad >= int(args.patience):
            print(f"[EARLY STOP] patience={args.patience} reached at epoch={ep} (best_ep={best_ep})")
            break

    out_txt.write_text(
        f"ckpt={args.ckpt}\nprobe={args.save_probe}\nrep_mode={args.rep_mode}\nuse_force={args.use_force}\n"
        f"cache_latents=False\n"
        f"signature={signature}\n"
        f"best_val_mse={best:.8f}\nbest_epoch={best_ep}\n",
        encoding="utf-8",
    )
    print("[OK] saved:", out_txt)
    print("[OK] saved probe:", args.save_probe)


if __name__ == "__main__":
    main()
