# src/wm/utils/rollout.py
from __future__ import annotations

import torch
from typing import Literal, Dict, Any, Optional

VisionMode = Literal["t0", "gt"]
InterveneMode = Literal["none", "all", "contact", "coast"]
ForceInputMode = Literal["auto", "on", "off"]


def _decode_model(model, h: torch.Tensor) -> Dict[str, torch.Tensor]:
    squeeze_time = False
    if h.ndim == 2:
        h_in = h.unsqueeze(1)
        squeeze_time = True
    elif h.ndim == 3:
        h_in = h
    else:
        raise ValueError(f"h must be (B,D) or (B,T,D), got {tuple(h.shape)}")

    if hasattr(model, "decode") and callable(getattr(model, "decode")):
        out = model.decode(h_in)
    elif hasattr(model, "decoder"):
        out = model.decoder(h_in)
    else:
        raise AttributeError("Model has neither decode() nor decoder")

    if isinstance(out, dict):
        out_dict = dict(out)
    elif isinstance(out, tuple):
        if len(out) < 3:
            raise ValueError(f"decoder tuple too short: len={len(out)}")
        out_dict = {"q_hat": out[0], "dq_hat": out[1], "block_pose_hat": out[2]}
        if len(out) >= 4:
            out_dict["force_hat"] = out[3]
    else:
        raise TypeError(f"decoder output must be dict or tuple, got {type(out)}")

    for k in ["q_hat", "dq_hat", "block_pose_hat"]:
        if k not in out_dict:
            raise KeyError(f"decoder output missing {k}. keys={list(out_dict.keys())}")

    if squeeze_time:
        for k, v in list(out_dict.items()):
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == 1:
                out_dict[k] = v[:, 0]

    return out_dict


def _compute_force_hat_from_head(model, h: torch.Tensor) -> Optional[torch.Tensor]:
    if not hasattr(model, "force_head"):
        return None
    head = getattr(model, "force_head")
    if not callable(head):
        return None

    if h.ndim == 3:
        h2 = h[:, 0] if h.shape[1] == 1 else h[:, -1]
    elif h.ndim == 2:
        h2 = h
    else:
        return None

    fh = head(h2)
    if not torch.is_tensor(fh):
        return None
    return fh


def _infer_force_embed_dim(model: torch.nn.Module, force_dim: int, device: torch.device) -> int:
    if not hasattr(model, "force_encoder"):
        return 0
    enc = getattr(model, "force_encoder")
    if not callable(enc):
        return 0
    z = torch.zeros(1, 1, int(force_dim), device=device)
    y = enc(z)
    return int(y.shape[-1])


@torch.no_grad()
def rollout(
    model,
    batch: Dict[str, torch.Tensor],
    horizon: int = 30,
    h0: torch.Tensor | None = None,

    # ----- force (only meaningful when model.force_in_dynamics==True) -----
    use_gt_force: bool = False,
    force_input_mode: ForceInputMode = "auto",

    gt_force_scale: float = 1.0,
    gt_force_pre_release: bool = False,
    gt_force_decay_lambda: float = 1.0,

    force_feedback: str = "hat",
    force_hat_scale: float = 1.0,
    force_cut_after_release: bool = False,
    force_decay_lambda: float = 1.0,
    force_hat_sign_x: float = 1.0,
    force_hat_sign_y: float = 1.0,
    force_hat_sign_z: float = 1.0,

    # ----- vision -----
    vision_mode: VisionMode = "t0",

    # ----- latent intervention -----
    intervene_direction: torch.Tensor | None = None,
    intervene_alpha: float = 0.0,
    intervene_normalize: bool = False,
    intervene_mode: InterveneMode = "none",

    # ----- segmentation boundary -----
    release_start: int | None = None,

    # ----- debug -----
    return_debug: bool = False,
) -> Dict[str, Any]:
    device = next(model.parameters()).device

    rgb = batch["rgb"].to(device)
    q = batch["q"].to(device)
    dq = batch["dq"].to(device)
    action = batch["action"].to(device)
    block_pose = batch["block_pose"].to(device)

    B, T = q.shape[0], q.shape[1]
    H = int(min(horizon, T - 1))
    if H <= 0:
        raise ValueError(f"horizon too small: horizon={horizon}, T={T}")

    # ★モデルが“forceをdynamicsに入れるか”を判定（VPF-aux strict は False）
    force_in_dyn = bool(getattr(model, "force_in_dynamics", False))

    has_force_encoder = hasattr(model, "force_encoder") and callable(getattr(model, "force_encoder"))
    has_force_head = hasattr(model, "force_head") and callable(getattr(model, "force_head"))
    has_f = ("f" in batch)

    f_gt = batch["f"].to(device) if has_f else None
    Fdim = int(f_gt.shape[-1]) if (f_gt is not None) else 0

    # release_start
    rs = None
    if release_start is not None:
        rs = int(max(0, min(int(release_start), H)))

    # normalize dir
    def maybe_normalize_dir(d: torch.Tensor) -> torch.Tensor:
        if not intervene_normalize:
            return d
        n = torch.norm(d) + 1e-9
        return d / n

    # debug
    dbg = {}
    if return_debug:
        dbg = {
            "force_gt_norm": [],
            "force_used_norm": [],
            "force_hat_norm": [],
            "h_norm": [],
            "dh_norm": [],
        }

    # ===== force embed settings only if force_in_dyn =====
    use_force_input = False
    force_embed_dim = 0
    last_contact_force = None

    if force_in_dyn and has_force_encoder and (f_gt is not None) and (Fdim > 0):
        if force_input_mode == "on":
            use_force_input = True
        elif force_input_mode == "off":
            use_force_input = False
        else:  # auto
            use_force_input = True

        force_embed_dim = _infer_force_embed_dim(model, Fdim, device=device)

        if use_gt_force and gt_force_pre_release and (rs is not None) and (rs > 0):
            last_contact_force = (f_gt[:, rs - 1] * float(gt_force_scale)).detach()

    # initial hidden
    if h0 is None:
        rgb0 = rgb[:, :1]
        q0 = q[:, :1]
        dq0 = dq[:, :1]
        bp0 = block_pose[:, :1]

        img_emb0 = model.image_encoder(rgb0)
        prop_emb0 = model.proprio_encoder(q0, dq0, bp0)

        if force_in_dyn and (force_embed_dim > 0):
            if use_force_input:
                if use_gt_force and (f_gt is not None):
                    f0_step = f_gt[:, :1] * float(gt_force_scale)
                else:
                    f0_step = torch.zeros(B, 1, Fdim, device=device)
                force_emb0 = model.force_encoder(f0_step)
            else:
                force_emb0 = torch.zeros(B, 1, force_embed_dim, device=device)

            emb0 = torch.cat([img_emb0[:, 0], prop_emb0[:, 0], force_emb0[:, 0]], dim=-1)
        else:
            emb0 = torch.cat([img_emb0[:, 0], prop_emb0[:, 0]], dim=-1)

        a0 = torch.zeros(B, model.action_dim, device=device)
        x0 = torch.cat([emb0, a0], dim=-1)
        h = model.dynamics.forward_step(x0, None)
    else:
        h = h0

    # feedback state
    f_used_prev = None
    if force_in_dyn and (force_embed_dim > 0) and use_force_input and (not use_gt_force) and (Fdim > 0):
        f_used_prev = torch.zeros(B, Fdim, device=device)

    q_hats, dq_hats, bp_hats = [], [], []
    force_hats = []
    force_hat_scaled = []

    for t in range(H):
        rgb_t = rgb[:, t:t + 1] if vision_mode == "gt" else rgb[:, :1]
        q_t = q[:, t:t + 1]
        dq_t = dq[:, t:t + 1]
        bp_t = block_pose[:, t:t + 1]
        a_t = action[:, t:t + 1]

        img_emb = model.image_encoder(rgb_t)
        prop_emb = model.proprio_encoder(q_t, dq_t, bp_t)

        f_in_step = None

        if force_in_dyn and (force_embed_dim > 0):
            if use_force_input:
                if use_gt_force:
                    if f_gt is None or Fdim == 0:
                        f_in_step = torch.zeros(B, 0, device=device)
                        f_in = torch.zeros(B, 1, 0, device=device)
                    else:
                        if gt_force_pre_release and (rs is not None) and (t >= rs):
                             # contact-only: after release, feed ZERO (hard cut)
                             f_in_step = torch.zeros(B, Fdim, device=device)
                        else:
                            f_in_step = f_gt[:, t] * float(gt_force_scale)

                        f_in = f_in_step.unsqueeze(1)
                else:
                    f_in_step = (f_used_prev if f_used_prev is not None else torch.zeros(B, Fdim, device=device))
                    f_in = f_in_step.unsqueeze(1)

                force_emb = model.force_encoder(f_in)
            else:
                force_emb = torch.zeros(B, 1, force_embed_dim, device=device)

            emb = torch.cat([img_emb[:, 0], prop_emb[:, 0], force_emb[:, 0]], dim=-1)
        else:
            emb = torch.cat([img_emb[:, 0], prop_emb[:, 0]], dim=-1)

        x = torch.cat([emb, a_t[:, 0]], dim=-1)

        h_prev = h
        h = model.dynamics.forward_step(x, h)

        # intervention
        if (intervene_mode != "none") and (intervene_direction is not None) and (intervene_alpha != 0.0):
            apply = False
            if intervene_mode == "all":
                apply = True
            elif intervene_mode == "contact":
                apply = (rs is not None and t < rs)
            elif intervene_mode == "coast":
                apply = (rs is not None and t >= rs)

            if apply:
                d = maybe_normalize_dir(intervene_direction)
                h = h + float(intervene_alpha) * d

        out = _decode_model(model, h)
        q_hat = out["q_hat"]
        dq_hat = out["dq_hat"]
        bp_hat = out["block_pose_hat"]

        q_hats.append(q_hat.unsqueeze(1))
        dq_hats.append(dq_hat.unsqueeze(1))
        bp_hats.append(bp_hat.unsqueeze(1))

        fh = out.get("force_hat", None)
        if fh is None and has_force_head:
            fh = _compute_force_hat_from_head(model, h)

        if fh is not None and torch.is_tensor(fh):
            if fh.ndim == 3:
                fh = fh[:, 0] if fh.shape[1] == 1 else fh[:, -1]
            elif fh.ndim != 2:
                fh = fh.view(B, -1)
            force_hats.append(fh.unsqueeze(1))

        # feedback update only when force_in_dyn
        if force_in_dyn and (force_embed_dim > 0) and use_force_input and (not use_gt_force) and (Fdim > 0):
            if fh is None or (not torch.is_tensor(fh)):
                fh2 = torch.zeros(B, Fdim, device=device)
            else:
                fh2 = fh
                if fh2.shape[-1] != Fdim:
                    fh2 = fh2[:, :Fdim] if fh2.shape[-1] > Fdim else torch.nn.functional.pad(
                        fh2, (0, Fdim - fh2.shape[-1])
                    )

            fh_scaled = fh2.clone()
            if Fdim >= 1:
                fh_scaled[:, 0] *= float(force_hat_sign_x)
            if Fdim >= 2:
                fh_scaled[:, 1] *= float(force_hat_sign_y)
            if Fdim >= 3:
                fh_scaled[:, 2] *= float(force_hat_sign_z)
            fh_scaled = fh_scaled * float(force_hat_scale)
            force_hat_scaled.append(fh_scaled.unsqueeze(1))

            if force_feedback == "zero":
                f_next = torch.zeros(B, Fdim, device=device)
            else:
                f_next = fh_scaled

            if rs is not None and (t + 1) >= rs:
                if force_cut_after_release:
                    f_next = torch.zeros_like(f_next)
                else:
                    lam = max(float(force_decay_lambda), 0.0)
                    k = int((t + 1) - rs + 1)
                    decay = torch.exp(torch.tensor(-lam * k, device=device, dtype=f_next.dtype))
                    f_next = f_next * decay



            f_used_prev = f_next

        if return_debug:
            dbg["h_norm"].append(torch.linalg.norm(h, dim=-1, keepdim=True))
            dbg["dh_norm"].append(torch.linalg.norm(h - h_prev, dim=-1, keepdim=True))

            if f_gt is not None and (Fdim > 0):
                gt_step = f_gt[:, t] * float(gt_force_scale)
                dbg["force_gt_norm"].append(torch.linalg.norm(gt_step, dim=-1, keepdim=True))
            else:
                dbg["force_gt_norm"].append(torch.zeros(B, 1, device=device))

            if force_in_dyn and (f_in_step is not None) and torch.is_tensor(f_in_step) and (f_in_step.numel() > 0):
                dbg["force_used_norm"].append(torch.linalg.norm(f_in_step, dim=-1, keepdim=True))
            else:
                dbg["force_used_norm"].append(torch.zeros(B, 1, device=device))

            if fh is not None and torch.is_tensor(fh):
                dbg["force_hat_norm"].append(torch.linalg.norm(fh, dim=-1, keepdim=True))
            else:
                dbg["force_hat_norm"].append(torch.zeros(B, 1, device=device))

    ret: Dict[str, Any] = {
        "q_hat": torch.cat(q_hats, dim=1),
        "dq_hat": torch.cat(dq_hats, dim=1),
        "block_pose_hat": torch.cat(bp_hats, dim=1),
    }

    if len(force_hats) > 0:
        ret["force_hat"] = torch.cat(force_hats, dim=1)
    if len(force_hat_scaled) > 0:
        ret["force_hat_scaled"] = torch.cat(force_hat_scaled, dim=1)

    if return_debug:
        dbg2 = {}
        for k, xs in dbg.items():
            if len(xs) == 0:
                continue
            dbg2[k] = torch.cat(xs, dim=1).squeeze(-1)
        ret["debug"] = dbg2

    return ret
