#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect.py（安定版データ収集：テーブルトップ / 世界モデル学習用）
=====================================================

★今回の変更（あなたの要件に合わせた本番仕様）
--------------------------------------------
1) 各(m, μ)について「OK が N 本揃うまで」繰り返す（N = --episodes-per-condition）
2) BAD は保存してよいが、カウントに入れない
3) BAD を混ぜないため、保存先を分離する：
   - OK  : datasets/raw/<split>/
   - BAD : datasets/raw/<split>/_bad/

注意
----
- 既存の eval 側が `datasets/raw/<split>/episode_*.npz` を glob しても
  _bad は別フォルダなので混入しません（これが狙い）。
- `--episode-id-start` は「新規採番の開始ID」。既存ファイルと衝突しない値にしてください。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data


# =========================
# 便利関数
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def resize_nn(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    in_h, in_w = img.shape[:2]
    ys = (np.arange(out_h) * (in_h / out_h)).astype(int)
    xs = (np.arange(out_w) * (in_w / out_w)).astype(int)
    return img[ys[:, None], xs[None, :], :]

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def save_video(frames_uint8: Sequence[np.ndarray], path: str, fps: int = 30) -> None:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    clip = ImageSequenceClip(list(frames_uint8), fps=fps)
    clip.write_videofile(path, codec="libx264", audio=False)

def fmt_m(m: float) -> str:
    return f"{m:.2f}"

def fmt_mu(mu: float) -> str:
    return f"{mu:.2f}"

def count_existing_ok(out_ok_dir: str, m: float, mu: float) -> int:
    """
    既存のOK本数をカウント。
    OK dir には OK しか置かない前提なので、単純 glob で良い。
    """
    pat = f"episode_*_m={fmt_m(m)}_mu={fmt_mu(mu)}.npz"
    full = os.path.join(out_ok_dir, pat)
    return len(sorted(glob_glob(full)))

def glob_glob(pattern: str) -> List[str]:
    import glob
    return glob.glob(pattern)


# =========================
# 設定（dataclass）
# =========================
@dataclass
class SimConfig:
    control_hz: int = 10
    delta_time: float = 1.0 / 240.0
    gravity: float = -9.8

    T: int = 80

    ee_z_offset: float = 0.08

    push_force: float = 250
    pos_gain: float = 0.05
    vel_gain: float = 1.0
    max_vel: float = 0.7

    push_ee_speed: float = 0.03

    num_solver_iters: int = 150
    num_substeps: int = 1
    enable_cone_friction: int = 1

    cam_w: int = 360
    cam_h: int = 240
    save_rgb_wh: Tuple[int, int] = (64, 64)

    save_preview_mp4: bool = False
    preview_fps: int = 30

    disable_robot_table_collision: bool = True

    release_steps: int = 40
    release_lift: float = 0.12
    release_backoff: float = 0.04

    release_backoff_step: float = 0.01
    release_backoff_max: float = 0.12

    release_pos_gain: float = 1.2
    release_max_vel: float = 6.0
    release_force: float = 2000.0
    release_start_iters: int = 3
    release_lift_only_steps: int = 3

    kick_steps: int = 3
    kick_multiplier: float = 3.0
    kick_pos_gain: float = 0.20
    kick_max_vel: float = 1.50
    kick_force: float = 350.0

    gui: bool = False
    gui_sleep: bool = True
    gui_cam_dist: float = 1.25
    gui_cam_yaw: float = 35.0
    gui_cam_pitch: float = -35.0

    settle_steps: int = 240

    push_height_mode: str = "side"  # offset / side
    push_side_bias: float = 0.003

    ik_use_fixed_orn: bool = True

    block_x: float = 0.60
    block_y: float = 0.00
    block_x_offset_toward_robot: float = 0.05

    ee_start_backoff: float = 0.13
    approach_z_high: float = 0.20

    init_noise_mode: str = "block"   # none / block
    noise_block_xy: float = 0.003

    # Force-target
    use_force_target: bool = True
    force_targets: Tuple[float, float] = (30.0, 70.0)
    adm_k: float = 0.0005
    v_min: float = 0.005
    v_max: float = 0.18

    # Air-force（今回は off で本番データを集める想定だが、コードは残す）
    use_air_force: bool = False
    air_force_steps: int = 30
    air_force_fx: float = 2.0
    air_z_lift: float = 0.15
    air_force_start_t: int = 0
    air_force_after_release_offset: int = 20
    air_hold_ee_high: float = 0.25

    air_zero_gravity: bool = True
    air_disable_damping: bool = True
    air_reset_vel_at_start: bool = True


@dataclass
class PolicyConfig:
    policy: str = "fixed"
    ensure_contact: bool = True
    contact_min_ratio: float = 0.30
    fixed_template_name: str = "grid_v1"

    approach_backoff: float = 0.08
    approach_step_limit: float = 0.006
    approach_y_clamp: float = 0.12
    pre_contact_max_iters: int = 60


@dataclass
class QCConfig:
    reject_q_jump: float = 0.60
    reject_max_nf_table: float = 800.0
    reject_max_ee_err: float = 0.080
    abort_table_contact_nf: float = 2000.0


@dataclass
class EpisodeResult:
    path: str
    is_bad: bool
    bad_reasons: List[str]
    stats: Dict[str, float]


def train_conditions() -> List[Tuple[float, float]]:
    m_list = [0.05, 0.2, 1, 10, 20]
    mu_list = [0.2, 0.35, 0.5, 0.65, 0.8]
    return [(m, mu) for m in m_list for mu in mu_list]

def intrap_conditions() -> List[Tuple[float, float]]:
    m_list = [0.65, 1.0, 1.4, 1.9]
    mu_list = [0.275, 0.425, 0.575, 0.725]
    conds = [(m, mu) for m in m_list for mu in mu_list]
    return conds[:12]

def extrap_conditions() -> List[Tuple[float, float]]:
    m_list  = [0.02, 0.03, 0.04, 25.0, 30.0, 40.0]
    mu_list = [0.2, 0.35, 0.5, 0.65]
    return [(m, mu) for m in m_list for mu in mu_list]

def build_fixed_template(name: str, push_mode: str) -> List[Dict]:
    if name == "grid_v1":
        segs: List[Dict] = []
        if push_mode == "vel":
            segs.append({"dx": 1.0, "dy": 0.0, "steps": 20, "label": "dir_x_1"})
            segs.append({"dx": 1.0, "dy": 0.0, "steps": 20, "label": "dir_x_2"})
            segs.append({"dx": 1.0, "dy": 0.3, "steps": 20, "label": "dir_diag"})
            return segs

        D, N = 0.03, 20
        segs.append({"dx": D / N, "dy": 0.0, "steps": N, "label": "slow_short_x"})
        D, N = 0.07, 20
        segs.append({"dx": D / N, "dy": 0.0, "steps": N, "label": "fast_long_x"})
        D, N = 0.05, 20
        segs.append({"dx": D / N, "dy": 0.0015, "steps": N, "label": "diag_xy"})
        D, N = 0.03, 10
        segs.append({"dx": -D / N, "dy": 0.0, "steps": N, "label": "reverse_x"})
        return segs

    raise ValueError(f"Unknown fixed template: {name}")

def estimate_contact_force_ee_on_block(c_block, ee_link_index: int) -> Tuple[np.ndarray, np.ndarray]:
    f_on_block = np.zeros(3, dtype=np.float32)
    for cp in c_block:
        linkA = cp[3]
        if linkA != ee_link_index:
            continue
        n_on_b = np.array(cp[7], dtype=np.float32)
        nf = float(cp[9])
        f = nf * n_on_b
        if len(cp) >= 14:
            lf1 = float(cp[10]); d1 = np.array(cp[11], dtype=np.float32)
            lf2 = float(cp[12]); d2 = np.array(cp[13], dtype=np.float32)
            f = f + lf1 * d1 + lf2 * d2
        f_on_block += f
    f_on_ee = -f_on_block
    return f_on_block.astype(np.float32), f_on_ee.astype(np.float32)

def estimate_contact_force_robot_on_block(c_block) -> Tuple[np.ndarray, np.ndarray]:
    f_on_block = np.zeros(3, dtype=np.float32)
    for cp in c_block:
        n_on_b = np.array(cp[7], dtype=np.float32)
        nf = float(cp[9])
        f = (-nf) * n_on_b
        if len(cp) >= 14:
            lf1 = float(cp[10]); d1 = np.array(cp[11], dtype=np.float32)
            lf2 = float(cp[12]); d2 = np.array(cp[13], dtype=np.float32)
            f = f + (-lf1) * d1 + (-lf2) * d2
        f_on_block += f
    f_on_robot = -f_on_block
    return f_on_block.astype(np.float32), f_on_robot.astype(np.float32)


def collect_one_episode(
    episode_id: int,
    out_ok_dir: str,
    out_bad_dir: str,
    mass: float,
    friction: float,
    rng: np.random.Generator,
    simcfg: SimConfig,
    polcfg: PolicyConfig,
    qccfg: QCConfig,
    push_mode: str,
    save_preview_dir: Optional[str] = None,
) -> EpisodeResult:
    """
    1エピソード分回して、OK/BAD 判定まで行い、保存先を分離して保存する。
    - OK  : out_ok_dir
    - BAD : out_bad_dir
    """

    cid = p.connect(p.GUI if simcfg.gui else p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, float(simcfg.gravity))
    p.setTimeStep(simcfg.delta_time)
    p.setPhysicsEngineParameter(
        fixedTimeStep=simcfg.delta_time,
        numSolverIterations=simcfg.num_solver_iters,
        numSubSteps=simcfg.num_substeps,
        enableConeFriction=simcfg.enable_cone_friction,
    )

    planeId = p.loadURDF("plane.urdf")
    tableId = p.loadURDF("table/table.urdf", basePosition=[0.5, 0.0, 0.0], useFixedBase=True)

    _, table_aabb_max = p.getAABB(tableId)
    table_top_z = table_aabb_max[2]
    p.changeDynamics(tableId, -1, lateralFriction=0.9, restitution=0.0)

    robotId = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True
    )
    num_joints = p.getNumJoints(robotId)
    ee_link = 6

    initial_q = [0, 0.3, 0, -1.2, 0, 1.0, 0.5]
    for i in range(num_joints):
        p.resetJointState(robotId, i, initial_q[i])

    joint_damping = [0.1] * num_joints
    rest_poses = initial_q

    if simcfg.disable_robot_table_collision:
        for j in range(-1, num_joints):
            p.setCollisionFilterPair(robotId, tableId, j, -1, enableCollision=0)
            p.setCollisionFilterPair(robotId, planeId, j, -1, enableCollision=0)

    block_half_extents = [0.05, 0.05, 0.02]
    block_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=block_half_extents)
    block_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=block_half_extents, rgbaColor=[1, 1, 1, 1])

    eps = 0.002
    block_center_z = table_top_z + block_half_extents[2] + eps

    mode = str(simcfg.init_noise_mode).lower()
    if mode not in ("none", "block"):
        raise ValueError("This version supports init_noise_mode in {'none','block'} only (EE start is fixed).")

    base_block_x = float(simcfg.block_x - simcfg.block_x_offset_toward_robot)
    base_block_y = float(simcfg.block_y)

    bx_n = 0.0
    by_n = 0.0
    if mode == "block":
        a = float(simcfg.noise_block_xy)
        bx_n = float(rng.uniform(-a, a))
        by_n = float(rng.uniform(-a, a))

    block_x_eff = float(base_block_x + bx_n)
    block_y_eff = float(base_block_y + by_n)

    blockId = p.createMultiBody(
        baseMass=float(mass),
        baseCollisionShapeIndex=block_col,
        baseVisualShapeIndex=block_vis,
        basePosition=[block_x_eff, block_y_eff, float(block_center_z)],
    )
    p.changeDynamics(
        blockId, -1,
        lateralFriction=float(friction),
        restitution=0.0,
        rollingFriction=0.0,
        spinningFriction=0.0,
        angularDamping=0.95,
        linearDamping=0.05,
    )

    if simcfg.gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=float(simcfg.gui_cam_dist),
            cameraYaw=float(simcfg.gui_cam_yaw),
            cameraPitch=float(simcfg.gui_cam_pitch),
            cameraTargetPosition=[0.60, 0.00, float(table_top_z + 0.20)],
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    for _ in range(int(simcfg.settle_steps)):
        p.stepSimulation()
        if simcfg.gui and simcfg.gui_sleep:
            time.sleep(simcfg.delta_time)

    init_link = p.getLinkState(robotId, ee_link, computeForwardKinematics=True)
    init_orn = init_link[5]

    if simcfg.push_height_mode == "side":
        z_push = float(block_center_z + simcfg.push_side_bias)
    else:
        z_push = float(table_top_z + simcfg.ee_z_offset)

    z_high = float(table_top_z + simcfg.ee_z_offset + simcfg.approach_z_high)

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.30, 0.60, 1.20],
        cameraTargetPosition=[0.60, 0.00, float(table_top_z + 0.20)],
        cameraUpVector=[0, 0, 1],
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=simcfg.cam_w / simcfg.cam_h, nearVal=0.1, farVal=5.0
    )

    def capture_rgb() -> np.ndarray:
        img = p.getCameraImage(
            simcfg.cam_w, simcfg.cam_h,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.reshape(img[2], (simcfg.cam_h, simcfg.cam_w, 4))[:, :, :3].astype(np.uint8)
        return resize_nn(rgb, simcfg.save_rgb_wh[0], simcfg.save_rgb_wh[1])

    rgb_buf: List[np.ndarray] = []
    q_buf: List[np.ndarray] = []
    dq_buf: List[np.ndarray] = []
    f_ee_buf: List[np.ndarray] = []
    action_buf: List[List[float]] = []
    block_pose_buf: List[List[float]] = []
    block_vel_buf: List[List[float]] = []
    is_contact_buf: List[int] = []
    phase_id_buf: List[int] = []
    action_mode_id_buf: List[int] = []

    ee_pos_buf: List[np.ndarray] = []
    target_pos_buf: List[np.ndarray] = []
    ee_err_buf: List[float] = []
    contacts_table_buf: List[int] = []
    contacts_block_buf: List[int] = []
    max_nf_table_buf: List[float] = []
    max_nf_block_buf: List[float] = []
    q_jump_buf: List[float] = []
    is_kick_buf: List[int] = []
    ee_speed_buf: List[float] = []
    f_push_buf: List[float] = []
    v_cmd_buf: List[float] = []

    # ===== FIX: control step 内 substep の最大接触を保存するバッファ =====
    contacts_block_step_max_buf: List[int] = []
    max_nf_block_step_max_buf: List[float] = []
    # ===========================================================

    f_ext_buf: List[np.ndarray] = []
    air_flag_buf: List[int] = []
    air_lifted_buf: List[int] = []

    policy_name = f"{polcfg.policy}_{polcfg.fixed_template_name}"

    cond_id = int(episode_id % 2)
    F_target = float(simcfg.force_targets[cond_id])

    SIM_HZ = int(round(1.0 / simcfg.delta_time))
    SUBSTEPS_PER_CTRL = int(round(SIM_HZ / simcfg.control_hz))

    def get_block_xy() -> Tuple[float, float]:
        pos, _ = p.getBasePositionAndOrientation(blockId)
        return float(pos[0]), float(pos[1])

    def current_contacts():
        c_table = p.getContactPoints(bodyA=robotId, bodyB=tableId)
        c_block = p.getContactPoints(bodyA=robotId, bodyB=blockId)
        return c_table, c_block

    def plan_approach_target(cur_x: float, cur_y: float, z_fixed: float) -> Tuple[float, float, float, float, float]:
        bx, by = get_block_xy()
        tx = bx - polcfg.approach_backoff
        ty = clamp(by, -polcfg.approach_y_clamp, polcfg.approach_y_clamp)
        dx = clamp(tx - cur_x, -polcfg.approach_step_limit, polcfg.approach_step_limit)
        dy = clamp(ty - cur_y, -polcfg.approach_step_limit, polcfg.approach_step_limit)
        return cur_x + dx, cur_y + dy, z_fixed, dx, dy

    def sample_random_dir() -> Tuple[float, float]:
        dx = float(rng.uniform(-1.0, 1.0))
        dy = float(rng.uniform(-1.0, 1.0))
        return dx, dy

    def compute_ik(target_pos: Sequence[float]) -> Sequence[float]:
        if simcfg.ik_use_fixed_orn:
            joint_poses = p.calculateInverseKinematics(
                robotId, ee_link,
                target_pos,
                targetOrientation=init_orn,
                restPoses=rest_poses,
                jointDamping=joint_damping,
                maxNumIterations=140,
                residualThreshold=1e-4,
            )
        else:
            joint_poses = p.calculateInverseKinematics(
                robotId, ee_link,
                target_pos,
                restPoses=rest_poses,
                jointDamping=joint_damping,
                maxNumIterations=140,
                residualThreshold=1e-4,
            )
        return joint_poses

    # ===== FIX: substep 内の robot-block 接触の最大値を返す =====
    def step_substeps_with_controls_and_optional_force(
        joint_poses: Sequence[float],
        pos_gain: float,
        max_vel: float,
        force: float,
        fx_world: float = 0.0,
        f_pos_world: Optional[Sequence[float]] = None,
    ) -> Tuple[int, float]:
        """
        1 control step (= SUBSTEPS_PER_CTRL 回の stepSimulation) を回す。
        その間の robot-block 接触点数の最大値 / block 側 normal force 最大値を返す。
        """
        cb_step_max = 0
        nf_block_step_max = 0.0

        for _ in range(SUBSTEPS_PER_CTRL):
            for j in range(num_joints):
                p.setJointMotorControl2(
                    robotId, j,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[j],
                    force=float(force),
                    positionGain=float(pos_gain),
                    velocityGain=float(simcfg.vel_gain),
                    maxVelocity=float(max_vel),
                )
            if (f_pos_world is not None) and (abs(fx_world) > 0.0):
                p.applyExternalForce(
                    objectUniqueId=blockId,
                    linkIndex=-1,
                    forceObj=[float(fx_world), 0.0, 0.0],
                    posObj=[float(f_pos_world[0]), float(f_pos_world[1]), float(f_pos_world[2])],
                    flags=p.WORLD_FRAME,
                )
            p.stepSimulation()
            if simcfg.gui and simcfg.gui_sleep:
                time.sleep(simcfg.delta_time)

            # substep ごとの接触を計測（robot-block）
            c_block_ss = p.getContactPoints(bodyA=robotId, bodyB=blockId)
            cb = len(c_block_ss)
            if cb > cb_step_max:
                cb_step_max = cb
            if cb > 0:
                nf = max([cp[9] for cp in c_block_ss], default=0.0)
                if nf > nf_block_step_max:
                    nf_block_step_max = float(nf)

        return int(cb_step_max), float(nf_block_step_max)
    # =======================================================

    # ===== FIX: apply_* が (cb_step_max, nf_step_max) を返す =====
    def apply_ee_target_pos(
        target_pos: Sequence[float],
        pos_gain_override: Optional[float] = None,
        max_vel_override: Optional[float] = None,
        force_override: Optional[float] = None,
    ) -> Tuple[int, float]:
        pos_gain = simcfg.pos_gain if pos_gain_override is None else float(pos_gain_override)
        max_vel = simcfg.max_vel if max_vel_override is None else float(max_vel_override)
        force = simcfg.push_force if force_override is None else float(force_override)
        joint_poses = compute_ik(target_pos)
        return step_substeps_with_controls_and_optional_force(
            joint_poses=joint_poses,
            pos_gain=pos_gain,
            max_vel=max_vel,
            force=force,
            fx_world=0.0,
            f_pos_world=None,
        )

    def normalize_to_step(dx: float, dy: float, step: float) -> Tuple[float, float]:
        n = float(math.sqrt(dx * dx + dy * dy))
        if n < 1e-9:
            return 0.0, 0.0
        return float(dx / n * step), float(dy / n * step)

    def apply_release_target(target_pos: Sequence[float]) -> Tuple[int, float]:
        return apply_ee_target_pos(
            target_pos,
            pos_gain_override=simcfg.release_pos_gain,
            max_vel_override=simcfg.release_max_vel,
            force_override=simcfg.release_force,
        )
    # ===========================================================

    # EE開始点
    x = float(base_block_x - simcfg.ee_start_backoff)
    y = float(base_block_y)

    ee_start_x_eff = float(x)
    ee_start_y_eff = float(y)

    for _ in range(8):
        _ = apply_ee_target_pos([x, y, z_push])  # (cb_step_max, nf_step_max) は捨てる

    if polcfg.ensure_contact:
        for _ in range(int(polcfg.pre_contact_max_iters)):
            _, c_block = current_contacts()
            if len(c_block) > 0:
                break
            x, y, _, _, _ = plan_approach_target(x, y, z_push)
            _ = apply_ee_target_pos([x, y, z_push])  # ここも捨てる

    prev_q: Optional[np.ndarray] = None
    fixed_segments = build_fixed_template(polcfg.fixed_template_name, push_mode=push_mode)
    seg_id, seg_step = 0, 0
    RANDOM_MODE_ID = 100

    contacts_block_last = 0
    aborted = False
    abort_reason = ""

    release_steps = int(max(0, min(simcfg.release_steps, simcfg.T)))
    release_start = int(simcfg.T - release_steps)

    kick_steps = int(max(0, simcfg.kick_steps))
    kick_mul = float(simcfg.kick_multiplier)
    kick_begin = max(0, release_start - kick_steps)
    release_backoff_done = False

    # ---- air-force区間（本番はOFF想定） ----
    air_steps = int(max(0, min(simcfg.air_force_steps, simcfg.T)))
    if simcfg.use_air_force and air_steps > 0:
        if int(simcfg.air_force_start_t) > 0:
            air_start = int(simcfg.air_force_start_t)
        else:
            air_start = int(min(simcfg.T - 1, release_start + int(simcfg.air_force_after_release_offset)))
        air_end = min(simcfg.T, air_start + air_steps)
        if air_start >= simcfg.T:
            air_start, air_end = -1, -1
    else:
        air_start, air_end = -1, -1

    safe_ee_x = float(ee_start_x_eff)
    safe_ee_y = float(ee_start_y_eff)
    safe_ee_z = float(table_top_z + simcfg.air_hold_ee_high)

    air_block_z = float(block_center_z + max(0.0, simcfg.air_z_lift))
    air_lift_applied = False
    air_base_xy: Optional[Tuple[float, float]] = None
    air_gravity_switched = False

    def lift_block_once(z_new: float) -> None:
        nonlocal air_lift_applied, air_base_xy
        pos, ori = p.getBasePositionAndOrientation(blockId)
        p.resetBasePositionAndOrientation(blockId, [pos[0], pos[1], z_new], ori)
        air_lift_applied = True
        air_base_xy = (float(pos[0]), float(pos[1]))

    for t in range(simcfg.T):
        z = z_push
        phase_id = 1
        action_mode_id = 0
        is_kick = 0

        F_meas = 0.0
        v_cmd = float(simcfg.push_ee_speed)
        f_ext = np.zeros(3, dtype=np.float32)
        air_flag = 0
        air_lifted = 0
        target_pos = [x, y, z_push]

        # ===== FIX: この control step 中の substep 最大接触をここに受ける =====
        cb_step_max_t = 0
        nf_block_step_max_t = 0.0
        # =============================================================

        # =========================
        # AIR-FORCE
        # =========================
        if (air_start >= 0) and (air_start <= t < air_end):
            phase_id = 3
            action_mode_id = -3
            air_flag = 1

            if (t == air_start) and simcfg.air_zero_gravity and (not air_gravity_switched):
                p.setGravity(0, 0, 0.0)
                air_gravity_switched = True

                if (not air_lift_applied) and (simcfg.air_z_lift > 0.0):
                    lift_block_once(air_block_z)

                if simcfg.air_disable_damping:
                    p.changeDynamics(
                        blockId, -1,
                        lateralFriction=0.0,
                        rollingFriction=0.0,
                        spinningFriction=0.0,
                        linearDamping=0.0,
                        angularDamping=0.0,
                    )

                if simcfg.air_reset_vel_at_start:
                    p.resetBaseVelocity(
                        blockId,
                        linearVelocity=[0.0, 0.0, 0.0],
                        angularVelocity=[0.0, 0.0, 0.0],
                    )

            if (not air_lift_applied) and (simcfg.air_z_lift > 0.0):
                lift_block_once(air_block_z)

            if air_lift_applied:
                air_lifted = 1

            if air_base_xy is None:
                pos_now, _ = p.getBasePositionAndOrientation(blockId)
                air_base_xy = (float(pos_now[0]), float(pos_now[1]))

            target_pos = [safe_ee_x, safe_ee_y, safe_ee_z]
            joint_poses = compute_ik(target_pos)

            fx = float(simcfg.air_force_fx)
            f_ext = np.array([fx, 0.0, 0.0], dtype=np.float32)

            cb_step_max_t, nf_block_step_max_t = step_substeps_with_controls_and_optional_force(
                joint_poses=joint_poses,
                pos_gain=float(simcfg.pos_gain),
                max_vel=float(simcfg.max_vel),
                force=float(simcfg.push_force),
                fx_world=fx,
                f_pos_world=[float(air_base_xy[0]), float(air_base_xy[1]), float(air_block_z)],
            )

            dx, dy = 0.0, 0.0

        elif simcfg.air_zero_gravity and air_gravity_switched and (t == air_end):
            p.setGravity(0, 0, float(simcfg.gravity))
            air_gravity_switched = False
            if simcfg.air_disable_damping:
                p.changeDynamics(
                    blockId, -1,
                    lateralFriction=float(friction),
                    rollingFriction=0.0,
                    spinningFriction=0.0,
                    linearDamping=0.05,
                    angularDamping=0.95,
                )
            # fallthrough

        # =========================
        # release
        # =========================
        elif t >= release_start:
            phase_id = 2
            action_mode_id = -2
            dx, dy = 0.0, 0.0

            z = z_push + simcfg.release_lift
            lift_only = (t < release_start + simcfg.release_lift_only_steps)

            _, c_block_now = current_contacts()
            contact_now = (len(c_block_now) > 0)

            if not lift_only:
                if (not release_backoff_done) and (not contact_now):
                    x -= float(simcfg.release_backoff)
                    release_backoff_done = True

                if contact_now:
                    x -= float(simcfg.release_backoff_step)

                x_min = float(ee_start_x_eff - simcfg.release_backoff_max)
                if x < x_min:
                    x = x_min

            target_pos = [x, y, z]

            if t == release_start:
                iters = int(max(1, simcfg.release_start_iters))
                cb_m = 0
                nf_m = 0.0
                for _ in range(iters):
                    cb_i, nf_i = apply_release_target(target_pos)
                    cb_m = max(cb_m, cb_i)
                    nf_m = max(nf_m, nf_i)
                cb_step_max_t, nf_block_step_max_t = cb_m, nf_m
            else:
                cb_step_max_t, nf_block_step_max_t = apply_release_target(target_pos)

        # =========================
        # push
        # =========================
        else:
            if polcfg.ensure_contact and (t > 0) and contacts_block_last == 0:
                phase_id = 0
                x, y, z, dx, dy = plan_approach_target(x, y, z)
                action_mode_id = -1
                target_pos = [x, y, z]
                cb_step_max_t, nf_block_step_max_t = apply_ee_target_pos(target_pos)
            else:
                phase_id = 1
                if polcfg.policy == "fixed":
                    seg = fixed_segments[seg_id]
                    dx, dy = float(seg["dx"]), float(seg["dy"])
                    action_mode_id = seg_id
                else:
                    dx, dy = sample_random_dir()
                    action_mode_id = RANDOM_MODE_ID

                if push_mode == "pos":
                    x += dx
                    y += dy

                elif push_mode == "vel":
                    v_base = float(simcfg.push_ee_speed)
                    v_cmd = v_base

                    n_dir = float(math.sqrt(dx*dx + dy*dy))
                    if n_dir < 1e-9:
                        dirx, diry = 0.0, 0.0
                    else:
                        dirx, diry = dx / n_dir, dy / n_dir

                    F_meas = 0.0
                    if simcfg.use_force_target:
                        _, c_block_now = current_contacts()
                        if len(c_block_now) > 0 and (abs(dirx) + abs(diry)) > 0:
                            f_on_block_now, _ = estimate_contact_force_robot_on_block(c_block_now)
                            F_meas = float(abs(f_on_block_now[0] * dirx + f_on_block_now[1] * diry))
                        else:
                            F_meas = 0.0

                        v_cmd = v_base + float(simcfg.adm_k) * (F_target - F_meas)
                        v_cmd = clamp(v_cmd, float(simcfg.v_min), float(simcfg.v_max))

                    if (kick_steps > 0) and (kick_mul > 1.0) and (t >= kick_begin) and (t < release_start):
                        v_cmd = clamp(v_cmd * kick_mul, float(simcfg.v_min), float(simcfg.v_max))
                        is_kick = 1

                    step_eff = float(v_cmd / simcfg.control_hz)
                    dx, dy = normalize_to_step(dx, dy, step_eff)
                    x += dx
                    y += dy

                else:
                    raise ValueError(f"Unknown push_mode: {push_mode}")

                target_pos = [x, y, z]
                if (
                    (push_mode == "vel")
                    and (kick_steps > 0)
                    and (kick_mul > 1.0)
                    and (t >= kick_begin)
                    and (t < release_start)
                ):
                    is_kick = 1
                    cb_step_max_t, nf_block_step_max_t = apply_ee_target_pos(
                        target_pos,
                        pos_gain_override=simcfg.kick_pos_gain,
                        max_vel_override=simcfg.kick_max_vel,
                        force_override=simcfg.kick_force,
                    )
                else:
                    cb_step_max_t, nf_block_step_max_t = apply_ee_target_pos(target_pos)

                if polcfg.policy == "fixed":
                    seg_step += 1
                    if seg_step >= fixed_segments[seg_id]["steps"]:
                        seg_id = (seg_id + 1) % len(fixed_segments)
                        seg_step = 0

        # ===== FIX: この時点で t の substep 最大値を保存 =====
        contacts_block_step_max_buf.append(int(cb_step_max_t))
        max_nf_block_step_max_buf.append(float(nf_block_step_max_t))
        # ====================================================

        # -------------------------
        # ログ保存（共通）
        # -------------------------
        rgb_buf.append(capture_rgb())

        joint_states = p.getJointStates(robotId, range(num_joints))
        q = np.array([js[0] for js in joint_states], dtype=np.float32)
        dq = np.array([js[1] for js in joint_states], dtype=np.float32)
        q_buf.append(q)
        dq_buf.append(dq)

        c_table, c_block = current_contacts()
        contacts_table = len(c_table)
        contacts_block = len(c_block)

        max_nf_table = max([cp[9] for cp in c_table], default=0.0)
        if max_nf_table > qccfg.abort_table_contact_nf:
            aborted = True
            abort_reason = f"ABORT:table_contact_nf={max_nf_table:.1f} > {qccfg.abort_table_contact_nf:.1f}"

        _, f_on_ee = estimate_contact_force_ee_on_block(c_block, ee_link)
        f_ee_buf.append(f_on_ee)

        if phase_id in (2, 3):
            action_buf.append([0.0, 0.0])
        else:
            action_buf.append([float(dx), float(dy)])

        pos, ori = p.getBasePositionAndOrientation(blockId)
        linvel, angvel = p.getBaseVelocity(blockId)
        block_pose_buf.append(list(pos) + list(ori))
        block_vel_buf.append(list(linvel) + list(angvel))

        is_contact = 1 if contacts_block > 0 else 0
        is_contact_buf.append(is_contact)
        phase_id_buf.append(int(phase_id))
        action_mode_id_buf.append(int(action_mode_id))

        link_state = p.getLinkState(robotId, ee_link, computeForwardKinematics=True)
        ee_pos = np.array(link_state[4], dtype=np.float32)
        ee_pos_buf.append(ee_pos)
        target_pos_buf.append(np.array(target_pos, dtype=np.float32))
        ee_err = float(np.linalg.norm(ee_pos - np.array(target_pos, dtype=np.float32)))
        ee_err_buf.append(ee_err)

        max_nf_block = max([cp[9] for cp in c_block], default=0.0)
        contacts_table_buf.append(contacts_table)
        contacts_block_buf.append(contacts_block)
        max_nf_table_buf.append(float(max_nf_table))
        max_nf_block_buf.append(float(max_nf_block))

        if prev_q is None:
            q_jump = 0.0
        else:
            q_jump = float(np.max(np.abs(q - prev_q)))
        q_jump_buf.append(q_jump)
        prev_q = q.copy()

        contacts_block_last = contacts_block
        is_kick_buf.append(int(is_kick))

        if len(ee_pos_buf) >= 2:
            v = float(np.linalg.norm(ee_pos_buf[-1] - ee_pos_buf[-2]) * simcfg.control_hz)
        else:
            v = 0.0
        ee_speed_buf.append(v)

        if (t < release_start) and (phase_id == 1) and (air_flag == 0):
            f_push_buf.append(float(F_meas))
            v_cmd_buf.append(float(v_cmd))
        else:
            f_push_buf.append(0.0)
            v_cmd_buf.append(0.0)

        f_ext_buf.append(f_ext.astype(np.float32))
        air_flag_buf.append(int(air_flag))
        air_lifted_buf.append(int(air_lifted))

        if aborted:
            break

    # air中断/abortでも重力復帰
    if simcfg.air_zero_gravity and air_gravity_switched:
        p.setGravity(0, 0, float(simcfg.gravity))
        air_gravity_switched = False
        if simcfg.air_disable_damping:
            p.changeDynamics(
                blockId, -1,
                lateralFriction=float(friction),
                rollingFriction=0.0,
                spinningFriction=0.0,
                linearDamping=0.05,
                angularDamping=0.95,
            )

    def safe_max(xs: Sequence[float]) -> float:
        return float(np.max(np.array(xs, dtype=np.float32))) if len(xs) else 0.0

    contact_ratio = float(np.mean(np.array(is_contact_buf, dtype=np.float32))) if len(is_contact_buf) else 0.0
    max_q_jump = safe_max(q_jump_buf)
    max_nf_table_all = safe_max(max_nf_table_buf)
    max_ee_err = safe_max(ee_err_buf)

    bad_reasons: List[str] = []
    if aborted:
        bad_reasons.append(abort_reason)
    if polcfg.ensure_contact and contact_ratio < polcfg.contact_min_ratio:
        bad_reasons.append(f"low_contact_ratio={contact_ratio:.2f} < {polcfg.contact_min_ratio:.2f}")
    if max_q_jump > qccfg.reject_q_jump:
        bad_reasons.append(f"q_jump={max_q_jump:.2f} > {qccfg.reject_q_jump:.2f}")
    if max_nf_table_all > qccfg.reject_max_nf_table:
        bad_reasons.append(f"max_nf_table={max_nf_table_all:.1f} > {qccfg.reject_max_nf_table:.1f}")
    if max_ee_err > qccfg.reject_max_ee_err:
        bad_reasons.append(f"max_ee_err={max_ee_err:.3f} > {qccfg.reject_max_ee_err:.3f}")

    is_bad = len(bad_reasons) > 0

    # =========================
    # 保存（OK/BADで分離）
    # =========================
    ensure_dir(out_ok_dir)
    ensure_dir(out_bad_dir)

    filename = f"episode_{episode_id:06d}_m={mass:.2f}_mu={friction:.2f}.npz"
    out_dir = out_bad_dir if is_bad else out_ok_dir
    out_path = os.path.join(out_dir, filename)

    np.savez_compressed(
        out_path,
        rgb=np.array(rgb_buf, dtype=np.uint8),
        q=np.stack(q_buf, axis=0) if len(q_buf) else np.zeros((0, num_joints), dtype=np.float32),
        dq=np.stack(dq_buf, axis=0) if len(dq_buf) else np.zeros((0, num_joints), dtype=np.float32),
        f=np.stack(f_ee_buf, axis=0) if len(f_ee_buf) else np.zeros((0, 3), dtype=np.float32),
        action=np.array(action_buf, dtype=np.float32),
        block_pose=np.array(block_pose_buf, dtype=np.float32),
        block_vel=np.array(block_vel_buf, dtype=np.float32),
        is_contact=np.array(is_contact_buf, dtype=np.int32),
        phase_id=np.array(phase_id_buf, dtype=np.int32),
        action_mode_id=np.array(action_mode_id_buf, dtype=np.int32),
        policy_name=np.array([policy_name], dtype=object),
        push_mode=np.array([push_mode], dtype=object),

        mass=float(mass),
        friction=float(friction),

        ee_pos=np.stack(ee_pos_buf, axis=0) if len(ee_pos_buf) else np.zeros((0, 3), dtype=np.float32),
        target_pos=np.stack(target_pos_buf, axis=0) if len(target_pos_buf) else np.zeros((0, 3), dtype=np.float32),
        ee_err=np.array(ee_err_buf, dtype=np.float32),

        contacts_table=np.array(contacts_table_buf, dtype=np.int32),
        contacts_block=np.array(contacts_block_buf, dtype=np.int32),
        max_nf_table=np.array(max_nf_table_buf, dtype=np.float32),
        max_nf_block=np.array(max_nf_block_buf, dtype=np.float32),

        # ===== FIX: 追加保存（あなたのデバッグコードが欲しいキー）=====
        contacts_block_step_max=np.array(contacts_block_step_max_buf, dtype=np.int32),
        max_nf_block_step_max=np.array(max_nf_block_step_max_buf, dtype=np.float32),
        # ============================================================

        q_jump=np.array(q_jump_buf, dtype=np.float32),

        table_top_z=float(table_top_z),
        contact_ratio=float(contact_ratio),
        max_q_jump=float(max_q_jump),
        max_nf_table_all=float(max_nf_table_all),
        max_ee_err=float(max_ee_err),
        aborted=bool(aborted),
        abort_reason=str(abort_reason),
        is_bad=bool(is_bad),
        bad_reasons=np.array(bad_reasons, dtype=object),

        episode_id=int(episode_id),
        push_ee_speed=float(simcfg.push_ee_speed),
        control_hz=int(simcfg.control_hz),
        push_height_mode=str(simcfg.push_height_mode),
        z_push=float(z_push),
        z_high=float(z_high),
        ik_use_fixed_orn=bool(simcfg.ik_use_fixed_orn),
        block_x=float(simcfg.block_x),
        block_y=float(simcfg.block_y),
        ee_start_backoff=float(simcfg.ee_start_backoff),
        block_x_eff=float(block_x_eff),
        block_y_eff=float(block_y_eff),
        block_x_offset_toward_robot=float(simcfg.block_x_offset_toward_robot),

        init_noise_mode=str(simcfg.init_noise_mode),
        noise_block_xy=float(simcfg.noise_block_xy),
        block_xy_noise=np.array([bx_n, by_n], dtype=np.float32),

        ee_start_x_eff=float(ee_start_x_eff),
        ee_start_y_eff=float(ee_start_y_eff),

        release_steps=int(simcfg.release_steps),
        release_lift=float(simcfg.release_lift),
        release_backoff=float(simcfg.release_backoff),
        release_backoff_step=float(simcfg.release_backoff_step),
        release_backoff_max=float(simcfg.release_backoff_max),
        release_pos_gain=float(simcfg.release_pos_gain),
        release_max_vel=float(simcfg.release_max_vel),
        release_force=float(simcfg.release_force),
        release_start_iters=int(simcfg.release_start_iters),

        kick_steps=int(simcfg.kick_steps),
        kick_multiplier=float(simcfg.kick_multiplier),
        kick_pos_gain=float(simcfg.kick_pos_gain),
        kick_max_vel=float(simcfg.kick_max_vel),
        kick_force=float(simcfg.kick_force),
        is_kick=np.array(is_kick_buf, dtype=np.int32),

        ee_speed=np.array(ee_speed_buf, dtype=np.float32),

        cond_id=int(cond_id),
        force_target=float(F_target),
        f_push=np.array(f_push_buf, dtype=np.float32),
        v_cmd=np.array(v_cmd_buf, dtype=np.float32),

        use_air_force=bool(simcfg.use_air_force),
        air_force_steps=int(simcfg.air_force_steps),
        air_force_fx=float(simcfg.air_force_fx),
        air_z_lift=float(simcfg.air_z_lift),
        air_force_start_t=int(air_start if air_start >= 0 else -1),
        air_force_end_t=int(air_end if air_end >= 0 else -1),
        air_force_after_release_offset=int(simcfg.air_force_after_release_offset),

        air_zero_gravity=bool(simcfg.air_zero_gravity),

        f_ext=np.stack(f_ext_buf, axis=0) if len(f_ext_buf) else np.zeros((0, 3), dtype=np.float32),
        is_air_force=np.array(air_flag_buf, dtype=np.int32),
        air_lifted=np.array(air_lifted_buf, dtype=np.int32),
    )

    if simcfg.save_preview_mp4 and save_preview_dir is not None and len(rgb_buf) > 0:
        ensure_dir(save_preview_dir)
        preview_path = os.path.join(save_preview_dir, f"episode_{episode_id:06d}_preview.mp4")
        preview_frames = [resize_nn(fr, 256, 256) for fr in rgb_buf]
        save_video(preview_frames, preview_path, fps=simcfg.preview_fps)

    if simcfg.gui:
        print("[GUI] エピソード終了。ウィンドウを閉じるか、Ctrl+Cで終了してください。")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass

    p.disconnect(cid)

    stats = {
        "contact_ratio": float(contact_ratio),
        "max_q_jump": float(max_q_jump),
        "max_nf_table": float(max_nf_table_all),
        "max_ee_err": float(max_ee_err),
        "aborted": float(1.0 if aborted else 0.0),
        "T_saved": float(len(rgb_buf)),
    }
    return EpisodeResult(path=out_path, is_bad=is_bad, bad_reasons=bad_reasons, stats=stats)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out-root", type=str, default="datasets/raw")
    ap.add_argument("--episodes-per-condition", type=int, default=1,
                    help="必要なOK本数（各条件でこの本数が揃うまで回す）")
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--episode-id-start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--policy", type=str, default="fixed", choices=["fixed", "random"])
    ap.add_argument("--control-hz", type=int, default=10)
    ap.add_argument("--T", type=int, default=80)

    ap.add_argument("--save-preview", action="store_true")
    ap.add_argument("--preview-fps", type=int, default=30)

    ap.add_argument("--mass", type=float, default=None)
    ap.add_argument("--friction", type=float, default=None)

    ap.add_argument("--release-steps", type=int, default=40)
    ap.add_argument("--release-lift", type=float, default=0.12)
    ap.add_argument("--release-backoff", type=float, default=0.04)

    ap.add_argument("--kick-steps", type=int, default=3)
    ap.add_argument("--kick-multiplier", type=float, default=3.0)
    ap.add_argument("--kick-pos-gain", type=float, default=0.20)
    ap.add_argument("--kick-max-vel", type=float, default=1.50)
    ap.add_argument("--kick-force", type=float, default=350.0)

    ap.add_argument("--push-mode", type=str, default="vel", choices=["pos", "vel"])
    ap.add_argument("--push-ee-speed", type=float, default=0.03)

    ap.add_argument("--cam-w", type=int, default=360)
    ap.add_argument("--cam-h", type=int, default=240)
    ap.add_argument("--save-rgb", type=int, default=64)

    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-gui-sleep", action="store_true")

    ap.add_argument("--push-height-mode", type=str, default="side", choices=["offset", "side"])
    ap.add_argument("--push-side-bias", type=float, default=0.003)

    ap.add_argument("--ik-fixed-orn", action="store_true")
    ap.add_argument("--ik-free-orn", action="store_true")

    ap.add_argument("--settle-steps", type=int, default=240)

    ap.add_argument("--block-x", type=float, default=0.60)
    ap.add_argument("--block-y", type=float, default=0.00)

    ap.add_argument("--ee-start-backoff", type=float, default=0.33)
    ap.add_argument("--approach-z-high", type=float, default=0.20)

    ap.add_argument("--init-noise-mode", type=str, default="block", choices=["none", "block"])
    ap.add_argument("--noise-block-xy", type=float, default=0.003)

    ap.add_argument("--no-force-target", action="store_true")

    ap.add_argument("--use-air-force", action="store_true")
    ap.add_argument("--air-force-steps", type=int, default=30)
    ap.add_argument("--air-force-fx", type=float, default=2.0)
    ap.add_argument("--air-z-lift", type=float, default=0.15)
    ap.add_argument("--air-force-start-t", type=int, default=0)
    ap.add_argument("--air-force-after-release-offset", type=int, default=20)

    ap.add_argument("--air-zero-gravity", action="store_true")
    ap.add_argument("--no-air-zero-gravity", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.ik_fixed_orn and args.ik_free_orn:
        raise ValueError("Choose only one: --ik-fixed-orn or --ik-free-orn")

    ik_use_fixed = True
    if args.ik_free_orn:
        ik_use_fixed = False
    if args.ik_fixed_orn:
        ik_use_fixed = True

    use_force_target = (not bool(args.no_force_target))

    air_zero_g = True
    if args.no_air_zero_gravity:
        air_zero_g = False
    if args.air_zero_gravity:
        air_zero_g = True

    simcfg = SimConfig(
        control_hz=int(args.control_hz),
        T=int(args.T),
        save_preview_mp4=bool(args.save_preview),
        preview_fps=int(args.preview_fps),
        release_steps=int(args.release_steps),
        release_lift=float(args.release_lift),
        release_backoff=float(args.release_backoff),
        kick_steps=int(args.kick_steps),
        kick_multiplier=float(args.kick_multiplier),
        kick_pos_gain=float(args.kick_pos_gain),
        kick_max_vel=float(args.kick_max_vel),
        kick_force=float(args.kick_force),
        push_ee_speed=float(args.push_ee_speed),
        cam_w=int(args.cam_w),
        cam_h=int(args.cam_h),
        save_rgb_wh=(int(args.save_rgb), int(args.save_rgb)),
        gui=bool(args.gui),
        gui_sleep=(not bool(args.no_gui_sleep)),
        push_height_mode=str(args.push_height_mode),
        push_side_bias=float(args.push_side_bias),
        ik_use_fixed_orn=bool(ik_use_fixed),
        settle_steps=int(args.settle_steps),
        block_x=float(args.block_x),
        block_y=float(args.block_y),
        ee_start_backoff=float(args.ee_start_backoff),
        approach_z_high=float(args.approach_z_high),
        init_noise_mode=str(args.init_noise_mode),
        noise_block_xy=float(args.noise_block_xy),
        use_force_target=use_force_target,

        use_air_force=bool(args.use_air_force),
        air_force_steps=int(args.air_force_steps),
        air_force_fx=float(args.air_force_fx),
        air_z_lift=float(args.air_z_lift),
        air_force_start_t=int(args.air_force_start_t),
        air_force_after_release_offset=int(args.air_force_after_release_offset),

        air_zero_gravity=bool(air_zero_g),
    )
    polcfg = PolicyConfig(policy=str(args.policy))
    qccfg = QCConfig()
    push_mode = str(args.push_mode)

    if args.episodes and (args.mass is None or args.friction is None):
        raise ValueError("--episodes requires --mass and --friction")

    # 収集条件の決定
    if args.episodes > 0:
        conditions = [(float(args.mass), float(args.friction))]
        require_ok = int(args.episodes)
    else:
        split = str(args.split)
        if split.startswith("train"):
            conditions = train_conditions()
        elif split.startswith("test_intrap") or split.startswith("test_interp"):
            conditions = intrap_conditions()
        elif split.startswith("test_extrap"):
            conditions = extrap_conditions()
        elif split.startswith("debug"):
            conditions = [(1.0, 0.5)]
        else:
            conditions = train_conditions()

        require_ok = int(args.episodes_per_condition)

    # 保存先（OK/BAD 分離）
    out_ok_dir = os.path.join(args.out_root, args.split)
    out_bad_dir = os.path.join(out_ok_dir, "_bad")
    preview_dir = os.path.join("outputs/videos", args.split) if simcfg.save_preview_mp4 else None

    ensure_dir(out_ok_dir)
    ensure_dir(out_bad_dir)
    if preview_dir is not None:
        ensure_dir(preview_dir)

    print(f"[INFO] split={args.split} conditions={len(conditions)} require_ok_per_condition={require_ok}")
    print(f"[INFO] out_ok={out_ok_dir}")
    print(f"[INFO] out_bad={out_bad_dir}")
    print(f"[INFO] policy={polcfg.policy} control_hz={simcfg.control_hz} T={simcfg.T} push_mode={push_mode}")
    print(f"[INFO] use_air_force={simcfg.use_air_force} air_steps={simcfg.air_force_steps} air_fx={simcfg.air_force_fx} "
          f"air_z_lift={simcfg.air_z_lift} air_start_t={simcfg.air_force_start_t} "
          f"air_after_release_offset={simcfg.air_force_after_release_offset} air_zero_gravity={simcfg.air_zero_gravity}")

    # summary / bad list
    bad_list: List[Dict] = []
    summary: Dict[str, object] = {
        "split": args.split,
        "out_ok_dir": out_ok_dir,
        "out_bad_dir": out_bad_dir,
        "policy": polcfg.policy,
        "control_hz": simcfg.control_hz,
        "T": simcfg.T,
        "push_mode": push_mode,
        "push_ee_speed": simcfg.push_ee_speed,
        "require_ok_per_condition": require_ok,
        "conditions": conditions,
        "good_count": 0,
        "bad_count": 0,
        "attempt_count": 0,
        "per_condition": [],
        "gui": simcfg.gui,
        "push_height_mode": simcfg.push_height_mode,
        "ik_fixed_orn": simcfg.ik_use_fixed_orn,
        "block_x": simcfg.block_x,
        "block_y": simcfg.block_y,
        "ee_start_backoff": simcfg.ee_start_backoff,
        "approach_z_high": simcfg.approach_z_high,
        "init_noise_mode": simcfg.init_noise_mode,
        "noise_block_xy": simcfg.noise_block_xy,
        "release_steps": simcfg.release_steps,
        "release_pos_gain": simcfg.release_pos_gain,
        "release_max_vel": simcfg.release_max_vel,
        "release_force": simcfg.release_force,
        "release_start_iters": simcfg.release_start_iters,
        "kick_steps": simcfg.kick_steps,
        "kick_multiplier": simcfg.kick_multiplier,
        "use_force_target": simcfg.use_force_target,
        "use_air_force": simcfg.use_air_force,
        "air_force_steps": simcfg.air_force_steps,
        "air_force_fx": simcfg.air_force_fx,
        "air_z_lift": simcfg.air_z_lift,
        "air_force_start_t": simcfg.air_force_start_t,
        "air_force_after_release_offset": simcfg.air_force_after_release_offset,
        "air_zero_gravity": simcfg.air_zero_gravity,
    }

    ep_id = int(args.episode_id_start)

    # 各条件で OK が require_ok 本揃うまで回す
    for (m, mu) in conditions:
        ok_now = count_existing_ok(out_ok_dir, float(m), float(mu))
        attempts = 0
        added_ok = 0
        added_bad = 0

        if ok_now >= require_ok:
            print(f"[SKIP] m={m:.2f} mu={mu:.2f} already_ok={ok_now}/{require_ok}")
            summary["per_condition"].append({
                "mass": float(m),
                "friction": float(mu),
                "ok_existing": int(ok_now),
                "ok_added": 0,
                "bad_added": 0,
                "attempts": 0,
                "done": True,
            })
            continue

        print(f"\n---- condition m={m:.2f} mu={mu:.2f} start_ok={ok_now}/{require_ok} ----")

        while ok_now < require_ok:
            rng = np.random.default_rng(int(args.seed) + ep_id)

            res = collect_one_episode(
                episode_id=ep_id,
                out_ok_dir=out_ok_dir,
                out_bad_dir=out_bad_dir,
                mass=float(m),
                friction=float(mu),
                rng=rng,
                simcfg=simcfg,
                polcfg=polcfg,
                qccfg=qccfg,
                push_mode=push_mode,
                save_preview_dir=preview_dir,
            )

            attempts += 1
            summary["attempt_count"] = int(summary["attempt_count"]) + 1

            tag = "BAD" if res.is_bad else "OK"
            if res.is_bad:
                added_bad += 1
                summary["bad_count"] = int(summary["bad_count"]) + 1
                bad_list.append({
                    "episode_id": ep_id,
                    "path": res.path,
                    "mass": float(m),
                    "friction": float(mu),
                    "bad_reasons": res.bad_reasons,
                    "stats": res.stats,
                })
            else:
                ok_now += 1
                added_ok += 1
                summary["good_count"] = int(summary["good_count"]) + 1

            print(
                f"[try={attempts:03d}] id={ep_id:06d} m={m:.2f} mu={mu:.2f} "
                f"contact={res.stats['contact_ratio']:.2f} "
                f"qjump={res.stats['max_q_jump']:.2f} "
                f"nf_table={res.stats['max_nf_table']:.1f} "
                f"ee_err={res.stats['max_ee_err']:.3f} "
                f"savedT={int(res.stats['T_saved'])} {tag} "
                f"|| ok={ok_now}/{require_ok}"
            )

            ep_id += 1  # 採番は常に進める（BADでもIDを消費する）

        print(f"[DONE] condition m={m:.2f} mu={mu:.2f} ok_added={added_ok} bad_added={added_bad} attempts={attempts}")

        summary["per_condition"].append({
            "mass": float(m),
            "friction": float(mu),
            "ok_existing": int(count_existing_ok(out_ok_dir, float(m), float(mu)) - added_ok),
            "ok_added": int(added_ok),
            "bad_added": int(added_bad),
            "attempts": int(attempts),
            "done": True,
        })

    # 保存
    summary_path = os.path.join(out_ok_dir, "summary.json")
    bad_path = os.path.join(out_ok_dir, "bad_episodes.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(bad_path, "w", encoding="utf-8") as f:
        json.dump(bad_list, f, ensure_ascii=False, indent=2)

    print("\n✅ Done.")
    print("[INFO] saved summary:", summary_path)
    print("[INFO] saved bad list:", bad_path)
    print(f"[INFO] good={summary['good_count']} bad={summary['bad_count']} attempts={summary['attempt_count']}")
    if simcfg.save_preview_mp4 and preview_dir is not None:
        print("[INFO] preview videos:", preview_dir)


if __name__ == "__main__":
    main()
