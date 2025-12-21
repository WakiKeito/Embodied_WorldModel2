import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data


# =========================
# Config
# =========================

@dataclass
class CollectConfig:
    out_dir: str = "data_npz_franka"
    gui: bool = False

    # episode
    T: int = 64
    hz: int = 30
    frame_skip: int = 1

    # camera
    img_w: int = 128
    img_h: int = 128
    fov: float = 60.0
    near: float = 0.01
    far: float = 5.0

    # physics grid
    masses: tuple = (0.2, 0.5, 1.0)
    frictions: tuple = (0.2, 0.6, 1.0)
    episodes_per_cell: int = 10

    # task geometry
    table_z: float = 0.0
    block_start_xy: tuple = (0.60, 0.00)
    goal_xy: tuple = (0.75, 0.00)         # goal is for script; not saved

    # push motion
    approach_height: float = 0.15         # above table
    contact_height: float = 0.025         # near table/block top
    ee_speed: float = 0.10                # m/s (for push)
    settle_steps: int = 30

    # control
    pos_kp: float = 0.08                  # position control gain-ish
    max_force: float = 120.0              # per joint


# =========================
# Helpers
# =========================

def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)

def pose7_from_base(body_id: int) -> np.ndarray:
    pos, orn = p.getBasePositionAndOrientation(body_id)  # orn: (x,y,z,w)
    pos = np.array(pos, dtype=np.float32)
    orn = np.array(orn, dtype=np.float32)
    return np.concatenate([pos, orn], axis=0).astype(np.float32)  # (7,)

def set_body_mass(body_id: int, mass: float):
    p.changeDynamics(body_id, -1, mass=mass)

def set_body_friction(body_id: int, friction: float):
    p.changeDynamics(body_id, -1, lateralFriction=friction)

def set_floor_friction(plane_id: int, friction: float):
    p.changeDynamics(plane_id, -1, lateralFriction=friction)

def render_rgb(cfg: CollectConfig, cam_target, cam_dist, cam_yaw, cam_pitch):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target,
        distance=cam_dist,
        yaw=cam_yaw,
        pitch=cam_pitch,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=cfg.fov,
        aspect=float(cfg.img_w) / float(cfg.img_h),
        nearVal=cfg.near,
        farVal=cfg.far,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=cfg.img_w,
        height=cfg.img_h,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL if cfg.gui else p.ER_TINY_RENDERER,
    )
    rgba = np.array(rgba, dtype=np.uint8).reshape(cfg.img_h, cfg.img_w, 4)
    return rgba[..., :3]  # uint8 (H,W,3)

def get_joint_indices_franka(robot_id: int):
    """
    Franka Panda: 7 arm joints + 2 finger joints
    あなたの仕様では q,dq,f は (T,J) なので、ここでは「腕7関節のみ」をJとするのが無難。
    """
    joint_names = []
    joint_indices = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        name = info[1].decode("utf-8")
        jtype = info[2]
        # arm joints are revolute; fingers are prismatic
        if jtype in (p.JOINT_REVOLUTE,):
            joint_indices.append(j)
            joint_names.append(name)

    # Pandaの腕7関節だけに絞る（多すぎる場合の保険）
    if len(joint_indices) > 7:
        joint_indices = joint_indices[:7]
        joint_names = joint_names[:7]

    return joint_indices, joint_names

def get_joint_states(robot_id: int, joint_indices):
    states = p.getJointStates(robot_id, joint_indices)
    q = np.array([s[0] for s in states], dtype=np.float32)
    dq = np.array([s[1] for s in states], dtype=np.float32)
    # f: applied motor torque (scalar per joint) -> (J,)
    f = np.array([s[3] for s in states], dtype=np.float32)
    return q, dq, f

def step_sim(cfg: CollectConfig):
    for _ in range(cfg.frame_skip):
        p.stepSimulation()
        if cfg.gui:
            time.sleep(1.0 / cfg.hz)

def ik_to_joint_positions(robot_id: int, ee_link: int, target_pos, target_orn=None):
    if target_orn is None:
        # end-effector orientation: keep tool pointing down-ish
        # (x,y,z,w) quaternion; tweak if needed
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
    q = p.calculateInverseKinematics(robot_id, ee_link, target_pos, target_orn)
    return q

def apply_joint_position_control(robot_id: int, joint_indices, target_positions, cfg: CollectConfig):
    # only pass positions for those joints
    tpos = [float(target_positions[i]) for i in range(len(joint_indices))]
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=tpos,
        forces=[cfg.max_force] * len(joint_indices),
    )


# =========================
# Scene setup (Franka + plane + table + block)
# =========================

def setup_scene(cfg: CollectConfig, mass: float, friction: float, seed: int):
    if cfg.gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    rng = np.random.default_rng(seed)

    plane_id = p.loadURDF("plane.urdf")

    # table (simple)
    table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0.0, cfg.table_z])

    # Franka Panda
    # pybullet_data には franka_panda/panda.urdf がある環境が多い
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    joint_indices, joint_names = get_joint_indices_franka(robot_id)
    J = len(joint_indices)

    # end-effector link index: Pandaでは "panda_hand" が末端に近い
    # 環境によってリンク番号が異なるので、名前から探す
    ee_link = None
    for j in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, j)[12].decode("utf-8")  # link name
        if name == "panda_hand":
            ee_link = j
            break
    if ee_link is None:
        # fallback: 最終関節
        ee_link = joint_indices[-1]

    # block
    bx, by = cfg.block_start_xy
    block_z = cfg.table_z + 0.65  # table height is ~0.62-0.65 in pybullet table.urdf
    # cube_small.urdf is about 0.05m
    block_id = p.loadURDF("cube_small.urdf", basePosition=[bx, by, block_z + 0.03])

    # apply episode-level physics
    set_body_mass(block_id, mass)
    set_body_friction(block_id, friction)
    set_floor_friction(plane_id, friction)

    # settle
    for _ in range(cfg.settle_steps):
        p.stepSimulation()

    return plane_id, table_id, robot_id, joint_indices, ee_link, block_id, J, rng


# =========================
# Scripted push policy (EE velocity -> IK position target)
# =========================

def push_trajectory_targets(cfg: CollectConfig):
    """
    T=64 内で
    1) ブロック上方へ移動
    2) 下降して接触高さへ
    3) x方向へ押す（goal側へ）
    を必ず含める設計（接触保証に寄与）
    """
    T = cfg.T
    # phases (sum to T)
    T1 = int(0.25 * T)  # approach
    T2 = int(0.20 * T)  # descend
    T3 = T - (T1 + T2)  # push

    bx, by = cfg.block_start_xy
    gx, gy = cfg.goal_xy

    # approach point
    p1 = np.array([bx - 0.10, by, cfg.approach_height], dtype=np.float32)
    # contact point (just behind block at contact height)
    p2 = np.array([bx - 0.03, by, cfg.contact_height], dtype=np.float32)
    # push end (toward goal)
    p3 = np.array([gx, gy, cfg.contact_height], dtype=np.float32)

    targets = []
    # linear interpolation helper
    def lerp(a, b, n):
        for i in range(n):
            t = (i + 1) / n
            targets.append((1 - t) * a + t * b)

    lerp(p1, p1, T1 // 2)      # hold
    lerp(p1, p2, T1 - T1 // 2) # move down behind block
    lerp(p2, p2, T2 // 2)      # hold
    lerp(p2, p2, T2 - T2 // 2) # keep (already at contact height)
    lerp(p2, p3, T3)           # push

    targets = np.stack(targets, axis=0).astype(np.float32)  # (T,3)
    assert targets.shape[0] == T
    return targets


# =========================
# Collect + Save NPZ
# =========================

def collect_episode(cfg: CollectConfig, mass: float, friction: float, ep_global_idx: int):
    ensure_dir(cfg.out_dir)

    plane_id, table_id, robot_id, joint_indices, ee_link, block_id, J, rng = setup_scene(
        cfg, mass, friction, seed=ep_global_idx
    )

    # camera: look at block area
    cam_target = [cfg.block_start_xy[0], cfg.block_start_xy[1], 0.7]
    cam_dist = 1.0
    cam_yaw = 90
    cam_pitch = -35

    T = cfg.T

    # allocate arrays (must match your fixed spec)
    rgb = np.zeros((T, cfg.img_h, cfg.img_w, 3), dtype=np.uint8)
    q = np.zeros((T, J), dtype=np.float32)
    dq = np.zeros((T, J), dtype=np.float32)
    f = np.zeros((T, J), dtype=np.float32)
    action = np.zeros((T, 3), dtype=np.float32)       # here A=3 (EE velocity-like)
    block_pose = np.zeros((T, 7), dtype=np.float32)

    targets = push_trajectory_targets(cfg)  # (T,3)

    dt = 1.0 / cfg.hz

    # initialize: move to first target
    q_ik = ik_to_joint_positions(robot_id, ee_link, targets[0])
    apply_joint_position_control(robot_id, joint_indices, q_ik, cfg)
    for _ in range(20):
        step_sim(cfg)

    # main loop
    prev_pos = targets[0].copy()
    for t in range(T):
        # obs
        rgb[t] = render_rgb(cfg, cam_target, cam_dist, cam_yaw, cam_pitch)
        q_t, dq_t, f_t = get_joint_states(robot_id, joint_indices)
        q[t], dq[t], f[t] = q_t, dq_t, f_t
        block_pose[t] = pose7_from_base(block_id)

        # action (EE "velocity-like" command in xyz)
        cur_pos = targets[t]
        vel = (cur_pos - prev_pos) / dt
        action[t] = vel.astype(np.float32)
        prev_pos = cur_pos

        # control via IK -> joint pos target
        q_ik = ik_to_joint_positions(robot_id, ee_link, cur_pos)
        apply_joint_position_control(robot_id, joint_indices, q_ik, cfg)

        # step
        step_sim(cfg)

    out_path = Path(cfg.out_dir) / f"ep_m{mass:.3f}_mu{friction:.3f}_{ep_global_idx:05d}.npz"
    np.savez_compressed(
        out_path,
        rgb=rgb,
        q=q,
        dq=dq,
        f=f,
        action=action,
        block_pose=block_pose,
        mass=np.array(mass, dtype=np.float32),
        friction=np.array(friction, dtype=np.float32),
    )

    p.disconnect()
    return str(out_path)


def main():
    cfg = CollectConfig(gui=False)

    idx = 0
    for m in cfg.masses:
        for mu in cfg.frictions:
            for _ in range(cfg.episodes_per_cell):
                path = collect_episode(cfg, float(m), float(mu), idx)
                print("saved:", path)
                idx += 1


if __name__ == "__main__":
    main()
