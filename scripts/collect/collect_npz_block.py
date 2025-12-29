# -*- coding: utf-8 -*-
"""Converted from a Jupyter notebook.

Usage:
  python scripts/collect/collect_npz_block.py

Output:
  datasets/raw/episode_XXXXXX_m=..._mu=....npz

Notes:
- This script runs headless (p.DIRECT).
"""

import os
import math
import numpy as np
import pybullet as p
import pybullet_data
import imageio.v2 as imageio  # ← ループ外でimport

# =========================
# Config
# =========================
EPISODE_ID = 123
SAVE_DIR = "datasets/raw"

BLOCK_MASS = 1.0
BLOCK_FRICTION = 0.5

TIME_STEPS = 800
DELTA_TIME = 1.0 / 240.0

# action = [push_speed, push_force]
PUSH_SPEED = 0.0008
PUSH_FORCE = 800

# まずはデバッグ用（renderと同等）
W, H = 360, 240

# =========================
# PyBullet Init
# =========================
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(DELTA_TIME)

# =========================
# Environment
# =========================
planeId = p.loadURDF("plane.urdf")

# render側と同じ：URDFテーブルを使う（床より上に配置）
tableId = p.loadURDF(
    "table/table.urdf",
    basePosition=[0.5, 0, 0.0],
    useFixedBase=True
)
print("tableId:", tableId, "AABB:", p.getAABB(tableId))

# =========================
# Robot (KUKA iiwa)
# =========================
robotId = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True
)

num_joints = p.getNumJoints(robotId)
ee_link = 6

initial_q = [0, 0.3, 0, -1.2, 0, 1.0, 0.5]
for i in range(num_joints):
    p.resetJointState(robotId, i, initial_q[i])

# =========================
# Block (white)
# =========================
block_half_extents = [0.05, 0.05, 0.02]

block_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=block_half_extents)
block_vis = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=block_half_extents,
    rgbaColor=[1, 1, 1, 1]
)

blockId = p.createMultiBody(
    baseMass=BLOCK_MASS,
    baseCollisionShapeIndex=block_col,
    baseVisualShapeIndex=block_vis,
    basePosition=[0.6, 0.0, 0.65]  # render側に寄せる（テーブル上を想定）
)

p.changeDynamics(
    blockId,
    -1,
    lateralFriction=BLOCK_FRICTION,
    angularDamping=0.95,
    linearDamping=0.05
)

# =========================
# Camera (renderと同じ)
# =========================
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[1.3, 0.6, 1.2],
    cameraTargetPosition=[0.60, 0.00, 0.45],
    cameraUpVector=[0, 0, 1]
)

proj_matrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=W / H,
    nearVal=0.1,
    farVal=5.0
)

def capture_rgb():
    img = p.getCameraImage(
        W, H,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
    )
    rgb = np.reshape(img[2], (H, W, 4))[:, :, :3]
    return rgb.astype(np.uint8)

# =========================
# Log Buffers
# =========================
rgb_buf = []
q_buf = []
dq_buf = []
f_buf = []
action_buf = []
block_pose_buf = []

# =========================
# Simulation Loop
# =========================
x = 0.55
y = 0.0
z = 0.68

for t in range(TIME_STEPS):
    x += PUSH_SPEED
    target_pos = [x, y, z]
    target_ori = p.getQuaternionFromEuler([0, math.pi / 2, 0])

    joint_poses = p.calculateInverseKinematics(robotId, ee_link, target_pos, target_ori)

    for j in range(num_joints):
        p.setJointMotorControl2(
            robotId,
            j,
            p.POSITION_CONTROL,
            joint_poses[j],
            force=PUSH_FORCE
        )

    p.stepSimulation()

    # --- Logging ---
    rgb = capture_rgb()
    rgb_buf.append(rgb)

    # 最初の1枚だけデバッグ保存
    if t == 0:
        imageio.imwrite("debug_frame0.png", rgb)
        print("saved debug_frame0.png")

    q, dq, ff = [], [], []
    joint_states = p.getJointStates(robotId, range(num_joints))
    for js in joint_states:
        q.append(js[0])
        dq.append(js[1])
        ff.append(js[3])  # joint reaction forces/torques

    q_buf.append(q)
    dq_buf.append(dq)
    f_buf.append(ff)

    action_buf.append([PUSH_SPEED, PUSH_FORCE])

    pos, ori = p.getBasePositionAndOrientation(blockId)
    block_pose_buf.append(list(pos) + list(ori))

# =========================
# Save Episode (.npz)
# =========================
os.makedirs(SAVE_DIR, exist_ok=True)

filename = (
    f"episode_{EPISODE_ID:06d}"
    f"_m={BLOCK_MASS:.2f}"
    f"_mu={BLOCK_FRICTION:.2f}.npz"
)
path = os.path.join(SAVE_DIR, filename)

np.savez_compressed(
    path,
    rgb=np.array(rgb_buf),
    q=np.array(q_buf, dtype=np.float32),
    dq=np.array(dq_buf, dtype=np.float32),
    f=np.array(f_buf, dtype=np.float32),
    action=np.array(action_buf, dtype=np.float32),
    block_pose=np.array(block_pose_buf, dtype=np.float32),
    mass=float(BLOCK_MASS),
    friction=float(BLOCK_FRICTION)
)

print(f"✅ Saved episode: {path}")
p.disconnect()
