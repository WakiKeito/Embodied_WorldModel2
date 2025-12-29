# -*- coding: utf-8 -*-
"""Converted from a Jupyter notebook.

Usage:
  python <this_file>.py
  例 python scripts/render/render_kuka_push_block.py

  出力例 kuka_push_block.mp4


Notes:
- Install dependencies first (examples):
    pip install pybullet moviepy==1.0.3 imageio numpy
- This script runs headless (p.DIRECT).
"""


import time
import math
import pybullet as p
import pybullet_data
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from base64 import b64encode
import numpy as np
import uuid, shutil
from typing import List

WIDTH = 360
HEIGHT = 240
DELTA_TIME = 1. / 240

# -------------------------
# 動画保存用関数
# -------------------------
def save_video(frames, path, fps=30):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(path, codec="libx264")

def play_mp4(path):
    # Notebook helper removed for .py usage.
    print('[INFO] Video saved:', path)
# -------------------------
# PyBullet 初期化
# -------------------------
p.connect(p.DIRECT)  # ← 動画取得OK
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# -------------------------
# 環境
# -------------------------
planeId = p.loadURDF("plane.urdf")

tableId = p.loadURDF(
    "table/table.urdf",
    basePosition=[0.5, 0, 0.0],   # ← -0.65 を 0.0 に
    useFixedBase=True
)

print("tableId:", tableId, "AABB:", p.getAABB(tableId))


# -------------------------
# アーム（KUKA iiwa）
# -------------------------
robotId = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True
)

num_joints = p.getNumJoints(robotId)
ee_link_index = 6
# -------------------------
# ブロック
# -------------------------
# ブロックサイズ（半分サイズ指定）
block_half_extents = [0.04, 0.04, 0.04]  # 8cm角の立方体

collision = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=block_half_extents
)

visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=block_half_extents,
    rgbaColor=[0, 0, 1, 1]  # 青色ブロック
)

blockId = p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=collision,
    baseVisualShapeIndex=visual,
    basePosition=[0.6, 0, 0.65]
)

p.changeDynamics(
    blockId,
    -1,
    lateralFriction=0.8,
    spinningFriction=0.01,
    rollingFriction=0.0
)

'''
# -------------------------
# ボール
# -------------------------
ball_radius = 0.04

collision = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
visual = p.createVisualShape(
    p.GEOM_SPHERE,
    radius=ball_radius,
    rgbaColor=[1, 0, 0, 1]
)

ballId = p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=collision,
    baseVisualShapeIndex=visual,
    basePosition=[0.6, 0, 0.65]
)

p.changeDynamics(ballId, -1, lateralFriction=0.8, rollingFriction=0.01)
'''
# -------------------------
# 初期姿勢
# -------------------------
initial_joint_positions = [0, 0.3, 0, -1.2, 0, 1.0, 0.5]
for i in range(num_joints):
    p.resetJointState(robotId, i, initial_joint_positions[i])

# -------------------------
# カメラ設定
# -------------------------
#WIDTH, HEIGHT = 640, 480
WIDTH, HEIGHT = 360, 240
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[1.3, 0.6, 1.2],      # 少し引いて斜め上から
    cameraTargetPosition=[0.60, 0.00, 0.45], # ロボット＋ブロックの中間を狙う
    cameraUpVector=[0, 0, 1]
)

proj_matrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=WIDTH / HEIGHT,
    nearVal=0.1,
    farVal=5.0   # 3.0 → 5.0
)

frames = []

def capture_frame():
    img = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER
    )

    rgb = np.reshape(img[2], (HEIGHT, WIDTH, 4))[:, :, :3]
    frames.append(rgb.astype(np.uint8))

# -------------------------
# 押す動作
# -------------------------
target_positions = [
    [0.55, 0.0, 0.7],
    [0.65, 0.0, 0.7],
]

for target_pos in target_positions:
    target_ori = p.getQuaternionFromEuler([0, math.pi / 2, 0])

    joint_poses = p.calculateInverseKinematics(
        robotId,
        ee_link_index,
        target_pos,
        target_ori
    )

    for _ in range(200):
        for i in range(num_joints):
            p.setJointMotorControl2(
                robotId,
                i,
                p.POSITION_CONTROL,
                joint_poses[i],
                force=500
            )
        p.stepSimulation()
        capture_frame()

# -------------------------
# 余韻
# -------------------------
for _ in range(300):
    p.stepSimulation()
    capture_frame()

# -------------------------
# 動画保存
# -------------------------
import os  # 先頭に追加（未importなら）

os.makedirs("outputs/videos", exist_ok=True)
out_path = "outputs/videos/kuka_push_block.mp4"
save_video(frames, out_path, fps=30)
print(f"🎥 動画保存完了: {out_path}")
play_mp4(out_path)

#p.disconnect()

