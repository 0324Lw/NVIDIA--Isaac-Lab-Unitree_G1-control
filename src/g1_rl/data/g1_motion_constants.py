from __future__ import annotations

G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "xl330_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "d455_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]

SENSOR_JOINT_NAMES = ["xl330_joint", "d455_joint"]

CONTROLLED_JOINT_NAMES = [
    name for name in G1_JOINT_NAMES if name not in SENSOR_JOINT_NAMES
]

JOINT_ID = {name: i for i, name in enumerate(G1_JOINT_NAMES)}

TASK1_REQUIRED_KEYS = [
    "pos",
    "vel",
    "num_frames",
    "joint_names",
    "controlled_joint_names",
    "sensor_joint_names",
    "fps",
    "dt",
    "phase",
    "contact_ref",
]

TASK2_REQUIRED_KEYS = [
    "pos",
    "vel",
    "cmd",
    "num_frames",
    "joint_names",
    "controlled_joint_names",
    "sensor_joint_names",
    "fps",
    "dt",
    "phase",
    "contact_ref",
    "mode_id",
    "mode_names",
]
