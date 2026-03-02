#!/usr/bin/env python3
"""
Move the xArm7 to a pose previously saved by save_xarm_pose.py.

The .npz file is expected to contain:
  - xyz:     (3,)   world-frame position
  - rot:     (3,3)  world-frame rotation matrix
  - gripper: scalar gripper openness [0, 1]

Usage:
    python move_xarm_to_pose.py --pose rope.npz
    python move_xarm_to_pose.py --pose rope.npz --ip 192.168.1.196 --base2world base2world.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from phystwin_mpc.robot import ActionStep, XArm7Robot


def _load_base2world_matrix(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    mat = np.asarray(data, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"base2world.pkl must store a 4x4 matrix, got {mat.shape}")
    return mat


def _load_pose_npz(path: Path) -> ActionStep:
    data = np.load(path)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    rot = np.asarray(data["rot"], dtype=np.float32)
    gripper = float(data["gripper"])
    if xyz.shape != (3,):
        raise ValueError(f"xyz must be shape (3,), got {xyz.shape}")
    if rot.shape != (3, 3):
        raise ValueError(f"rot must be shape (3,3), got {rot.shape}")
    return ActionStep(xyz=xyz, rot=rot, gripper=gripper)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move xArm7 to a saved pose (.npz from save_xarm_pose.py)"
    )
    parser.add_argument(
        "--pose", type=str, required=True,
        help="Path to the .npz file containing xyz, rot, gripper",
    )
    parser.add_argument("--ip", type=str, default="192.168.1.196", help="xArm7 IP address")
    parser.add_argument("--speed", type=float, default=100.0, help="Movement speed")
    parser.add_argument("--acc", type=float, default=2000.0, help="Movement acceleration")
    parser.add_argument("--gripper-enable", action="store_true", help="Enable gripper control")
    parser.add_argument(
        "--tool-extension-mm", type=float, default=65.0,
        help="Tool extension from xArm flange to end-effector tip in mm",
    )
    parser.add_argument(
        "--base2world", type=str, default="base2world.pkl",
        help="Path to base2world.pkl (4x4 matrix)",
    )
    args = parser.parse_args()

    # Load target pose
    pose_path = Path(args.pose)
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    target = _load_pose_npz(pose_path)

    print(f"Target pose from {pose_path}:")
    print(f"  xyz:     {target.xyz}")
    print(f"  rot:\n{target.rot}")
    print(f"  gripper: {target.gripper}")

    # Load base2world
    base2world = None
    if args.base2world:
        b2w_path = Path(args.base2world)
        if not b2w_path.exists():
            raise FileNotFoundError(f"base2world file not found: {b2w_path}")
        base2world = _load_base2world_matrix(b2w_path)

    # Connect and move
    robot = XArm7Robot(
        ip=args.ip,
        speed=args.speed,
        acc=args.acc,
        gripper_enable=args.gripper_enable,
        base_to_world=base2world,
        tool_extension_m=args.tool_extension_mm / 1000.0,
    )

    current = robot.get_current_pose()
    dist = float(np.linalg.norm(target.xyz - current.xyz))
    print(f"Current xyz: {current.xyz}")
    print(f"Distance to target: {dist:.4f} m")

    robot.execute_action_sequence([target])
    print("Done — xArm7 moved to target pose.")


if __name__ == "__main__":
    main()
