#!/usr/bin/env python3
"""
Simple script to read current pose from a connected xArm7 and save it to disk
so the MockRobot can load it later.

The pose is saved in **world coordinates** (after applying the base2world
transform), which is the same frame that ``XArm7Robot.get_current_pose``
returns when ``base_to_world`` is set.  MockRobot loads the file and returns
the pose as-is — no extra transforms needed.

Usage:
    python save_xarm_pose.py --ip 192.168.1.196 --base2world base2world.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import numpy as np
import os
from phystwin_mpc.robot import XArm7Robot

DEFAULT_OUT = "./rope.npz"


def _load_base2world_matrix(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    mat = np.asarray(data, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"base2world.pkl must store a 4x4 matrix, got {mat.shape}")
    return mat


def main() -> None:
    parser = argparse.ArgumentParser(description="Save current xArm7 pose to disk for MockRobot")
    parser.add_argument("--ip", default="192.168.1.196", help="xArm7 IP address")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output .npz path")
    parser.add_argument("--base2world", type=str, default="base2world.pkl",
                        help="Path to base2world.pkl (must match what plan.py uses)")
    parser.add_argument("--tool-extension-mm", type=float, default=65.0, help="Tool extension in mm")
    parser.add_argument("--gripper-enable", action="store_true", help="Read gripper state")
    args = parser.parse_args()

    base2world = None
    if args.base2world:
        base2world = _load_base2world_matrix(Path(args.base2world))
    else:
        raise ValueError("base2world argument is required to ensure correct world-frame pose")

    xr = XArm7Robot(
        ip=args.ip,
        tool_extension_m=args.tool_extension_mm / 1000.0,
        gripper_enable=args.gripper_enable,
        base_to_world=base2world,
    )
    pose = xr.get_current_pose()  # already in world coordinates

    out_path = os.path.expanduser(args.out)
    np.savez(
        out_path,
        xyz=pose.xyz.astype(np.float32),
        rot=pose.rot.astype(np.float32),
        gripper=float(pose.gripper),
    )
    print(f"Saved xArm7 world-frame pose to {out_path}")


if __name__ == "__main__":
    main()
