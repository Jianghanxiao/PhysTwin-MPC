from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from phystwin_mpc.robot import ActionStep, XArm7Robot


def _load_base2world_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".pkl":
        raise ValueError(f"base2world file must be .pkl, got: {path.suffix}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    mat = np.asarray(data, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"base2world.pkl must store a raw 4x4 matrix, got {mat.shape}")
    return mat


def _sequence_from_npy(path: Path) -> list[ActionStep]:
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 13:
        raise ValueError(
            f"best action npy must be shape [T,13] (xyz + rot(9) + gripper), got {arr.shape}"
        )

    sequence: list[ActionStep] = []
    for i in range(arr.shape[0]):
        row = arr[i]
        sequence.append(
            ActionStep(
                xyz=row[:3].astype(np.float32).copy(),
                rot=row[3:12].reshape(3, 3).astype(np.float32).copy(),
                gripper=float(row[12]),
            )
        )
    return sequence


def _sequence_from_json(path: Path) -> list[ActionStep]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("best action json must be a list of steps")

    sequence: list[ActionStep] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"json step {i} must be a dict")
        xyz = np.asarray(item.get("xyz"), dtype=np.float32)
        rot = np.asarray(item.get("rot"), dtype=np.float32)
        gripper = float(item.get("gripper", 0.0))
        if xyz.shape != (3,):
            raise ValueError(f"json step {i} xyz must be shape (3,), got {xyz.shape}")
        if rot.shape != (3, 3):
            raise ValueError(f"json step {i} rot must be shape (3,3), got {rot.shape}")
        sequence.append(ActionStep(xyz=xyz.copy(), rot=rot.copy(), gripper=gripper))
    return sequence


def _load_sequence(path: Path) -> list[ActionStep]:
    if not path.exists():
        raise FileNotFoundError(f"action file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _sequence_from_npy(path)
    if suffix == ".json":
        return _sequence_from_json(path)
    raise ValueError(f"unsupported action file type: {suffix}, expected .npy or .json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved best action sequence on xArm7")
    parser.add_argument(
        "--action-file",
        type=str,
        default="results_rope/plan/best_action_sequence.npy",
        help="Path to best action sequence (.npy or .json)",
    )
    parser.add_argument("--xarm-ip", type=str, default="192.168.1.196")
    parser.add_argument("--xarm-speed", type=float, default=100.0)
    parser.add_argument("--xarm-acc", type=float, default=2000.0)
    parser.add_argument("--xarm-gripper-enable", action="store_true")
    parser.add_argument(
        "--xarm-tool-extension-mm",
        type=float,
        default=65.0,
        help="Additional tool length from xArm flange/TCP to real end-effector tip in mm",
    )
    parser.add_argument(
        "--base2world",
        type=str,
        default="base2world.pkl",
        help="Path to base2world.pkl storing a raw 4x4 base-to-world matrix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action_file = Path(args.action_file)
    sequence = _load_sequence(action_file)
    if len(sequence) == 0:
        raise ValueError("loaded action sequence is empty")

    base2world = None
    if args.base2world:
        base2world_path = Path(args.base2world)
        if not base2world_path.exists():
            raise FileNotFoundError(f"base2world file not found: {base2world_path}")
        base2world = _load_base2world_matrix(base2world_path)

    robot = XArm7Robot(
        ip=args.xarm_ip,
        speed=args.xarm_speed,
        acc=args.xarm_acc,
        gripper_enable=args.xarm_gripper_enable,
        base_to_world=base2world,
        tool_extension_m=args.xarm_tool_extension_mm / 1000.0,
    )

    print(f"Loaded {len(sequence)} steps from: {action_file}")
    print(
        f"Replaying on xArm7(ip={args.xarm_ip}, speed={args.xarm_speed}, acc={args.xarm_acc}, "
        f"gripper_enable={args.xarm_gripper_enable})"
    )
    robot.execute_action_sequence(sequence)
    print("Replay finished.")


if __name__ == "__main__":
    main()
