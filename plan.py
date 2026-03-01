import argparse
from pathlib import Path
import pickle
import numpy as np

from phystwin_mpc.config import build_plan_config
from phystwin_mpc.pipeline import OpenLoopPlanningPipeline
from phystwin_mpc.robot import MockRobot, XArm7Robot


def _load_base2world_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".pkl":
        raise ValueError(f"base2world file must be .pkl, got: {path.suffix}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    mat = np.asarray(data, dtype=np.float32)

    if mat.shape != (4, 4):
        raise ValueError(
            f"base2world.pkl must store a raw 4x4 matrix, got {mat.shape} from {path}"
        )
    return mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Root-level clean QQTT-style open-loop planner")
    parser.add_argument("--task", type=str, choices=["rope", "cloth"], default="rope")
    parser.add_argument("--current-pcd", type=str, default="source_rope/object.ply", help="Current object point cloud file")
    parser.add_argument("--target-pcd", type=str, default="target_rope/object.ply", help="Target object point cloud file")
    parser.add_argument("--robot", type=str, choices=["mock", "xarm7"], default="mock")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true", help="Execute full open-loop sequence on robot")
    parser.add_argument("--save-dir", type=str, default="outputs/plan", help="Directory to save planning results")
    parser.add_argument("--max-points", type=int, default=1000)
    parser.add_argument(
        "--base2world",
        type=str,
        default="base2world.pkl",
        help="Path to base2world.pkl storing a raw 4x4 base-to-world matrix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = build_plan_config(task=args.task, seed=args.seed, max_points=args.max_points)

    if args.robot == "mock":
        robot = MockRobot()
    elif args.robot == "xarm7":
        base2world = None
        if args.base2world:
            base2world = _load_base2world_matrix(Path(args.base2world))
        robot = XArm7Robot(
            ip=args.xarm_ip,
            speed=args.xarm_speed,
            acc=args.xarm_acc,
            gripper_enable=args.xarm_gripper_enable,
            base_to_world=base2world,
            tool_extension_m=args.xarm_tool_extension_mm / 1000.0,
        )
    else:
        raise ValueError(f"Unsupported robot type: {args.robot}")

    pipeline = OpenLoopPlanningPipeline(config=config, robot=robot)
    result = pipeline.run(
        current_pcd_path=Path(args.current_pcd),
        target_pcd_path=Path(args.target_pcd),
        execute=args.execute,
        save_dir=save_dir,
    )

    print("Planning finished.")
    print(f"Best reward: {result['best_reward']:.6f}")
    print(f"Final chamfer: {result['final_chamfer']:.6f}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
