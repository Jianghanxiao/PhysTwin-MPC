import argparse
from dataclasses import replace
from pathlib import Path
import pickle
import time
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


def _load_camera_c2w_from_calibrate(path: Path, camera_idx: int) -> np.ndarray:
    if path.suffix.lower() != ".pkl":
        raise ValueError(f"calibrate file must be .pkl, got: {path.suffix}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    mats = np.asarray(data, dtype=np.float32)
    if mats.ndim == 2 and mats.shape == (4, 4):
        return mats
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"calibrate.pkl must store [N,4,4] or [4,4], got {mats.shape}")
    if not (0 <= camera_idx < mats.shape[0]):
        raise IndexError(f"camera_idx={camera_idx} out of range for calibrate with {mats.shape[0]} cameras")
    return mats[camera_idx]


def _load_camera_intrinsic(path: Path, camera_idx: int) -> np.ndarray:
    intrinsics = np.load(path)
    if intrinsics.ndim == 2 and intrinsics.shape == (3, 3):
        return intrinsics.astype(np.float64)
    if intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3):
        if not (0 <= camera_idx < intrinsics.shape[0]):
            raise IndexError(
                f"camera_idx={camera_idx} out of range for intrinsics with {intrinsics.shape[0]} cameras"
            )
        return intrinsics[camera_idx].astype(np.float64)
    raise ValueError(f"intrinsics.npy must be [3,3] or [N,3,3], got {intrinsics.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Root-level clean QQTT-style open-loop planner")
    parser.add_argument("--task", type=str, choices=["rope", "cloth", "bear"], default="cloth")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="phystwin_dataset_assets",
        help="Dataset root containing different_types/experiments/experiments_optimization",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default=None,
        help="Optional explicit case folder name (e.g. rope_0). If omitted, auto-selected from task keyword.",
    )
    parser.add_argument("--current-pcd", type=str, default="source_cloth/object.ply", help="Current object point cloud file")
    parser.add_argument("--target-pcd", type=str, default="target_cloth/object.ply", help="Target object point cloud file")
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
    parser.add_argument(
        "--video-calibrate-pkl",
        type=str,
        default="source_cloth/calibrate.pkl",
        help="Path to calibrate.pkl for rollout video camera extrinsic (required for camera-matched rendering)",
    )
    parser.add_argument(
        "--video-camera-idx",
        type=int,
        default=0,
        help="Camera index used with --video-calibrate-pkl",
    )
    parser.add_argument(
        "--video-intrinsics-npy",
        type=str,
        default="source_cloth/intrinsics.npy",
        help="Path to intrinsics.npy for rollout video camera intrinsic (required for camera-matched rendering)",
    )
    parser.add_argument(
        "--video-overlay-image",
        type=str,
        default="source_cloth/color.png",
        help="RGB image path for white-background compositing (default: source_cloth/color.png)",
    )
    parser.add_argument(
        "--mock-saved-pose",
        type=str,
        default="./cloth.npz",
        help="Path to saved pose file for MockRobot (expanded)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = build_plan_config(
        task=args.task,
        seed=args.seed,
        max_points=args.max_points,
        dataset_root=args.dataset_root,
        case_name=args.case_name,
    )
    config = replace(
        config,
        qqtt_dynamics=replace(config.qqtt_dynamics, output_dir=str(save_dir)),
    )
    print(f"[Dataset] root={args.dataset_root}, task={args.task}, case={config.qqtt_dynamics.case_name}")

    if args.robot == "mock":
        robot = MockRobot(saved_pose_path=args.mock_saved_pose)
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

    video_calibrate_path = Path(args.video_calibrate_pkl)
    if not video_calibrate_path.exists():
        raise FileNotFoundError(f"video calibrate file not found: {video_calibrate_path}")
    video_camera_c2w = _load_camera_c2w_from_calibrate(video_calibrate_path, args.video_camera_idx)

    video_intrinsics_path = Path(args.video_intrinsics_npy)
    if not video_intrinsics_path.exists():
        raise FileNotFoundError(f"video intrinsics file not found: {video_intrinsics_path}")
    video_camera_intrinsic = _load_camera_intrinsic(video_intrinsics_path, args.video_camera_idx)

    video_overlay_image = Path(args.video_overlay_image) if args.video_overlay_image else None
    if video_overlay_image is not None and not video_overlay_image.exists():
        raise FileNotFoundError(f"video overlay image not found: {video_overlay_image}")

    pipeline = OpenLoopPlanningPipeline(config=config, robot=robot)
    result = pipeline.run(
        current_pcd_path=Path(args.current_pcd),
        target_pcd_path=Path(args.target_pcd),
        execute=args.execute,
        save_dir=save_dir,
        video_camera_c2w=video_camera_c2w,
        video_camera_intrinsic=video_camera_intrinsic,
        video_overlay_image=video_overlay_image,
    )
    total_seconds = time.perf_counter() - t0

    print("Planning finished.")
    print(f"Best reward: {result['best_reward']:.6f}")
    print(f"Final chamfer: {result['final_chamfer']:.6f}")
    if "timing" in result:
        timing = result["timing"]
        print(
            "Timing (s): "
            f"total={timing.get('total_seconds', total_seconds):.3f}, "
            f"load_pcd={timing.get('load_pcd_seconds', 0.0):.3f}, "
            f"plan={timing.get('plan_seconds', 0.0):.3f}, "
            f"rollout={timing.get('rollout_seconds', 0.0):.3f}, "
            f"visualize={timing.get('visualize_seconds', 0.0):.3f}, "
            f"execute={timing.get('execute_seconds', 0.0):.3f}"
        )
    else:
        print(f"Timing (s): total={total_seconds:.3f}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
