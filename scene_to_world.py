#!/usr/bin/env python3
"""
Project full scene depth to world coordinates and visualize with Open3D.

Expected saved files (default):
- outputs/depth.npy        (depth image, usually in millimeters)
- outputs/color.png        (optional, for coloring point cloud)
- outputs/intrinsics.npy   (3x3 or Nx3x3)
- outputs/calibrate.pkl    (list of 4x4 camera-to-world transforms)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d


def depth_to_points_cam(depth_m: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Lift depth map to camera-frame 3D points, output shape (H, W, 3)."""
    height, width = depth_m.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pix = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32)

    k_inv = np.linalg.inv(intrinsic).astype(np.float32)
    pts = pix.reshape(-1, 3) @ k_inv.T
    pts = pts.reshape(height, width, 3)
    pts *= depth_m[..., None]
    return pts


def load_intrinsics(intrinsics_path: Path, camera_idx: int) -> np.ndarray:
    intrinsics = np.load(intrinsics_path)
    if intrinsics.ndim == 2:
        if intrinsics.shape != (3, 3):
            raise ValueError(f"Invalid intrinsics shape: {intrinsics.shape}, expected (3,3)")
        return intrinsics.astype(np.float32)

    if intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3):
        if not (0 <= camera_idx < intrinsics.shape[0]):
            raise IndexError(
                f"camera_idx={camera_idx} out of range for intrinsics with {intrinsics.shape[0]} cameras"
            )
        return intrinsics[camera_idx].astype(np.float32)

    raise ValueError(
        f"Invalid intrinsics shape: {intrinsics.shape}, expected (3,3) or (N,3,3)"
    )


def load_c2w(calibrate_pkl: Path, camera_idx: int) -> Optional[np.ndarray]:
    if not calibrate_pkl.exists():
        return None

    with open(calibrate_pkl, "rb") as f:
        c2ws = pickle.load(f)

    if isinstance(c2ws, list):
        if not (0 <= camera_idx < len(c2ws)):
            raise IndexError(
                f"camera_idx={camera_idx} out of range for calibrate.pkl with {len(c2ws)} cameras"
            )
        c2w = np.asarray(c2ws[camera_idx], dtype=np.float32)
    else:
        c2ws_arr = np.asarray(c2ws)
        if c2ws_arr.shape == (4, 4):
            c2w = c2ws_arr.astype(np.float32)
        elif c2ws_arr.ndim == 3 and c2ws_arr.shape[1:] == (4, 4):
            if not (0 <= camera_idx < c2ws_arr.shape[0]):
                raise IndexError(
                    f"camera_idx={camera_idx} out of range for c2w array with {c2ws_arr.shape[0]} cameras"
                )
            c2w = c2ws_arr[camera_idx].astype(np.float32)
        else:
            raise ValueError(
                f"Invalid c2w format in calibrate.pkl: got shape {c2ws_arr.shape}, expected (4,4) or (N,4,4)"
            )

    if c2w.shape != (4, 4):
        raise ValueError(f"Invalid c2w shape: {c2w.shape}, expected (4,4)")

    return c2w


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project full scene depth to world coordinate and visualize with Open3D"
    )
    parser.add_argument("--depth_path", type=str, default="outputs/depth.npy", help="Depth NPY path")
    parser.add_argument("--color_path", type=str, default="outputs/color.png", help="Color image path (optional)")
    parser.add_argument(
        "--intrinsics_path", type=str, default="outputs/intrinsics.npy", help="Intrinsics NPY path"
    )
    parser.add_argument(
        "--calibrate_pkl",
        type=str,
        default="outputs/calibrate.pkl",
        help="Calibration PKL path storing camera-to-world transforms",
    )
    parser.add_argument("--camera_idx", type=int, default=0, help="Camera index to use")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="Depth scale to meters (1000 for mm-depth, 1 for meters)",
    )
    parser.add_argument("--min_depth", type=float, default=0.15, help="Min valid depth in meters")
    parser.add_argument("--max_depth", type=float, default=2.0, help="Max valid depth in meters")
    parser.add_argument("--sample_step", type=int, default=1, help="Keep every Nth valid point")
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.005,
        help="Voxel downsample size in meters (<=0 to disable)",
    )
    parser.add_argument(
        "--axis_size",
        type=float,
        default=0.2,
        help="Open3D world coordinate frame size in meters",
    )
    parser.add_argument("--save_ply", type=str, default="", help="Optional output PLY path")
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Skip Open3D visualization and only save/print stats",
    )
    args = parser.parse_args()

    depth_path = Path(args.depth_path)
    color_path = Path(args.color_path)
    intrinsics_path = Path(args.intrinsics_path)
    calibrate_pkl = Path(args.calibrate_pkl)

    if not depth_path.exists():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")

    depth = np.load(depth_path)
    if depth.ndim == 3:
        if not (0 <= args.camera_idx < depth.shape[0]):
            raise IndexError(
                f"camera_idx={args.camera_idx} out of range for depth with {depth.shape[0]} cameras"
            )
        depth = depth[args.camera_idx]
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D (H,W), got shape {depth.shape}")

    depth_m = depth.astype(np.float32) / args.depth_scale
    intrinsic = load_intrinsics(intrinsics_path, args.camera_idx)
    c2w = load_c2w(calibrate_pkl, args.camera_idx)

    points_cam = depth_to_points_cam(depth_m, intrinsic)

    valid = np.isfinite(depth_m)
    valid &= depth_m > args.min_depth
    valid &= depth_m < args.max_depth

    points = points_cam[valid]

    colors = None
    if color_path.exists():
        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_bgr is not None:
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if color_rgb.shape[:2] == depth.shape:
                colors = color_rgb[valid]
            else:
                print(
                    f"[WARN] Color/depth shape mismatch: color={color_rgb.shape[:2]}, depth={depth.shape}. Ignore color."
                )
        else:
            print(f"[WARN] Failed to read color image: {color_path}. Ignore color.")

    if args.sample_step > 1:
        points = points[:: args.sample_step]
        if colors is not None:
            colors = colors[:: args.sample_step]

    if len(points) == 0:
        raise RuntimeError("No valid points after filtering. Check depth scale/range.")

    if c2w is not None:
        homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        points_world = (c2w @ homo.T).T[:, :3]
        frame_name = "world"
    else:
        points_world = points
        frame_name = "camera (calibrate.pkl not found)"
        print("[WARN] calibrate.pkl not found, points stay in camera frame.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world.astype(np.float64))
    if colors is not None and len(colors) == len(points_world):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)

    print(f"[INFO] Frame: {frame_name}")
    print(f"[INFO] Points after filtering/downsampling: {len(pcd.points)}")

    if args.save_ply:
        out_path = Path(args.save_ply)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"[INFO] Saved point cloud: {out_path}")

    if not args.no_visualize:
        world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, world_axis], window_name="Scene in World Coordinate")


if __name__ == "__main__":
    main()
