#!/usr/bin/env python3
"""
Convert segmentation mask to point cloud.
Uses depth array + intrinsics + binary mask to generate 3D point cloud.
Applies statistical outlier removal for noise filtering.
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d


DEFAULT_IO_DIR = "target_rope_boba"


def get_pcd_from_depth(depth_m: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Lift depth map to camera-frame 3D points using K^{-1} (same as proj-QQTT)."""
    height, width = depth_m.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth_flat = depth_m.reshape(-1)

    points = np.stack([x, y, np.ones_like(x)], axis=1)
    points = points * depth_flat[:, None]
    points = points @ np.linalg.inv(intrinsic).T
    points = points.reshape(height, width, 3)
    return points


def mask_to_pointcloud(
    depth_img: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    filter_outliers: bool = True,
    std_ratio: float = 2.5,
    nb_neighbors: int = 30,
    voxel_size: Optional[float] = None,
) -> o3d.geometry.PointCloud:
    """
    Convert segmentation mask to 3D point cloud.
    
    Args:
        depth_img: Depth image (H, W) in millimeters
        mask: Binary segmentation mask (H, W) with values 0 or 1
        intrinsics: Camera intrinsics matrix (3, 3)
        extrinsics: Camera-to-world transformation matrix (4, 4). If provided, transforms points to world frame.
        filter_outliers: Apply statistical outlier removal
        std_ratio: Standard deviation ratio for outlier filtering
        nb_neighbors: Number of neighbors for outlier removal
        voxel_size: Optional voxel downsampling size in meters
    
    Returns:
        o3d.geometry.PointCloud: Filtered point cloud in camera or world frame
    """
    depth_m = depth_img.astype(np.float32) / 1000.0
    points_cam = get_pcd_from_depth(depth_m, intrinsics)

    if mask.dtype != np.bool_:
        mask = mask > 0

    valid_depth = np.logical_and(depth_m > 0.2, depth_m < 1.5)
    valid_depth = np.logical_and(valid_depth, np.isfinite(depth_m))
    valid_mask = np.logical_and(mask, valid_depth)

    points = points_cam[valid_mask]
    
    if len(points) == 0:
        print("[WARNING] No valid points after filtering. Returning empty point cloud.")
        return o3d.geometry.PointCloud()
    
    print(f"[INFO] Generated point cloud with {len(points)} points in camera frame")
    
    # Transform to world frame if extrinsics provided
    if extrinsics is not None:
        c2w = extrinsics
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        points = (c2w @ homogeneous_points.T).T[:, :3]
        print(f"[INFO] Transformed points to world frame using extrinsics")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Optional: Apply voxel downsampling
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"[INFO] After voxel downsampling ({voxel_size}m): {len(pcd.points)} points")
    
    # Apply statistical outlier removal
    if filter_outliers:
        print(f"[INFO] Applying statistical outlier removal (nb={nb_neighbors}, std={std_ratio})...")
        pcd_filtered = pcd
        iteration = 0
        while True:
            _, inlier_idx = pcd_filtered.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio + iteration * 0.5,
            )
            pcd_filtered = pcd_filtered.select_by_index(inlier_idx)
            iteration += 1
            
            print(f"  - Iteration {iteration}: {len(pcd_filtered.points)} points")
            
            # Stop if not many outliers are removed
            if iteration > 0 and len(pcd_filtered.points) > len(pcd.points) * 0.5:
                break
            if iteration > 5:  # Max iterations
                break
        
        pcd = pcd_filtered
    
    print(f"[INFO] Final point cloud: {len(pcd.points)} points")
    return pcd


def main():
    """
    Main entry point: process directory of captures or individual files.
    """
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(
        description="Convert segmentation mask to point cloud"
    )
    parser.add_argument(
        "--depth_path", type=str, default=f"{DEFAULT_IO_DIR}/depth.npy", help="Path to depth array (NPY) in mm"
    )
    parser.add_argument(
        "--mask_path", type=str, default=f"{DEFAULT_IO_DIR}/mask.png", help="Path to binary mask image (PNG)"
    )
    parser.add_argument(
        "--intrinsics_path", type=str, default=f"{DEFAULT_IO_DIR}/intrinsics.npy", help="Path to intrinsics (NPY file)"
    )
    parser.add_argument(
        "--output_path", type=str, default=f"{DEFAULT_IO_DIR}/object.ply", help="Path to save point cloud (PLY)"
    )
    parser.add_argument(
        "--calibrate_pkl", type=str, default=f"{DEFAULT_IO_DIR}/calibrate.pkl", help="Path to calibrate.pkl (for extrinsics)"
    )
    parser.add_argument(
        "--camera_idx", type=int, default=0, help="Camera index to use from calibrate.pkl"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.01, help="Voxel size for downsampling (m)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize point cloud"
    )
    
    args = parser.parse_args()
    
    # Load depth array
    print(f"[INFO] Loading depth array: {args.depth_path}")
    try:
        depth_img = np.load(args.depth_path)
    except Exception as e:
        print(f"[ERROR] Failed to load depth array: {args.depth_path}")
        print(f"  - {e}")
        sys.exit(1)
    
    # Load mask image
    print(f"[INFO] Loading mask image: {args.mask_path}")
    mask_img = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"[ERROR] Failed to load mask image: {args.mask_path}")
        sys.exit(1)
    
    # Convert mask to boolean
    mask_img = mask_img > 0
    
    # Load intrinsics
    print(f"[INFO] Loading intrinsics: {args.intrinsics_path}")
    intrinsics = np.load(args.intrinsics_path)
    # If intrinsics is (N, 3, 3), select camera_idx
    if intrinsics.ndim == 3:
        if args.camera_idx >= intrinsics.shape[0]:
            print(
                f"[ERROR] camera_idx={args.camera_idx} out of range for intrinsics with {intrinsics.shape[0]} cameras"
            )
            sys.exit(1)
        intrinsics = intrinsics[args.camera_idx]
    print(f"  - Intrinsics shape: {intrinsics.shape}")
    
    # Load extrinsics from calibrate.pkl
    extrinsics = None
    if Path(args.calibrate_pkl).exists():
        print(f"[INFO] Loading extrinsics from: {args.calibrate_pkl}")
        with open(args.calibrate_pkl, "rb") as f:
            c2ws = pickle.load(f)
        if isinstance(c2ws, list) and len(c2ws) > args.camera_idx:
            extrinsics = c2ws[args.camera_idx]
            print(f"  - Using camera index {args.camera_idx}")
            print(f"  - Extrinsics shape: {extrinsics.shape}")
        else:
            print(f"[WARNING] Camera index {args.camera_idx} not found in calibrate.pkl")
    else:
        print(f"[WARNING] calibrate.pkl not found at {args.calibrate_pkl}. Points will be in camera frame.")
    
    # Convert mask to point cloud with world transformation if extrinsics available
    print("[INFO] Converting mask to point cloud...")
    pcd = mask_to_pointcloud(
        depth_img=depth_img,
        mask=mask_img,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        voxel_size=args.voxel_size,
    )
    
    # Save point cloud
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"[INFO] Saved point cloud: {output_path}")
    
    # Visualize if requested
    if args.visualize:
        print("[INFO] Visualizing point cloud...")
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
