#!/usr/bin/env python3
"""
Capture a single observation from all cameras and save RGB + Depth data.
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from phystwin_mpc.qqtt.env.camera import CameraSystem


def make_dirs(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    """
    Capture observation from all cameras and save to disk.
    """
    output_dir = Path("outputs")
    make_dirs(output_dir)
    
    print(f"[INFO] Output directory: {output_dir}")
    
    print("[INFO] Initializing camera system...")
    camera_system = CameraSystem(num_cam=1)
    
    print("[INFO] Capturing observation...")
    obs = camera_system.get_observation()
    
    # Single-camera save convention
    cam_idx = 0
    if cam_idx not in obs:
        print("[ERROR] Camera 0 observation not found.")
        camera_system.realsense.stop()
        sys.exit(1)

    cam_data = obs[cam_idx]

    rgb_path = output_dir / "color.png"
    cv2.imwrite(str(rgb_path), cam_data["color"])
    print(f"  - Saved RGB:   {rgb_path}")

    depth = cam_data["depth"]
    depth_path = output_dir / "depth.npy"
    np.save(depth_path, depth)
    print(f"  - Saved Depth: {depth_path}")
    
    # Save metadata as JSON
    metadata = {
        "num_cam": camera_system.num_cam,
        "saved_files": ["color.png", "depth.npy", "intrinsics.npy"],
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  - Saved Metadata: {metadata_path}")
    
    # Get and save intrinsics
    intrinsics = camera_system.realsense.get_intrinsics()
    intrinsics_path = output_dir / "intrinsics.npy"
    np.save(intrinsics_path, intrinsics)
    print(f"  - Saved Intrinsics: {intrinsics_path}")
    
    camera_system.realsense.stop()
    print("[INFO] Observation capture completed.")


if __name__ == "__main__":
    main()
