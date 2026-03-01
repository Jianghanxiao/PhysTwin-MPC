#!/usr/bin/env python3
"""
Camera calibration script using CharucoBoard.
Outputs calibrate.pkl containing camera intrinsics and extrinsics (c2w transforms).
"""

import sys
import os
import pickle
import cv2
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from phystwin_mpc.qqtt.env.camera import CameraSystem

OUTPUT_PATH = Path("outputs")

def main():
    """
    Run camera calibration using CharucoBoard detection.
    Saves extrinsics as list of 4x4 c2w (camera-to-world) transformation matrices.
    """
    # Create output directory
    output_dir = OUTPUT_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Initializing camera system...")
    camera_system = CameraSystem(WH=[1280, 720], fps=5, num_cam=1)
    
    print("[INFO] Starting calibration process...")
    print("  - Press Enter when ready...")
    input()
    
    # Change to output directory for calibration
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        camera_system.calibrate(visualize=True)
        print(f"[INFO] Calibration completed. Check 'calibrate.pkl' in {output_dir}")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
