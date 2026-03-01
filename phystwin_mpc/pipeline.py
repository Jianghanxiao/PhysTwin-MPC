from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np

from .config import PlanningConfig
from .pcd import load_point_cloud, downsample_points
from .planner import OpenLoopQQTTPlanner
from .robot import BaseRobot, ActionStep
from .visualization import save_rollout_mp4


class OpenLoopPlanningPipeline:
    def __init__(self, config: PlanningConfig, robot: BaseRobot):
        self.cfg = config
        self.robot = robot
        self.planner = OpenLoopQQTTPlanner(config)

    def run(
        self,
        current_pcd_path: Path,
        target_pcd_path: Path,
        execute: bool,
        save_dir: Path,
        video_camera_c2w: np.ndarray,
        video_camera_intrinsic: np.ndarray,
        video_overlay_image: Path | None = None,
    ) -> dict:
        t_start = time.perf_counter()

        t_load_start = time.perf_counter()
        current_pts = load_point_cloud(current_pcd_path)
        target_pts = load_point_cloud(target_pcd_path)
        load_pcd_seconds = time.perf_counter() - t_load_start

        if video_camera_c2w is None:
            raise ValueError("video_camera_c2w is required")
        if video_camera_intrinsic is None:
            raise ValueError("video_camera_intrinsic is required")

        current_pts = downsample_points(current_pts, self.cfg.max_points, self.cfg.seed)
        target_pts = downsample_points(target_pts, self.cfg.max_points, self.cfg.seed)

        t_plan_start = time.perf_counter()
        pose = self.robot.get_current_pose()
        result = self.planner.plan(current_pts=current_pts, target_pts=target_pts, current_pose=pose)
        plan_seconds = time.perf_counter() - t_plan_start

        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "best_action_sequence.npy", result.best_action_seq)

        steps = []
        for i in range(result.best_action_seq.shape[0]):
            row = result.best_action_seq[i]
            step = {
                "index": i,
                "xyz": row[:3].tolist(),
                "rot": row[3:12].reshape(3, 3).tolist(),
                "gripper": float(row[12]),
            }
            steps.append(step)

        with open(save_dir / "best_action_sequence.json", "w", encoding="utf-8") as f:
            json.dump(steps, f, indent=2)

        metrics = {
            "task": self.cfg.task.task,
            "best_reward": result.best_reward,
            "final_chamfer": result.final_chamfer,
            "horizon": int(result.best_action_seq.shape[0]),
            "n_sample": self.cfg.mppi.n_sample,
            "n_update_iter": self.cfg.mppi.n_update_iter,
        }
        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        t_rollout_start = time.perf_counter()
        rollout_seq = self.planner.rollout_trajectory(current_pts=current_pts, action_seq=result.best_action_seq)
        np.save(save_dir / "best_rollout_points.npy", rollout_seq)
        rollout_seconds = time.perf_counter() - t_rollout_start

        eef_xyz_seq = np.concatenate([result.best_action_seq[:1, :3], result.best_action_seq[:, :3]], axis=0)
        eef_rot_seq = np.concatenate(
            [
                result.best_action_seq[:1, 3:12].reshape(1, 3, 3),
                result.best_action_seq[:, 3:12].reshape(result.best_action_seq.shape[0], 3, 3),
            ],
            axis=0,
        )

        eef_traj = [
            {
                "index": int(i),
                "xyz": eef_xyz_seq[i].tolist(),
                "rot": eef_rot_seq[i].tolist(),
            }
            for i in range(eef_xyz_seq.shape[0])
        ]
        with open(save_dir / "eef_trajectory.json", "w", encoding="utf-8") as f:
            json.dump(eef_traj, f, indent=2)

        t_visualize_start = time.perf_counter()
        save_rollout_mp4(
            object_points_seq=rollout_seq,
            eef_xyz_seq=eef_xyz_seq,
            eef_rot_seq=eef_rot_seq,
            target_points=target_pts,
            camera_c2w=video_camera_c2w,
            camera_intrinsic=video_camera_intrinsic,
            overlay_image_path=video_overlay_image,
            save_path=save_dir / "best_rollout.mp4",
        )
        visualize_seconds = time.perf_counter() - t_visualize_start

        execute_seconds = 0.0
        if execute:
            t_execute_start = time.perf_counter()
            seq = [
                ActionStep(
                    xyz=result.best_action_seq[i, :3].copy(),
                    rot=result.best_action_seq[i, 3:12].reshape(3, 3).copy(),
                    gripper=float(result.best_action_seq[i, 12]),
                )
                for i in range(result.best_action_seq.shape[0])
            ]
            self.robot.execute_action_sequence(seq)
            execute_seconds = time.perf_counter() - t_execute_start

        total_seconds = time.perf_counter() - t_start
        timing = {
            "total_seconds": total_seconds,
            "load_pcd_seconds": load_pcd_seconds,
            "plan_seconds": plan_seconds,
            "rollout_seconds": rollout_seconds,
            "visualize_seconds": visualize_seconds,
            "execute_seconds": execute_seconds,
        }
        metrics["timing"] = timing
        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return {
            "best_reward": result.best_reward,
            "final_chamfer": result.final_chamfer,
            "timing": timing,
        }
