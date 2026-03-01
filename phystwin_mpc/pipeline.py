from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .config import PlanningConfig
from .pcd import load_point_cloud, downsample_points
from .planner import OpenLoopQQTTPlanner
from .robot import BaseRobot, ActionStep


class OpenLoopPlanningPipeline:
    def __init__(self, config: PlanningConfig, robot: BaseRobot):
        self.cfg = config
        self.robot = robot
        self.planner = OpenLoopQQTTPlanner(config)

    def run(self, current_pcd_path: Path, target_pcd_path: Path, execute: bool, save_dir: Path) -> dict:
        current_pts = load_point_cloud(current_pcd_path)
        target_pts = load_point_cloud(target_pcd_path)

        current_pts = downsample_points(current_pts, self.cfg.max_points, self.cfg.seed)
        target_pts = downsample_points(target_pts, self.cfg.max_points, self.cfg.seed)

        pose = self.robot.get_current_pose()
        result = self.planner.plan(current_pts=current_pts, target_pts=target_pts, current_pose=pose)

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

        if execute:
            seq = [
                ActionStep(
                    xyz=result.best_action_seq[i, :3].copy(),
                    rot=result.best_action_seq[i, 3:12].reshape(3, 3).copy(),
                    gripper=float(result.best_action_seq[i, 12]),
                )
                for i in range(result.best_action_seq.shape[0])
            ]
            self.robot.execute_action_sequence(seq)

        return {
            "best_reward": result.best_reward,
            "final_chamfer": result.final_chamfer,
        }
