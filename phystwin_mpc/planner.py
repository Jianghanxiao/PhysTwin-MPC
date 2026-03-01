from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import kornia

from .config import PlanningConfig
from .planning_utils import batch_chamfer_dist
from .qqtt_dynamics import QQTTDynamicsModule
from .robot import RobotPose


@dataclass
class PlanResult:
    best_action_seq: np.ndarray
    best_reward: float
    final_chamfer: float


class OpenLoopQQTTPlanner:
    def __init__(self, config: PlanningConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)
        self.dynamics = QQTTDynamicsModule(batch_size=config.mppi.n_sample, num_steps_total=config.mppi.n_look_ahead)
        self.bbox_t = torch.tensor(config.task.bbox, dtype=torch.float32, device=self.device)
        self.margin = float(config.task.bbox_margin)
        self.height_threshold = float(config.task.eef_height_penalty_threshold)

    def _init_action_seq(self, pose: RobotPose) -> torch.Tensor:
        t = self.cfg.mppi.n_look_ahead
        action = torch.zeros((t, self.cfg.action_dim), dtype=torch.float32, device=self.device)
        action[:, :3] = torch.as_tensor(pose.xyz, dtype=torch.float32, device=self.device).reshape(1, 3)
        action[:, 3:12] = torch.as_tensor(pose.rot, dtype=torch.float32, device=self.device).reshape(1, 9)
        action[:, 12] = float(pose.gripper)
        return action

    def _clip_actions(self, action_seqs: torch.Tensor) -> torch.Tensor:
        seqs = action_seqs.clone()
        seqs[..., 0] = torch.clamp(seqs[..., 0], self.bbox_t[0, 0], self.bbox_t[0, 1])
        seqs[..., 1] = torch.clamp(seqs[..., 1], self.bbox_t[1, 0], self.bbox_t[1, 1])
        seqs[..., 2] = torch.clamp(seqs[..., 2], self.bbox_t[2, 0], self.bbox_t[2, 1])
        seqs[..., 12] = torch.clamp(seqs[..., 12], 0.0, 1.0)

        rot = seqs[..., 3:12].reshape(-1, 3, 3)
        quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(rot)
        quat = quat / (torch.linalg.norm(quat, dim=-1, keepdim=True) + 1e-8)
        seqs[..., 3:12] = kornia.geometry.conversions.quaternion_to_rotation_matrix(quat).reshape(*seqs[..., 3:12].shape)
        return seqs

    def _sample_action_seqs(self, action_seq: torch.Tensor, iter_index: int) -> torch.Tensor:
        n_sample = self.cfg.mppi.n_sample
        horizon = self.cfg.mppi.n_look_ahead
        n_parts = self.cfg.mppi.segmented_parts

        base_xyz = action_seq[:, :3]
        base_rot = action_seq[:, 3:12].reshape(horizon, 3, 3)
        base_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(base_rot)
        base_gripper = action_seq[:, 12:13]

        xyz_delta_parts = []
        quat_delta_parts = []
        gripper_delta_parts = []
        for p in range(n_parts):
            if p < n_parts - 1:
                p_len = horizon // n_parts
            else:
                p_len = horizon - (n_parts - 1) * (horizon // n_parts)

            decay = 1.0 / float(iter_index + 1)
            xyz_delta = torch.randn((n_sample, 1, 3), device=self.device, dtype=torch.float32) * (self.cfg.mppi.xyz_noise_level * decay)
            quat_delta = torch.randn((n_sample, 1, 4), device=self.device, dtype=torch.float32) * (self.cfg.mppi.quat_noise_level * decay)
            gripper_delta = torch.randn((n_sample, 1, 1), device=self.device, dtype=torch.float32) * (self.cfg.mppi.gripper_noise_level * decay)
            xyz_delta = xyz_delta.repeat(1, p_len, 1)
            quat_delta = quat_delta.repeat(1, p_len, 1)
            gripper_delta = gripper_delta.repeat(1, p_len, 1)
            xyz_delta_parts.append(xyz_delta)
            quat_delta_parts.append(quat_delta)
            gripper_delta_parts.append(gripper_delta)

        xyz_delta = torch.cat(xyz_delta_parts, dim=1)
        quat_delta = torch.cat(quat_delta_parts, dim=1)
        gripper_delta = torch.cat(gripper_delta_parts, dim=1)

        xyz_delta_cum = torch.cumsum(xyz_delta, dim=1)
        quat_delta_cum = torch.cumsum(quat_delta, dim=1)
        gripper_delta_cum = torch.cumsum(gripper_delta, dim=1)

        xyz = base_xyz[None] + xyz_delta_cum
        quat = base_quat[None] + quat_delta_cum
        quat = quat / (torch.linalg.norm(quat, dim=-1, keepdim=True) + 1e-8)
        gripper = base_gripper[None] + gripper_delta_cum

        seqs = torch.zeros((n_sample, horizon, 13), dtype=torch.float32, device=self.device)
        seqs[:, :, :3] = xyz
        seqs[:, :, 12:13] = gripper
        seqs[:, :, 3:12] = kornia.geometry.conversions.quaternion_to_rotation_matrix(quat).reshape(n_sample, horizon, 9)

        return self._clip_actions(seqs)

    def _evaluate_rewards(self, state_finals: torch.Tensor, action_seqs: torch.Tensor, target_pts_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        chamfers = batch_chamfer_dist(state_finals, target_pts_t)

        xyz = action_seqs[:, :, :3]
        eef_height_penalty = (xyz[:, :, 2].amax(dim=1) > self.height_threshold).to(torch.float32) * 100.0

        out_of_bbox_upper = (xyz.amax(dim=1) > (self.bbox_t[:, 1] - self.margin)).any(dim=1)
        out_of_bbox_lower = (xyz.amin(dim=1) < (self.bbox_t[:, 0] + self.margin)).any(dim=1)
        bbox_penalty = (out_of_bbox_upper | out_of_bbox_lower).to(torch.float32) * 100.0

        rewards = -chamfers - eef_height_penalty - bbox_penalty
        return rewards, chamfers

    def _rollout_final_states(self, pts_t: torch.Tensor, action_seqs: torch.Tensor) -> torch.Tensor:
        horizon = self.cfg.mppi.n_look_ahead

        n_sample = action_seqs.shape[0]
        eef_xyz = action_seqs[:, :, :3].reshape(n_sample, horizon, 1, 3)
        eef_rot = action_seqs[:, :, 3:12].reshape(n_sample, horizon, 1, 3, 3)
        eef_gripper = action_seqs[:, :, 12:13].reshape(n_sample, horizon, 1, 1)

        eef_xyz = torch.cat([eef_xyz[:, :1], eef_xyz], dim=1)
        eef_rot = torch.cat([eef_rot[:, :1], eef_rot], dim=1)
        eef_gripper = torch.cat([eef_gripper[:, :1], eef_gripper], dim=1)

        x, _ = self.dynamics.rollout(
            pts_t,
            eef_xyz,
            eef_rot,
            eef_gripper,
            pts_his=None,
            visualize_pv=False,
        )
        return x[:, -1]

    def _mppi_update(self, sampled_action: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        w = torch.softmax((rewards - rewards.max()) * self.cfg.mppi.reward_weight, dim=0)

        xyz = (sampled_action[:, :, :3] * w[:, None, None]).sum(dim=0)
        gripper = (sampled_action[:, :, 12:13] * w[:, None, None]).sum(dim=0)

        quat_all = kornia.geometry.conversions.rotation_matrix_to_quaternion(
            sampled_action[:, :, 3:12].reshape(sampled_action.shape[0], sampled_action.shape[1], 3, 3)
        )
        quat = (quat_all * w[:, None, None]).sum(dim=0)
        quat = quat / (torch.linalg.norm(quat, dim=-1, keepdim=True) + 1e-8)

        out = torch.zeros((sampled_action.shape[1], 13), dtype=torch.float32, device=self.device)
        out[:, :3] = xyz
        out[:, 12:13] = gripper
        out[:, 3:12] = kornia.geometry.conversions.quaternion_to_rotation_matrix(quat).reshape(sampled_action.shape[1], 9)

        return self._clip_actions(out)

    def plan(self, current_pts: np.ndarray, target_pts: np.ndarray, current_pose: RobotPose) -> PlanResult:
        current_pts_t = torch.as_tensor(current_pts, dtype=torch.float32, device=self.device)
        target_pts_t = torch.as_tensor(target_pts, dtype=torch.float32, device=self.device)
        action_seq = self._init_action_seq(current_pose)

        best_seq = action_seq.clone()
        best_reward = -float("inf")
        best_chamfer = float("inf")

        with torch.no_grad():
            for it in range(self.cfg.mppi.n_update_iter):
                sampled = self._sample_action_seqs(action_seq, iter_index=it)
                final_states = self._rollout_final_states(current_pts_t, sampled)
                rewards, chamfers = self._evaluate_rewards(final_states, sampled, target_pts_t)

                action_seq = self._mppi_update(sampled, rewards)

                idx = int(torch.argmax(rewards).item())
                if float(rewards[idx].item()) > best_reward:
                    best_reward = float(rewards[idx].item())
                    best_seq = sampled[idx].clone()
                    best_chamfer = float(chamfers[idx].item())

                print(
                    f"[MPPI] iter={it:02d}/{self.cfg.mppi.n_update_iter - 1} "
                    f"best_iter_reward={rewards[idx].item():.6f} best_iter_chamfer={chamfers[idx].item():.6f}"
                )

        return PlanResult(best_action_seq=best_seq.detach().cpu().numpy(), best_reward=best_reward, final_chamfer=best_chamfer)
