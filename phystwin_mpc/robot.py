from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol
import math
import time
import numpy as np


@dataclass
class RobotPose:
    xyz: np.ndarray
    rot: np.ndarray
    gripper: float


@dataclass
class ActionStep:
    xyz: np.ndarray
    rot: np.ndarray
    gripper: float


class BaseRobot(Protocol):
    def get_current_pose(self) -> RobotPose:
        ...

    def execute_action_sequence(self, sequence: list[ActionStep]) -> None:
        ...


class MockRobot:
    def __init__(self) -> None:
        self._pose = RobotPose(
            xyz=np.array([0.35, 0.0, -0.25], dtype=np.float32),
            rot=np.eye(3, dtype=np.float32),
            gripper=0.2,
        )

    def get_current_pose(self) -> RobotPose:
        return RobotPose(
            xyz=self._pose.xyz.copy(),
            rot=self._pose.rot.copy(),
            gripper=float(self._pose.gripper),
        )

    def execute_action_sequence(self, sequence: list[ActionStep]) -> None:
        for i, step in enumerate(sequence):
            self._pose = RobotPose(step.xyz.copy(), step.rot.copy(), float(step.gripper))
            print(
                f"[MockRobot] step={i:03d} xyz={np.round(step.xyz, 4).tolist()} gripper={step.gripper:.3f}"
            )


class CallbackRobotAdapter:
    def __init__(
        self,
        read_pose_fn: Callable[[], RobotPose],
        send_action_fn: Callable[[ActionStep], None],
    ) -> None:
        self._read_pose_fn = read_pose_fn
        self._send_action_fn = send_action_fn

    def get_current_pose(self) -> RobotPose:
        return self._read_pose_fn()

    def execute_action_sequence(self, sequence: list[ActionStep]) -> None:
        for step in sequence:
            self._send_action_fn(step)


def _rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def _rotmat_to_rpy(rot: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(float(rot[0, 0] ** 2 + rot[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw


def _wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


class XArm7Robot:
    def __init__(
        self,
        ip: str,
        speed: float = 100.0,
        acc: float = 2000.0,
        gripper_enable: bool = False,
        base_to_world: np.ndarray | None = None,
        tool_extension_m: float = 0.065,
    ) -> None:
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise RuntimeError(
                "xArm SDK not installed. Install with: pip install xarm-python-sdk"
            ) from exc

        self.arm = XArmAPI(ip)
        self.speed = float(speed)
        self.acc = float(acc)
        self.gripper_enable = bool(gripper_enable)
        self.tool_extension_m = float(tool_extension_m)
        self.tool_offset_local = np.array([0.0, 0.0, self.tool_extension_m], dtype=np.float32)
        if base_to_world is None:
            self.base_to_world = np.eye(4, dtype=np.float32)
        else:
            self.base_to_world = np.asarray(base_to_world, dtype=np.float32)
            if self.base_to_world.shape != (4, 4):
                raise ValueError(f"base_to_world must be shape (4,4), got {self.base_to_world.shape}")
        self.world_to_base = np.linalg.inv(self.base_to_world).astype(np.float32)

        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.move_rate_hz = 250.0
        self.move_sleep = 1.0 / self.move_rate_hz
        self.xyz_velocity_mm = 1.0
        self.angle_velocity_max_deg = 0.2
        if self.gripper_enable:
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_mode(0)

    def _set_servo_cartesian_compat(self, pose: list[float]) -> int:
        call_patterns = [
            lambda: self.arm.set_servo_cartesian(
                pose,
                speed=self.speed,
                mvacc=self.acc,
                is_radian=True,
            ),
            lambda: self.arm.set_servo_cartesian(
                pose=pose,
                speed=self.speed,
                mvacc=self.acc,
                is_radian=True,
            ),
            lambda: self.arm.set_servo_cartesian(
                pose,
                speed=self.speed,
                is_radian=True,
            ),
            lambda: self.arm.set_servo_cartesian(
                pose=pose,
                speed=self.speed,
                is_radian=True,
            ),
            lambda: self.arm.set_servo_cartesian(pose),
            lambda: self.arm.set_servo_cartesian(*pose),
        ]

        for pattern in call_patterns:
            try:
                ret = pattern()
            except TypeError:
                continue
            except Exception:
                continue
            if isinstance(ret, tuple):
                code = int(ret[0])
            else:
                code = int(ret)
            return code
        return -1

    def get_current_pose(self) -> RobotPose:
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"xArm get_position failed: code={code}")
        x_mm, y_mm, z_mm, roll, pitch, yaw = pose
        xyz_base = np.array([x_mm, y_mm, z_mm], dtype=np.float32) / 1000.0
        rot_base = _rpy_to_rotmat(float(roll), float(pitch), float(yaw))
        xyz_base = xyz_base + rot_base @ self.tool_offset_local
        rot_world = self.base_to_world[:3, :3] @ rot_base
        xyz_world = self.base_to_world[:3, :3] @ xyz_base + self.base_to_world[:3, 3]

        gripper = 0.0
        if self.gripper_enable:
            g_code, g_pos = self.arm.get_gripper_position()
            if g_code == 0:
                gripper = float(np.clip(float(g_pos) / 800.0, 0.0, 1.0))
        return RobotPose(xyz=xyz_world, rot=rot_world, gripper=gripper)

    def execute_action_sequence(self, sequence: list[ActionStep]) -> None:
        if len(sequence) == 0:
            return

        mode_code = self.arm.set_mode(1)
        state_code = self.arm.set_state(state=0)
        if mode_code != 0 or state_code != 0:
            raise RuntimeError(f"xArm failed to enter servo mode: set_mode={mode_code}, set_state={state_code}")

        time.sleep(0.05)

        code_cur, pose_cur = self.arm.get_position(is_radian=True)
        if code_cur == 0 and pose_cur is not None:
            prev_pose = np.asarray(pose_cur[:6], dtype=np.float64)
        else:
            first = sequence[0]
            rot_base = self.world_to_base[:3, :3] @ first.rot
            xyz_base = self.world_to_base[:3, :3] @ first.xyz + self.world_to_base[:3, 3]
            xyz_base = xyz_base - rot_base @ self.tool_offset_local
            roll, pitch, yaw = _rotmat_to_rpy(rot_base)
            xyz_mm = xyz_base * 1000.0
            prev_pose = np.array([xyz_mm[0], xyz_mm[1], xyz_mm[2], roll, pitch, yaw], dtype=np.float64)

        for step in sequence:
            rot_base = self.world_to_base[:3, :3] @ step.rot
            xyz_base = self.world_to_base[:3, :3] @ step.xyz + self.world_to_base[:3, 3]
            xyz_base = xyz_base - rot_base @ self.tool_offset_local
            roll, pitch, yaw = _rotmat_to_rpy(rot_base)
            xyz_mm = xyz_base * 1000.0
            target_pose = np.array([xyz_mm[0], xyz_mm[1], xyz_mm[2], roll, pitch, yaw], dtype=np.float64)

            pos_dist = float(np.linalg.norm(target_pose[:3] - prev_pose[:3]))
            min_steps = int(math.ceil(pos_dist / self.xyz_velocity_mm)) if pos_dist > 1e-9 else 1

            ang_delta = _wrap_angle_delta(target_pose[3:] - prev_pose[3:])
            ang_dist_deg = float(np.rad2deg(np.max(np.abs(ang_delta))))
            ang_steps = int(math.ceil(ang_dist_deg / self.angle_velocity_max_deg)) if ang_dist_deg > 1e-9 else 1

            steps = max(1, min_steps, ang_steps)
            for i in range(steps):
                ratio = float(i + 1) / float(steps)
                interp = prev_pose.copy()
                interp[:3] = prev_pose[:3] + ratio * (target_pose[:3] - prev_pose[:3])
                interp[3:] = prev_pose[3:] + ratio * ang_delta

                code = self._set_servo_cartesian_compat(interp.tolist())
                if code != 0:
                    self.arm.set_mode(0)
                    self.arm.set_state(state=0)
                    fallback = self.arm.set_position(
                        x=float(target_pose[0]),
                        y=float(target_pose[1]),
                        z=float(target_pose[2]),
                        roll=float(target_pose[3]),
                        pitch=float(target_pose[4]),
                        yaw=float(target_pose[5]),
                        speed=self.speed,
                        mvacc=self.acc,
                        wait=True,
                        is_radian=True,
                    )
                    if fallback != 0:
                        raise RuntimeError(f"xArm set_servo_cartesian failed: code={code}; fallback set_position failed: code={fallback}")
                    self.arm.set_mode(1)
                    self.arm.set_state(state=0)
                    break
                time.sleep(self.move_sleep)

            prev_pose = target_pose

            if self.gripper_enable:
                g_target = int(np.clip(step.gripper, 0.0, 1.0) * 800.0)
                self.arm.set_gripper_position(g_target, wait=False)

        self.arm.set_mode(0)
        self.arm.set_state(state=0)
