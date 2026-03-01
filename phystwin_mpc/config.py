from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MPPIConfig:
    n_look_ahead: int = 120
    n_sample: int = 20
    n_update_iter: int = 10
    reward_weight: float = 100.0
    xyz_noise_level: float = 0.004
    quat_noise_level: float = 0.0
    gripper_noise_level: float = 0.0
    segmented_parts: int = 4


@dataclass(frozen=True)
class TaskConfig:
    task: str
    bbox: np.ndarray
    eef_height_penalty_threshold: float
    bbox_margin: float


@dataclass(frozen=True)
class PlanningConfig:
    seed: int
    max_points: int
    mppi: MPPIConfig
    task: TaskConfig
    action_dim: int = 13  # xyz(3) + rot(9) + gripper(1)


def build_plan_config(task: str, seed: int, max_points: int) -> PlanningConfig:
    # TODO: adjust bbox for different tasks
    if task == "rope":
        bbox = np.array([[0.0, 0.6], [-0.35, 0.45], [-0.65, 0.05]], dtype=np.float32)
    elif task == "cloth":
        bbox = np.array([[0.0, 0.7], [-0.35, 0.45], [-0.8, 0.0]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown task: {task}")

    task_cfg = TaskConfig(
        task=task,
        bbox=bbox,
        eef_height_penalty_threshold=-0.02,
        bbox_margin=0.02,
    )
    return PlanningConfig(
        seed=seed,
        max_points=max_points,
        mppi=MPPIConfig(),
        task=task_cfg,
    )
