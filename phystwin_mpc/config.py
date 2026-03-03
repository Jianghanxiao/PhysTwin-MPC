from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


# @dataclass(frozen=True)
# class MPPIConfig:
#     n_look_ahead: int = 120
#     n_sample: int = 100
#     n_update_iter: int = 25
#     reward_weight: float = 100.0
#     xyz_noise_level: float = 0.004
#     quat_noise_level: float = 0.0
#     gripper_noise_level: float = 0.0
#     segmented_parts: int = 10


@dataclass(frozen=True)
class MPPIConfig:
    n_look_ahead: int = 120
    n_sample: int = 20
    n_update_iter: int = 10
    reward_weight: float = 100.0
    xyz_noise_level: float = 0.004
    quat_noise_level: float = 0.0
    gripper_noise_level: float = 0.0
    segmented_parts: int = 10

@dataclass(frozen=True)
class TaskConfig:
    task: str
    bbox: np.ndarray
    eef_height_penalty_threshold: float
    bbox_margin: float


@dataclass(frozen=True)
class QQTTDynamicsConfig:
    base_path: str = "phystwin_dataset_assets/different_types"
    case_name: str = "rope_0"
    experiments_path: str = "phystwin_dataset_assets/experiments"
    experiments_optimization_path: str = "phystwin_dataset_assets/experiments_optimization"
    output_dir: str = "outputs_exp"
    cloth_config_path: str = "phystwin_mpc/qqtt/configs/cloth.yaml"
    real_config_path: str = "phystwin_mpc/qqtt/configs/real.yaml"


@dataclass(frozen=True)
class PlanningConfig:
    seed: int
    max_points: int
    mppi: MPPIConfig
    task: TaskConfig
    qqtt_dynamics: QQTTDynamicsConfig
    action_dim: int = 13  # xyz(3) + rot(9) + gripper(1)


def _list_available_cases(dataset_root: Path) -> list[str]:
    base_path = dataset_root / "different_types"
    experiments_path = dataset_root / "experiments"
    experiments_optimization_path = dataset_root / "experiments_optimization"

    required_dirs = [base_path, experiments_path, experiments_optimization_path]
    missing = [str(p) for p in required_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset root layout is incomplete. Missing directories: "
            + ", ".join(missing)
        )

    base_cases = {p.name for p in base_path.iterdir() if p.is_dir()}
    exp_cases = {p.name for p in experiments_path.iterdir() if p.is_dir()}
    opt_cases = {p.name for p in experiments_optimization_path.iterdir() if p.is_dir()}
    cases = sorted(base_cases & exp_cases & opt_cases)

    if not cases:
        raise FileNotFoundError(
            "No common case folders found across different_types/experiments/experiments_optimization."
        )
    return cases


def _resolve_case_name(dataset_root: Path, task: str, case_name: Optional[str]) -> str:
    available_cases = _list_available_cases(dataset_root)

    if case_name is not None:
        if case_name not in available_cases:
            raise ValueError(
                f"Requested case_name '{case_name}' not found in dataset. "
                f"Available common cases: {available_cases}"
            )
        return case_name

    task_candidates = [name for name in available_cases if task in name]
    if not task_candidates:
        raise ValueError(
            f"No case matched task '{task}' in dataset root '{dataset_root}'. "
            f"Available common cases: {available_cases}"
        )
    return task_candidates[0]


def build_plan_config(
    task: str,
    seed: int,
    max_points: int,
    dataset_root: str = "phystwin_dataset_assets",
    case_name: Optional[str] = None,
) -> PlanningConfig:
    if task == "rope":
        bbox = np.array([[0.0, 0.6], [-0.35, 0.45], [-0.65, 0.05]], dtype=np.float32)
    elif task in {"cloth", "bear"}:
        bbox = np.array([[0.0, 0.7], [-0.35, 0.45], [-0.8, 0.02]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown task: {task}. Expected one of: rope, cloth, bear")

    dataset_root_path = Path(dataset_root)
    resolved_case_name = _resolve_case_name(dataset_root_path, task=task, case_name=case_name)

    task_cfg = TaskConfig(
        task=task,
        bbox=bbox,
        eef_height_penalty_threshold=0,
        bbox_margin=0.02,
    )
    return PlanningConfig(
        seed=seed,
        max_points=max_points,
        mppi=MPPIConfig(),
        task=task_cfg,
        qqtt_dynamics=QQTTDynamicsConfig(
            base_path=str(dataset_root_path / "different_types"),
            case_name=resolved_case_name,
            experiments_path=str(dataset_root_path / "experiments"),
            experiments_optimization_path=str(dataset_root_path / "experiments_optimization"),
        ),
    )
