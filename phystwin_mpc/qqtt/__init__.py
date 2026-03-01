import importlib
import sys

__all__ = ["SpringMassSystemWarp", "InvPhyTrainerWarp", "OptimizerCMA"]


def __getattr__(name):
	if name == "SpringMassSystemWarp":
		try:
			module = importlib.import_module(".model", __name__)
			return module.SpringMassSystemWarp
		except Exception as error:
			raise ImportError(
				"Failed to import SpringMassSystemWarp. Install QQTT model dependencies "
				"(e.g. warp/open3d/torch stack) before using this symbol."
			) from error

	if name in {"InvPhyTrainerWarp", "OptimizerCMA"}:
		try:
			module = importlib.import_module(".engine", __name__)
			return getattr(module, name)
		except Exception as error:
			raise ImportError(
				f"Failed to import {name}. Install QQTT engine dependencies "
				"(e.g. warp/open3d/torch/wandb/pynput) before using this symbol."
			) from error

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Keep backward compatibility for historical absolute imports like `from qqtt...`.
sys.modules.setdefault("qqtt", sys.modules[__name__])
