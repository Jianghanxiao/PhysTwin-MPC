import importlib

__all__ = ["OptimizerCMA", "InvPhyTrainerWarp"]


def __getattr__(name):
	if name == "OptimizerCMA":
		module = importlib.import_module(".cma_optimize_warp", __name__)
		return module.OptimizerCMA
	if name == "InvPhyTrainerWarp":
		module = importlib.import_module(".trainer_warp", __name__)
		return module.InvPhyTrainerWarp
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")