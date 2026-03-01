import importlib

__all__ = ["SpringMassSystemWarp"]


def __getattr__(name):
	if name == "SpringMassSystemWarp":
		module = importlib.import_module(".diff_simulator", __name__)
		return module.SpringMassSystemWarp
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")