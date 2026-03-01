import importlib

__all__ = ["CameraSystem"]


def __getattr__(name):
	if name == "CameraSystem":
		module = importlib.import_module(".camera", __name__)
		return module.CameraSystem
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")