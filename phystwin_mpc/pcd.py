from pathlib import Path
import numpy as np


def load_point_cloud(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        pts = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if "pts" in data:
            pts = data["pts"]
        else:
            first_key = list(data.keys())[0]
            pts = data[first_key]
    elif suffix in {".txt", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        pts = np.loadtxt(path, delimiter=delimiter)
    elif suffix == ".ply":
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("Loading .ply requires open3d: pip install open3d") from exc
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")

    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Point cloud must be [N,3+] but got shape {pts.shape}")
    return pts[:, :3]


def downsample_points(pts: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=max_points, replace=False)
    return pts[idx]
