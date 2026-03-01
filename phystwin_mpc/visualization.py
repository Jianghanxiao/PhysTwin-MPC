from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2
import open3d as o3d


def _make_pose(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot.astype(np.float64)
    pose[:3, 3] = xyz.astype(np.float64)
    return pose


def save_rollout_mp4(
    object_points_seq: np.ndarray,
    eef_xyz_seq: np.ndarray,
    eef_rot_seq: np.ndarray,
    save_path: Path,
    target_points: np.ndarray | None = None,
    fps: int = 20,
    width: int = 1280,
    height: int = 720,
) -> None:
    if object_points_seq.ndim != 3 or object_points_seq.shape[-1] != 3:
        raise ValueError(f"object_points_seq must be [T, N, 3], got {object_points_seq.shape}")
    if eef_xyz_seq.ndim != 2 or eef_xyz_seq.shape[-1] != 3:
        raise ValueError(f"eef_xyz_seq must be [T, 3], got {eef_xyz_seq.shape}")
    if eef_rot_seq.ndim != 3 or eef_rot_seq.shape[-2:] != (3, 3):
        raise ValueError(f"eef_rot_seq must be [T, 3, 3], got {eef_rot_seq.shape}")

    n_frames = object_points_seq.shape[0]
    if eef_xyz_seq.shape[0] != n_frames or eef_rot_seq.shape[0] != n_frames:
        raise ValueError(
            "object and eef sequence lengths must match: "
            f"object={n_frames}, xyz={eef_xyz_seq.shape[0]}, rot={eef_rot_seq.shape[0]}"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_points_seq[0])
    pcd.paint_uniform_color([0.2, 0.6, 1.0])
    vis.add_geometry(pcd)

    target_pcd = None
    if target_points is not None:
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        target_pcd.paint_uniform_color([1.0, 0.2, 0.2])
        vis.add_geometry(target_pcd)

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    vis.add_geometry(world_frame)

    eef_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    eef_marker.compute_vertex_normals()
    eef_marker.paint_uniform_color([1.0, 0.5, 0.0])
    eef_marker.translate(eef_xyz_seq[0].astype(np.float64))
    vis.add_geometry(eef_marker)

    eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
    eef_frame.transform(_make_pose(eef_xyz_seq[0], eef_rot_seq[0]))
    vis.add_geometry(eef_frame)

    eef_path = o3d.geometry.LineSet()
    eef_path.points = o3d.utility.Vector3dVector(eef_xyz_seq[:1].astype(np.float64))
    eef_path.lines = o3d.utility.Vector2iVector([])
    eef_path.colors = o3d.utility.Vector3dVector([])
    vis.add_geometry(eef_path)

    all_points = [object_points_seq[0]]
    if target_points is not None:
        all_points.append(target_points)
    cloud_stack = np.concatenate(all_points, axis=0)
    center = cloud_stack.mean(axis=0)
    extent = np.maximum(cloud_stack.max(axis=0) - cloud_stack.min(axis=0), 1e-3)
    radius = float(np.linalg.norm(extent))

    view = vis.get_view_control()
    view.set_lookat(center.tolist())
    view.set_front([0.8, -0.3, 0.5])
    view.set_up([0.0, 0.0, 1.0])
    view.set_zoom(max(0.2, min(0.9, 0.5 / (radius + 1e-6))))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, float(fps), (width, height))

    prev_eef = eef_xyz_seq[0].astype(np.float64)
    for i in range(n_frames):
        pcd.points = o3d.utility.Vector3dVector(object_points_seq[i])
        vis.update_geometry(pcd)

        curr_eef = eef_xyz_seq[i].astype(np.float64)
        eef_marker.translate(curr_eef - prev_eef)
        vis.update_geometry(eef_marker)
        prev_eef = curr_eef

        vis.remove_geometry(eef_frame, reset_bounding_box=False)
        eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        eef_frame.transform(_make_pose(eef_xyz_seq[i], eef_rot_seq[i]))
        vis.add_geometry(eef_frame, reset_bounding_box=False)

        if i > 0:
            pts = eef_xyz_seq[: i + 1].astype(np.float64)
            lines = np.column_stack([np.arange(i), np.arange(1, i + 1)]).astype(np.int32)
            colors = np.tile(np.array([[1.0, 0.8, 0.1]], dtype=np.float64), (lines.shape[0], 1))
            eef_path.points = o3d.utility.Vector3dVector(pts)
            eef_path.lines = o3d.utility.Vector2iVector(lines)
            eef_path.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(eef_path)

        vis.poll_events()
        vis.update_renderer()

        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    vis.destroy_window()
