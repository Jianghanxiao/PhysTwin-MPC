import numpy as np
import torch
import pickle
from ..utils import logger, visualize_pc, cfg
import matplotlib.pyplot as plt


REF_T_MARKER2WORLD = np.array(
    [
        [9.92500579e-01, -1.22225711e-01, 1.86443478e-03, 1.36186366e-01],
        [5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
        [1.22238720e-01, 9.92443176e-01, -1.06881781e-02, 7.19721945e-02],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _apply_rigid_transform(points, transform):
    original_shape = points.shape
    points_flat = points.reshape(-1, 3)
    points_homogeneous = np.hstack((points_flat, np.ones((points_flat.shape[0], 1))))
    transformed = (transform @ points_homogeneous.T).T[:, :3]
    return transformed.reshape(original_shape)


class RealData:
    def __init__(self, visualize=False, save_gt=True):
        logger.info(f"[DATA]: loading data from {cfg.data_path}")
        self.data_path = cfg.data_path
        self.base_dir = cfg.base_dir
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        self.T_world2marker = np.linalg.inv(REF_T_MARKER2WORLD)
        cfg.T_world2marker = self.T_world2marker
        logger.info("[DATA]: using hardcoded reference T_world2marker")

        object_points = data["object_points"]
        object_colors = data["object_colors"]
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        controller_points = data["controller_points"]
        other_surface_points = data["surface_points"]
        interior_points = data["interior_points"]

        object_points = _apply_rigid_transform(object_points, self.T_world2marker)
        controller_points = _apply_rigid_transform(
            controller_points, self.T_world2marker
        )
        other_surface_points = _apply_rigid_transform(
            other_surface_points, self.T_world2marker
        )
        interior_points = _apply_rigid_transform(
            interior_points, self.T_world2marker
        )

        # Get the rainbow color for the object_colors
        y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
        y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
        rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

        self.num_original_points = object_points.shape[1]
        self.num_surface_points = (
            self.num_original_points + other_surface_points.shape[0]
        )
        self.num_all_points = self.num_surface_points + interior_points.shape[0]

        # Concatenate the surface points and interior points
        self.structure_points = np.concatenate(
            [object_points[0], other_surface_points, interior_points], axis=0
        )
        self.structure_points = torch.tensor(
            self.structure_points, dtype=torch.float32, device=cfg.device
        )

        self.object_points = torch.tensor(
            object_points, dtype=torch.float32, device=cfg.device
        )
        # self.object_colors = torch.tensor(
        #     object_colors, dtype=torch.float32, device=cfg.device
        # )
        self.original_object_colors = torch.tensor(
            object_colors, dtype=torch.float32, device=cfg.device
        )
        # Apply the rainbow color to the object_colors
        rainbow_colors = torch.tensor(
            rainbow_colors, dtype=torch.float32, device=cfg.device
        )
        # Make the same rainbow color for each frame
        self.object_colors = rainbow_colors.repeat(self.object_points.shape[0], 1, 1)

        # # Apply the first frame color to all frames
        # first_frame_colors = torch.tensor(
        #     object_colors[0], dtype=torch.float32, device=cfg.device
        # )
        # self.object_colors = first_frame_colors.repeat(self.object_points.shape[0], 1, 1)

        self.object_visibilities = torch.tensor(
            object_visibilities, dtype=torch.bool, device=cfg.device
        )
        self.object_motions_valid = torch.tensor(
            object_motions_valid, dtype=torch.bool, device=cfg.device
        )
        self.controller_points = torch.tensor(
            controller_points, dtype=torch.float32, device=cfg.device
        )

        self.frame_len = self.object_points.shape[0]
        # Visualize/save the GT frames
        self.visualize_data(visualize=visualize, save_gt=save_gt)

    def visualize_data(self, visualize=False, save_gt=True):
        if visualize:
            visualize_pc(
                self.object_points,
                self.object_colors,
                self.controller_points,
                self.object_visibilities,
                self.object_motions_valid,
                visualize=True,
            )
        if save_gt:
            visualize_pc(
                self.object_points,
                self.object_colors,
                self.controller_points,
                self.object_visibilities,
                self.object_motions_valid,
                visualize=False,
                save_video=True,
                save_path=f"{self.base_dir}/gt.mp4",
            )
