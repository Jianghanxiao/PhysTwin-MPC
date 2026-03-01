import numpy as np
from pathlib import Path
import torch
import warp as wp
import random
import open3d as o3d
import pickle
import json
import glob
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


controller_points_position = np.array(
    [
        [0.01, 0.01, 0.01],
        [-0.01, 0.01, 0.01],
        [0.01, -0.01, 0.01],
        [-0.01, -0.01, 0.01],
        [0.01, 0.01, -0.01],
        [-0.01, 0.01, -0.01],
        [0.01, -0.01, -0.01],
        [-0.01, -0.01, -0.01],
    ]
)


class PhysDynamicModule:
    def __init__(
        self,
        base_path,
        case_name,
        experiments_path,
        experiments_optimization_path,
        output_dir,
        init_pts,
        init_colors,
        init_controller_xyz,
        init_controller_rot,
        action_num,
        batch_size,
        cloth_config_path="experiments/real_world/qqtt/configs/cloth.yaml",
        real_config_path="experiments/real_world/qqtt/configs/real.yaml",
        device="cuda",
    ):
        seed = 42
        set_all_seeds(seed)

        self.batch_size = batch_size

        if "cloth" in case_name or "package" in case_name:
            cfg.load_from_yaml(cloth_config_path)
        else:
            cfg.load_from_yaml(real_config_path)

        base_dir = f"{output_dir}/{case_name}"

        optimal_path = f"{experiments_optimization_path}/{case_name}/optimal_params.pkl"
        logger.info(f"Load optimal parameters from: {optimal_path}")
        assert os.path.exists(
            optimal_path
        ), f"{case_name}: Optimal parameters not found: {optimal_path}"
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

        with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        cfg.c2ws = np.array(c2ws)
        cfg.w2cs = np.array(w2cs)
        with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
            data = json.load(f)
        cfg.intrinsics = np.array(data["intrinsics"])
        cfg.WH = data["WH"]

        logger.set_log_file(path=base_dir, name="inference_log")
        self.trainer = InvPhyTrainerWarp(
            data_path=f"{base_path}/{case_name}/final_data.pkl",
            base_dir=base_dir,
            pure_inference_mode=True,
        )

        self.device = device

        init_controller_xyz = torch.tensor(
            init_controller_xyz, dtype=torch.float, device=self.device
        )
        init_controller_rot = torch.tensor(
            init_controller_rot, dtype=torch.float32, device=self.device
        )
        init_controller_rot = init_controller_rot.permute(0, 2, 1)

        self.controller_points_position = torch.tensor(
            controller_points_position, dtype=torch.float, device=self.device
        )

        self.init_controller_points = (
            torch.einsum(
                "gij,nj->gni", init_controller_rot, self.controller_points_position
            )
            + init_controller_xyz[:, None]
        )
        self.init_controller_points = torch.cat(
            list(self.init_controller_points), dim=0
        )

        final_points = self.align(
            init_pts,
            init_colors,
            self.trainer.dataset.structure_points.cpu().numpy(),
            self.trainer.dataset.original_object_colors[0].cpu().numpy(),
            visualize=True,
        )

        best_model_path = glob.glob(f"{experiments_path}/{case_name}/train/best_*.pth")[
            0
        ]
        # todo: can use load_model_transfer_no_acc
        self.trainer.load_model_transfer(
            best_model_path,
            self.init_controller_points.clone(),
            final_points.copy(),
            action_num=action_num,
            dt=5e-5,
        )

    def align(self, to_pts, to_colors, from_pts, from_colors, visualize=False):
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(from_pts)
        source.colors = o3d.utility.Vector3dVector(from_colors)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(to_pts)
        target.colors = o3d.utility.Vector3dVector(to_colors)
        target = target.voxel_down_sample(voxel_size=0.005)

        source.translate(
            np.mean(target.points, axis=0) - np.mean(source.points, axis=0)
        )

        threshold = 0.02
        trans_init = np.identity(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        final_points = np.array(source.transform(reg_p2p.transformation).points)

        if visualize:
            controller_meshes = []
            # Use sphere mesh for each controller point
            for j in range(self.init_controller_points.shape[0]):
                origin = self.init_controller_points[j]
                origin_color = [1, 0, 0]
                controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.01
                ).translate(origin)
                controller_mesh.compute_vertex_normals()
                controller_mesh.paint_uniform_color(origin_color)
                controller_meshes.append(controller_mesh)

            o3d.visualization.draw_geometries([source, target, controller_meshes])

        return final_points

    def rollout_serialize(self, eef_xyz, eef_rot, visualize=False):
        batch_size = eef_xyz.shape[0]
        assert batch_size == self.batch_size
        all_pts = []

        for i in range(batch_size):
            with wp.ScopedTimer("rollout"):
                controller_points_array = torch.einsum(
                    "bgij,nj->bgni",
                    eef_rot[i].permute(0, 1, 3, 2),
                    self.controller_points_position,
                )
                controller_points_array = (
                    controller_points_array + eef_xyz[i][:, :, None, :]
                )
                controller_points_array = torch.reshape(
                    controller_points_array, [controller_points_array.shape[0], -1, 3]
                )
                controller_points_array = torch.tensor(
                    controller_points_array, dtype=torch.float, device=self.device
                ).contiguous()
                # TODO: can use rollout_no_acc
                pts = self.trainer.rollout(controller_points_array, visualize=(visualize and i < 10))
                all_pts.append(pts.clone())

        return all_pts


class QQTTDynamicsModule:

    def __init__(
        self,
        batch_size,
        num_steps_total,
        base_path,
        case_name,
        experiments_path,
        experiments_optimization_path,
        output_dir,
        cloth_config_path,
        real_config_path,
        device="cuda",
    ):

        self.dynamics_module = None

        self.batch_size = batch_size
        self.action_num = num_steps_total + 1
        self.base_path = base_path
        self.case_name = case_name
        self.experiments_path = experiments_path
        self.experiments_optimization_path = experiments_optimization_path
        self.output_dir = output_dir
        self.cloth_config_path = cloth_config_path
        self.real_config_path = real_config_path
        self.device = device

    def reset_model(self, x=None):
        return

    def reset_preprocess_meta(self, pts):
        pass

    def reset_downsample_indices(self, pts, uniform=True):
        pass

    def rollout(self, pts, eef_xyz, eef_rot, eef_gripper, pts_his=None, visualize_pv=False):

        assert eef_xyz.shape[1] == self.action_num
        assert eef_rot.shape[1] == self.action_num
        assert eef_gripper.shape[1] == self.action_num

        eef_rot = eef_rot.permute(0, 1, 2, 4, 3)

        if self.dynamics_module is None:
            init_pts = pts.cpu().numpy()
            init_colors = np.zeros_like(init_pts)

            init_controller_xyz = eef_xyz[0, 0].cpu().numpy()
            init_controller_rot = eef_rot[0, 0].cpu().numpy()

            self.dynamics_module = PhysDynamicModule(
                base_path=self.base_path,
                case_name=self.case_name,
                experiments_path=self.experiments_path,
                experiments_optimization_path=self.experiments_optimization_path,
                output_dir=self.output_dir,
                init_pts=init_pts,
                init_colors=init_colors,
                init_controller_xyz=init_controller_xyz,
                init_controller_rot=init_controller_rot,
                action_num=self.action_num,
                batch_size=self.batch_size,
                cloth_config_path=self.cloth_config_path,
                real_config_path=self.real_config_path,
                device=self.device,
            )

        controller_xyzs = eef_xyz[:, 1:]
        controller_rots = eef_rot[:, 1:]
        print("Finish initialization!!!!!!!!!!!!!!!!!!!!")

        results = self.dynamics_module.rollout_serialize(
            controller_xyzs, controller_rots,
            visualize=visualize_pv,
        )
        x = torch.stack(results, dim=0)
        x = x[:, None]

        v = torch.zeros_like(x)
        return x, v

    def do_pv(self, x_seq, grippers_seq, gripper_points_seq=None, pv_id=0, clean_bg=False):
        pass
