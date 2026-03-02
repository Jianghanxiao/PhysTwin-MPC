from ..data import RealData, SimpleData
from ..utils import logger, visualize_pc, cfg
from ..model.diff_simulator import (
    SpringMassSystemWarp,
    SpringMassSystemWarpAccelerate,
    SpringMassSystemWarpBatched,
)
import open3d as o3d
import numpy as np
import torch
import wandb
import os
from tqdm import tqdm
import warp as wp
from scipy.spatial import KDTree
import pickle
import cv2
from pynput import keyboard


class InvPhyTrainerWarp:
    def __init__(
        self,
        data_path,
        base_dir,
        train_frame=None,
        mask_path=None,
        velocity_path=None,
        pure_inference_mode=False,
        device="cuda:0",
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        cfg.train_frame = train_frame

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False, save_gt=False)
            # Get the object points and controller points
            self.object_points = self.dataset.object_points
            self.object_colors = self.dataset.object_colors
            self.object_visibilities = self.dataset.object_visibilities
            self.object_motions_valid = self.dataset.object_motions_valid
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_original_points = self.dataset.num_original_points
            self.num_surface_points = self.dataset.num_surface_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            self.dataset = SimpleData(visualize=False)
            self.object_points = self.dataset.data
            self.object_colors = None
            self.object_visibilities = None
            self.object_motions_valid = None
            self.controller_points = None
            self.structure_points = self.dataset.data[0]
            self.num_original_points = None
            self.num_surface_points = None
            self.num_all_points = len(self.dataset.data[0])
            # Prepare for the multiple object case
            if mask_path is not None:
                mask = np.load(mask_path)
                self.init_masks = torch.tensor(
                    mask, dtype=torch.float32, device=cfg.device
                )
            if velocity_path is not None:
                velocity = np.load(velocity_path)
                self.init_velocities = torch.tensor(
                    velocity, dtype=torch.float32, device=cfg.device
                )
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")

        # Initialize the vertices, springs, rest lengths and masses
        if self.controller_points is None:
            firt_frame_controller_points = None
        else:
            firt_frame_controller_points = self.controller_points[0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            firt_frame_controller_points,
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
            mask=self.init_masks,
        )

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
        )

        if not pure_inference_mode:
            self.optimizer = torch.optim.Adam(
                [
                    wp.to_torch(self.simulator.wp_spring_Y),
                    wp.to_torch(self.simulator.wp_collide_elas),
                    wp.to_torch(self.simulator.wp_collide_fric),
                    wp.to_torch(self.simulator.wp_collide_object_elas),
                    wp.to_torch(self.simulator.wp_collide_object_fric),
                ],
                lr=cfg.base_lr,
                betas=(0.9, 0.99),
            )

            if "debug" not in cfg.run_name:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="final_pipeline",
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                )
            else:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Debug",
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                )
            if not os.path.exists(f"{cfg.base_dir}/train"):
                # Create directory if it doesn't exist
                os.makedirs(f"{cfg.base_dir}/train")

        self.batch_simulator = None
        self.batch_size_loaded = None
        self._single_transfer = None

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

            # Connect the springs of the objects first
            points = np.asarray(object_pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (
                        spring_flags[i, j] == 0
                        and spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)

            if controller_points is not None:
                # Connect the springs between the controller points and the object points
                num_object_points = len(points)
                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j])
                        )

            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = object_points[mask == value]
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
                temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
                temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                        temp_points[i], object_radius, object_max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                        if (
                            temp_spring_flags[i, j] == 0
                            and temp_spring_flags[j, i] == 0
                            and rest_length > 1e-4
                        ):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))

            return (
                torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )

    def train(self, start_epoch=-1):
        # Render the initial visualization
        video_path = f"{cfg.base_dir}/train/init.mp4"
        self.visualize_sim(save_only=True, video_path=video_path)

        best_loss = None
        best_epoch = None
        # Train the model with the physical simulator
        for i in range(start_epoch + 1, cfg.iterations):
            total_loss = 0.0
            if cfg.data_type == "real":
                total_chamfer_loss = 0.0
                total_track_loss = 0.0
                # total_acc_loss = 0.0
            self.simulator.set_init_state(
                self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
            )
            # if cfg.data_type == "real":
            #     self.simulator.set_acc_count(False)
            with wp.ScopedTimer("backward"):
                for j in tqdm(range(1, cfg.train_frame)):
                    self.simulator.set_controller_target(j)
                    if self.simulator.object_collision_flag:
                        self.simulator.update_collision_graph()

                    if cfg.use_graph:
                        wp.capture_launch(self.simulator.graph)
                    else:
                        if cfg.data_type == "real":
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_loss()
                            self.simulator.tape.backward(self.simulator.loss)
                        else:
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_simple_loss()
                            self.simulator.tape.backward(self.simulator.loss)

                    self.optimizer.step()

                    if cfg.data_type == "real":
                        chamfer_loss = wp.to_torch(
                            self.simulator.chamfer_loss, requires_grad=False
                        )
                        track_loss = wp.to_torch(
                            self.simulator.track_loss, requires_grad=False
                        )
                        total_chamfer_loss += chamfer_loss.item()
                        total_track_loss += track_loss.item()

                        # if (
                        #     wp.to_torch(self.simulator.acc_count, requires_grad=False)[
                        #         0
                        #     ]
                        #     == 1
                        # ):
                        #     acc_loss = wp.to_torch(
                        #         self.simulator.acc_loss, requires_grad=False
                        #     )
                        #     total_acc_loss += acc_loss.item()
                        # else:
                        #     self.simulator.set_acc_count(True)

                        # # Update the prev_acc used to calculate the acceleration loss
                        # self.simulator.update_acc()

                    loss = wp.to_torch(self.simulator.loss, requires_grad=False)
                    total_loss += loss.item()

                    if cfg.use_graph:
                        # Only need to clear the gradient, the tape is created in the graph
                        self.simulator.tape.zero()
                    else:
                        # Need to reset the compute graph and clear the gradient
                        self.simulator.tape.reset()
                    self.simulator.clear_loss()
                    # Set the intial state for the next step
                    self.simulator.set_init_state(
                        self.simulator.wp_states[-1].wp_x,
                        self.simulator.wp_states[-1].wp_v,
                    )

            total_loss /= cfg.train_frame - 1
            if cfg.data_type == "real":
                total_chamfer_loss /= cfg.train_frame - 1
                total_track_loss /= cfg.train_frame - 1
                # total_acc_loss /= cfg.train_frame - 2
            wandb.log(
                {
                    "loss": total_loss,
                    "chamfer_loss": (
                        total_chamfer_loss if cfg.data_type == "real" else 0
                    ),
                    "track_loss": total_track_loss if cfg.data_type == "real" else 0,
                    # "acc_loss": total_acc_loss if cfg.data_type == "real" else 0,
                    "collide_else": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ).item(),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ).item(),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ).item(),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ).item(),
                },
                step=i,
            )

            logger.info(f"[Train]: Iteration: {i}, Loss: {total_loss}")

            if i % cfg.vis_interval == 0 or i == cfg.iterations - 1:
                video_path = f"{cfg.base_dir}/train/sim_iter{i}.mp4"
                self.visualize_sim(save_only=True, video_path=video_path)
                wandb.log(
                    {
                        "video": wandb.Video(
                            video_path,
                            format="mp4",
                            fps=cfg.FPS,
                        ),
                    },
                    step=i,
                )
                # Save the parameters
                cur_model = {
                    "epoch": i,
                    "num_object_springs": self.num_object_springs,
                    "spring_Y": torch.exp(
                        wp.to_torch(self.simulator.wp_spring_Y, requires_grad=False)
                    ),
                    "collide_elas": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                if best_loss == None or total_loss < best_loss:
                    # Remove old best model file if it exists
                    if best_loss is not None:
                        old_best_model_path = (
                            f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                        )
                        if os.path.exists(old_best_model_path):
                            os.remove(old_best_model_path)

                    # Update best loss and best epoch
                    best_loss = total_loss
                    best_epoch = i

                    # Save new best model
                    best_model_path = f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                    torch.save(cur_model, best_model_path)
                    logger.info(
                        f"Latest best model saved: epoch {best_epoch} with loss {best_loss}"
                    )

                torch.save(cur_model, f"{cfg.base_dir}/train/iter_{i}.pth")
                logger.info(
                    f"[Visualize]: Visualize the simulation at iteration {i} and save the model"
                )

        wandb.finish()

    def test(self, model_path=None):
        if model_path is not None:
            # Load the model
            logger.info(f"Load model from {model_path}")
            checkpoint = torch.load(model_path, map_location=cfg.device)

            spring_Y = checkpoint["spring_Y"]
            collide_elas = checkpoint["collide_elas"]
            collide_fric = checkpoint["collide_fric"]
            collide_object_elas = checkpoint["collide_object_elas"]
            collide_object_fric = checkpoint["collide_object_fric"]
            num_object_springs = checkpoint["num_object_springs"]

            assert (
                len(spring_Y) == self.simulator.n_springs
            ), "Check if the loaded checkpoint match the config file to connect the springs"

            self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
            self.simulator.set_collide(
                collide_elas.detach().clone(), collide_fric.detach().clone()
            )
            self.simulator.set_collide_object(
                collide_object_elas.detach().clone(),
                collide_object_fric.detach().clone(),
            )

        # Render the initial visualization
        video_path = f"{cfg.base_dir}/inference.mp4"
        save_path = f"{cfg.base_dir}/inference.pkl"
        self.visualize_sim(
            save_only=True,
            video_path=video_path,
            save_trajectory=True,
            save_path=save_path,
        )

    def outdomain_inference(self, model_path, to_case_data_path, final_points):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        spring_Y = spring_Y[: self.num_object_springs]

        # Loda the to_case data
        with open(to_case_data_path, "rb") as f:
            data = pickle.load(f)
        self.object_points = torch.tensor(
            data["object_points"], dtype=torch.float32, device=cfg.device
        )
        self.object_colors = self.dataset.object_colors[0].repeat(
            self.object_points.shape[0], 1, 1
        )

        self.controller_points = torch.tensor(
            data["controller_points"], dtype=torch.float32, device=cfg.device
        )
        self.dataset.frame_len = self.object_points.shape[0]

        controller_points = self.controller_points.cpu().numpy()

        assert self.num_all_points == len(
            final_points
        ), "Check the length of the final points"

        # Update the rest lengths for the springs among the object points
        springs = self.init_springs.cpu().numpy()[: self.num_object_springs]
        rest_lengths = np.linalg.norm(
            final_points[springs[:, 0]] - final_points[springs[:, 1]], axis=1
        )

        springs = springs.tolist()
        rest_lengths = rest_lengths.tolist()
        # Update the connection between the final points and the controller points
        first_frame_controller_points = controller_points[0]
        points = np.concatenate([final_points, first_frame_controller_points], axis=0)
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(final_points)
        pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

        # Process to get the connection distance among the controller points and object points
        # Locate the nearest object point for each controller point
        kdtree = KDTree(final_points)
        _, idx = kdtree.query(first_frame_controller_points, k=1)
        # find the distances
        distances = np.linalg.norm(
            final_points[idx] - first_frame_controller_points, axis=1
        )
        # find the indices of the top 4 controller points that are close
        top_k = 10
        top_k_idx = np.argsort(distances)[:top_k]
        controller_radius = np.ones(first_frame_controller_points.shape[0]) * 0.01
        controller_radius[top_k_idx] = distances[top_k_idx] + 0.005

        for i in range(len(first_frame_controller_points)):
            [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                first_frame_controller_points[i],
                controller_radius[i],
                30,
            )
            for j in idx:
                springs.append([self.num_all_points + i, j])
                rest_lengths.append(
                    np.linalg.norm(first_frame_controller_points[i] - points[j])
                )

        self.init_springs = torch.tensor(
            np.array(springs), dtype=torch.int32, device=cfg.device
        )

        self.init_rest_lengths = torch.tensor(
            np.array(rest_lengths), dtype=torch.float32, device=cfg.device
        )
        self.init_masses = torch.tensor(
            np.ones(len(points)), dtype=torch.float32, device=cfg.device
        )

        self.init_vertices = torch.tensor(
            points,
            dtype=torch.float32,
            device=cfg.device,
        )
        self.controller_points = torch.tensor(
            controller_points, dtype=torch.float32, device=cfg.device
        )

        cfg.dt = 5e-6
        cfg.num_substeps = round(1.0 / cfg.FPS / cfg.dt)
        cfg.collision_dist = 0.005

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
        )

        spring_Y = torch.cat(
            [
                spring_Y,
                3e4
                * torch.ones(
                    self.simulator.n_springs - self.num_object_springs,
                    dtype=torch.float32,
                    device=cfg.device,
                ),
            ]
        )

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(), collide_object_fric.detach().clone()
        )

        # Render the final results
        video_path = f"{cfg.base_dir}/inference.mp4"
        save_path = f"{cfg.base_dir}/inference.pkl"
        self.visualize_sim(
            save_only=True,
            video_path=video_path,
            save_trajectory=True,
            save_path=save_path,
        )

    def visualize_sim(
        self, save_only=True, video_path=None, save_trajectory=False, save_path=None
    ):
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        vertices = [
            wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).cpu()
        ]

        with wp.ScopedTimer("simulate"):
            for i in tqdm(range(1, frame_len)):
                if cfg.data_type == "real":
                    self.simulator.set_controller_target(i, pure_inference=True)
                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(self.simulator.forward_graph)
                else:
                    self.simulator.step()
                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                vertices.append(x.cpu())
                # Set the intial state for the next step
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

        vertices = torch.stack(vertices, dim=0)

        if save_trajectory:
            logger.info(f"Save the trajectory to {save_path}")
            vertices_to_save = vertices.cpu().numpy()
            with open(save_path, "wb") as f:
                pickle.dump(vertices_to_save, f)

        if not save_only:
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=True,
            )
        else:
            assert video_path is not None, "Please provide the video path to save"
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )

    def on_press(self, key, scale=1):
        try:
            if key.char == "w":
                self.target_change = np.array([0.005, 0, 0]) * scale
            elif key.char == "s":
                self.target_change = np.array([-0.005, 0, 0]) * scale
            elif key.char == "a":
                self.target_change = np.array([0, -0.005, 0]) * scale
            elif key.char == "d":
                self.target_change = np.array([0, 0.005, 0]) * scale
            elif key.char == "j":
                self.target_change = np.array([0, 0, 0.005]) * scale
            elif key.char == "k":
                self.target_change = np.array([0, 0, -0.005]) * scale
        except AttributeError:
            pass

    def on_release(self, key):
        self.target_change = np.array([0.0, 0.0, 0.0])

    def interactive_playground(self, model_path):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        logger.info("Party Time Start!!!!")
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)

        vis_vertices = (
            wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
            .cpu()
            .numpy()
        )

        current_target = self.simulator.controller_points[0]
        prev_target = current_target

        vis_controller_points = current_target.cpu().numpy()

        object_colors = self.object_colors.cpu().numpy()[0]
        if object_colors.shape[0] < vis_vertices.shape[0]:
            # If the object_colors is not the same as object_points, fill the colors with black
            object_colors = np.concatenate(
                [
                    object_colors,
                    np.ones(
                        (
                            vis_vertices.shape[0] - object_colors.shape[0],
                            3,
                        )
                    )
                    * 0.3,
                ],
                axis=0,
            )

        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(vis_vertices)
        object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
        vis.add_geometry(object_pcd)

        controller_meshes = []
        prev_center = []
        if vis_controller_points is not None:
            # Use sphere mesh for each controller point
            for j in range(vis_controller_points.shape[0]):
                origin = vis_controller_points[j]
                origin_color = [1, 0, 0]
                controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.01
                ).translate(origin)
                controller_mesh.compute_vertex_normals()
                controller_mesh.paint_uniform_color(origin_color)
                controller_meshes.append(controller_mesh)
                vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)

        view_control = vis.get_view_control()
        camera_params = o3d.camera.PinholeCameraParameters()
        intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic
        )
        camera_params.intrinsic = intrinsic_parameter
        camera_params.extrinsic = w2c
        view_control.convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True
        )

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.target_change = np.zeros(3)

        while True:
            self.simulator.set_controller_interactive(prev_target, current_target)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )
            # add the visualization code here
            vis_vertices = x.cpu().numpy()

            object_pcd.points = o3d.utility.Vector3dVector(vis_vertices)
            vis.update_geometry(object_pcd)

            if vis_controller_points is not None:
                for j in range(vis_controller_points.shape[0]):
                    origin = vis_controller_points[j]
                    controller_meshes[j].translate(origin - prev_center[j])
                    vis.update_geometry(controller_meshes[j])
                    prev_center[j] = origin
            vis.poll_events()
            vis.update_renderer()

            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)

            # Get the mask where the pixel is white
            mask = np.all(frame == [255, 255, 255], axis=-1)
            image_path = f"{cfg.overlay_path}/{vis_cam_idx}/204.png"
            overlay = cv2.imread(image_path)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            frame[mask] = overlay[mask]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Interactive Playground", frame)
            cv2.waitKey(1)

            prev_target = current_target
            current_target += torch.tensor(
                self.target_change, dtype=torch.float32, device=cfg.device
            )
            vis_controller_points = current_target.cpu().numpy()
        listener.stop()

    # ================================================================
    # Preprocessing for batched simulation (ported from boba)
    # ================================================================

    @staticmethod
    def _stable_lexsort(keys):
        """Stable lexicographic sort (most-significant first)."""
        assert len(keys) > 0
        n = keys[0].numel()
        idx = torch.arange(n, device=keys[0].device)
        for k in reversed(keys):
            idx = idx[torch.argsort(k[idx], stable=True)]
        return idx

    def _edge_color_greedy(self, springs, num_vertices, num_object_points):
        """
        Greedy edge coloring on CPU.
        Only object vertices enforce the "no repeated color on incident edges" constraint.
        Controller endpoints are ignored in the constraint (ignore_controllers=True).
        """
        s = springs.detach().to("cpu", dtype=torch.int64).contiguous()
        u = s[:, 0].numpy()
        v = s[:, 1].numpy()
        E = u.shape[0]
        N = int(num_object_points)

        deg_obj = np.bincount(
            np.concatenate([u[u < N], v[v < N]], axis=0), minlength=N
        )

        def deg_of(idx_arr):
            out = np.zeros_like(idx_arr, dtype=np.int64)
            m = idx_arr < N
            out[m] = deg_obj[idx_arr[m]]
            return out

        key = deg_of(u) + deg_of(v)
        order = np.argsort(-key, kind="stable")
        used = [[] for _ in range(N)]
        colors = np.full(E, -1, dtype=np.int32)
        mark = np.zeros(64, dtype=np.int32)
        stamp = 1
        max_color = -1

        for e in order:
            a = int(u[e]); b = int(v[e])
            stamp += 1
            endpoints = []
            if a < N:
                endpoints.append(a)
            if b < N:
                endpoints.append(b)

            for vv in endpoints:
                for cc in used[vv]:
                    mark[cc] = stamp

            cc = 0
            while cc <= max_color and mark[cc] == stamp:
                cc += 1
            if cc > max_color:
                max_color = cc
                if max_color >= mark.shape[0]:
                    new_size = max(2 * mark.shape[0], max_color + 1)
                    new_mark = np.zeros(new_size, dtype=np.int32)
                    new_mark[: mark.shape[0]] = mark
                    mark = new_mark

            colors[e] = cc
            for vv in endpoints:
                used[vv].append(cc)

        num_colors = int(max_color + 1)
        return torch.from_numpy(colors), num_colors

    def _reorder_springs_spatial_blocking_by_color(
        self, springs, rest_lengths, colors_cpu, num_colors, num_object_points, block_size=32
    ):
        """Reorder springs within each color group by spatial blocking."""
        device = springs.device
        N = int(num_object_points)
        s = springs.long()
        i_all = s[:, 0]
        j_all = s[:, 1]

        perm_chunks = []
        color_offsets = [0]

        for c in range(num_colors):
            ids_cpu = torch.nonzero(colors_cpu == c, as_tuple=False).squeeze(1)
            if ids_cpu.numel() == 0:
                color_offsets.append(color_offsets[-1])
                continue

            ids = ids_cpu.to(device)
            i = i_all[ids]
            j = j_all[ids]

            obj_obj_mask = (i < N) & (j < N)
            ctrl_mask = ~obj_obj_mask
            parts = []

            obj_loc = torch.nonzero(obj_obj_mask, as_tuple=False).squeeze(1)
            if obj_loc.numel() > 0:
                io = i[obj_loc]
                jo = j[obj_loc]
                a = torch.minimum(io, jo)
                b = torch.maximum(io, jo)
                perm_obj = self._stable_lexsort([a // block_size, b // block_size, a, b])
                parts.append(ids[obj_loc[perm_obj]])

            ctrl_loc = torch.nonzero(ctrl_mask, as_tuple=False).squeeze(1)
            if ctrl_loc.numel() > 0:
                ic = i[ctrl_loc]
                jc = j[ctrl_loc]
                obj_ep = torch.where(ic < N, ic, jc)
                ctrl_ep = torch.where(ic >= N, ic, jc)
                ctrl_id = (ctrl_ep - N).clamp_min(0)
                perm_ctrl = self._stable_lexsort([obj_ep // block_size, obj_ep, ctrl_id])
                parts.append(ids[ctrl_loc[perm_ctrl]])

            ids_ordered = torch.cat(parts, dim=0)
            perm_chunks.append(ids_ordered)
            color_offsets.append(color_offsets[-1] + int(ids_ordered.numel()))

        perm = torch.cat(perm_chunks, dim=0)
        assert perm.numel() == springs.shape[0]
        color_offsets_cpu = torch.tensor(color_offsets, dtype=torch.long, device="cpu")
        return springs[perm], rest_lengths[perm], perm, color_offsets_cpu

    def _reorder_springs_spatial_blocking(self, springs, rest_lengths, num_object_points, block_size=32):
        """Reorder springs for spatial blocking (without color groups)."""
        N = int(num_object_points)
        s = springs.long()
        i = s[:, 0]
        j = s[:, 1]

        obj_obj_mask = (i < N) & (j < N)
        ctrl_mask = ~obj_obj_mask

        obj_ids = torch.nonzero(obj_obj_mask, as_tuple=False).squeeze(1)
        if obj_ids.numel() > 0:
            io = i[obj_ids]
            jo = j[obj_ids]
            a = torch.minimum(io, jo)
            b = torch.maximum(io, jo)
            perm_obj_local = self._stable_lexsort([a // block_size, b // block_size, a, b])
            obj_order = obj_ids[perm_obj_local]
        else:
            obj_order = obj_ids

        ctrl_ids = torch.nonzero(ctrl_mask, as_tuple=False).squeeze(1)
        if ctrl_ids.numel() > 0:
            ic = i[ctrl_ids]
            jc = j[ctrl_ids]
            obj_ep = torch.where(ic < N, ic, jc)
            ctrl_ep = torch.where(ic >= N, ic, jc)
            ctrl_id = (ctrl_ep - N).clamp_min(0)
            perm_ctrl_local = self._stable_lexsort([obj_ep // block_size, obj_ep, ctrl_id])
            ctrl_order = ctrl_ids[perm_ctrl_local]
        else:
            ctrl_order = ctrl_ids

        perm = torch.cat([obj_order, ctrl_order], dim=0)
        return springs[perm], rest_lengths[perm], perm

    @staticmethod
    def _build_massnode_coloring_map(
        springs_colored, spring_color_offsets, perm_colored_to_base,
        object_massnode_single, num_spring_colors, device="cuda:0",
    ):
        """
        Build the massnode coloring map for the two-phase spring force reduction.
        map[obj_node, color] = ±(base_spring_id + 1), or 0 if no spring.
        """
        sc = springs_colored.detach().cpu().to(torch.int64)
        perm = perm_colored_to_base.detach().cpu().to(torch.int64)
        offs = spring_color_offsets.detach().cpu().to(torch.int64).tolist()

        E = sc.shape[0]
        map_cpu = torch.zeros((object_massnode_single, num_spring_colors), dtype=torch.int32)

        for c in range(num_spring_colors):
            beg, end = offs[c], offs[c + 1]
            for k in range(beg, end):
                base_idx = int(perm[k].item())
                a = int(sc[k, 0].item())
                b = int(sc[k, 1].item())

                if a < object_massnode_single:
                    val = base_idx + 1
                    if map_cpu[a, c] != 0:
                        raise RuntimeError(f"Conflict: node {a} already assigned in color {c}")
                    map_cpu[a, c] = val

                if b < object_massnode_single:
                    val = -(base_idx + 1)
                    if map_cpu[b, c] != 0:
                        raise RuntimeError(f"Conflict: node {b} already assigned in color {c}")
                    map_cpu[b, c] = val

        return map_cpu.reshape(-1).contiguous().to(device=device)

    def _apply_morton_reordering_with_spring_coloring(self, vertices, springs, rest_lengths, num_object_points):
        """
        Apply Morton (Z-order) reordering to object vertices, then edge-color springs,
        then spatially block within each color. Returns preprocessed data.
        """
        device = vertices.device
        obj_end = num_object_points
        has_controllers = (num_object_points < len(vertices))

        # Step 1: Morton reorder object vertices
        obj_verts = vertices[:obj_end].detach().cpu().numpy()
        mins = obj_verts.min(axis=0)
        maxs = obj_verts.max(axis=0)
        range_vals = maxs - mins
        range_vals[range_vals < 1e-8] = 1.0
        normalized = (obj_verts - mins) / range_vals

        BITS = 21
        MAX_VAL = (1 << BITS) - 1
        int_coords = (normalized * MAX_VAL).astype(np.uint64)

        def part1by2(n):
            n = np.uint64(n)
            n = (n | (n << 32)) & np.uint64(0x1f00000000ffff)
            n = (n | (n << 16)) & np.uint64(0x1f0000ff0000ff)
            n = (n | (n << 8)) & np.uint64(0x100f00f00f00f00f)
            n = (n | (n << 4)) & np.uint64(0x10c30c30c30c30c3)
            n = (n | (n << 2)) & np.uint64(0x1249249249249249)
            return n

        x = part1by2(int_coords[:, 0])
        y = part1by2(int_coords[:, 1])
        z = part1by2(int_coords[:, 2])
        morton_codes = x | (y << 1) | (z << 2)

        perm = np.argsort(morton_codes, kind="stable")
        inv_perm = np.empty(obj_end, dtype=np.int64)
        inv_perm[perm] = np.arange(obj_end)

        perm_torch = torch.from_numpy(perm).to(device).long()
        inv_perm_torch = torch.from_numpy(inv_perm).to(device).long()

        reordered_obj_verts = torch.index_select(vertices[:obj_end], 0, perm_torch)
        if has_controllers:
            new_vertices = torch.cat([reordered_obj_verts, vertices[obj_end:]], dim=0)
        else:
            new_vertices = reordered_obj_verts

        # Step 2: Remap spring indices
        springs_dtype = springs.dtype
        new_springs = springs.clone()
        for col in [0, 1]:
            obj_mask = springs[:, col] < obj_end
            obj_indices = springs[obj_mask, col].long()
            remapped = inv_perm_torch[obj_indices].to(springs_dtype)
            new_springs[obj_mask, col] = remapped

        # Step 3: Spatial blocking (base springs, no coloring yet)
        springs_base, rest_base, perm_base = self._reorder_springs_spatial_blocking(
            new_springs, rest_lengths, num_object_points, block_size=32
        )

        # Step 4: Edge coloring
        colors_cpu, num_colors = self._edge_color_greedy(
            springs_base,
            num_vertices=new_vertices.shape[0],
            num_object_points=num_object_points,
        )
        logger.info(f"Edge coloring: {num_colors} colors assigned")

        # Step 5: Reorder by color + spatial blocking
        springs_colored, rest_colored, spring_perm, color_offsets_cpu = \
            self._reorder_springs_spatial_blocking_by_color(
                springs_base, rest_base, colors_cpu, num_colors,
                num_object_points, block_size=32,
            )

        # Step 6: Build massnode coloring map
        massnode_coloring_map_flat = self._build_massnode_coloring_map(
            springs_colored=springs_colored,
            spring_color_offsets=color_offsets_cpu,
            perm_colored_to_base=spring_perm,
            object_massnode_single=num_object_points,
            num_spring_colors=num_colors,
            device=str(device),
        )

        return {
            "vertices": new_vertices,
            "springs_base": springs_base,
            "rest_lengths_base": rest_base,
            "springs_colored": springs_colored,
            "rest_lengths_colored": rest_colored,
            "spring_perm_base": perm_base,
            "spring_perm_colored": spring_perm,
            "num_spring_colors": num_colors,
            "spring_color_offsets": color_offsets_cpu,
            "massnode_coloring_map_flat": massnode_coloring_map_flat,
        }

    def load_model_transfer(
        self, model_path, init_controller_points, final_points, action_num, dt=5e-5
    ):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        spring_Y = spring_Y[: self.num_object_springs]

        self.init_controller_points = init_controller_points.contiguous()

        # Reconnect the springs between the controller points and the object points
        controller_points = self.init_controller_points.cpu().numpy()
        springs = self.init_springs.cpu().numpy()[: self.num_object_springs]

        rest_lengths = np.linalg.norm(
            final_points[springs[:, 0]] - final_points[springs[:, 1]], axis=1
        )

        springs = springs.tolist()
        rest_lengths = rest_lengths.tolist()
        # Update the connection between the final points and the controller points
        first_frame_controller_points = controller_points
        points = np.concatenate([final_points, first_frame_controller_points], axis=0)
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(final_points)
        pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

        # Process to get the connection distance among the controller points and object points
        # Locate the nearest object point for each controller point
        kdtree = KDTree(final_points)
        _, idx = kdtree.query(first_frame_controller_points, k=1)
        # find the distances
        distances = np.linalg.norm(
            final_points[idx] - first_frame_controller_points, axis=1
        )
        # find the indices of the top 4 controller points that are close
        controller_radius = np.ones(first_frame_controller_points.shape[0]) * 0.01
        controller_radius = distances + 0.005

        for i in range(len(first_frame_controller_points)):
            [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                first_frame_controller_points[i],
                controller_radius[i],
                30,
            )
            for j in idx:
                springs.append([self.num_all_points + i, j])
                rest_lengths.append(
                    np.linalg.norm(first_frame_controller_points[i] - points[j])
                )

        self.init_springs = torch.tensor(
            np.array(springs), dtype=torch.int32, device=cfg.device
        )

        self.init_rest_lengths = torch.tensor(
            np.array(rest_lengths), dtype=torch.float32, device=cfg.device
        )
        self.init_masses = torch.tensor(
            np.ones(len(points)), dtype=torch.float32, device=cfg.device
        )

        self.init_vertices = torch.tensor(
            points,
            dtype=torch.float32,
            device=cfg.device,
        )
        self.controller_points = torch.tensor(
            [controller_points] * action_num,
            dtype=torch.float32,
            device=cfg.device,
        )

        cfg.dt = dt
        cfg.num_substeps = round(1.0 / cfg.FPS / cfg.dt)
        cfg.collision_dist = 0.005

        self.simulator = SpringMassSystemWarpAccelerate(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
        )

        spring_Y = torch.cat(
            [
                spring_Y,
                3e4
                * torch.ones(
                    self.simulator.n_springs - self.num_object_springs,
                    dtype=torch.float32,
                    device=cfg.device,
                ),
            ]
        )

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(), collide_object_fric.detach().clone()
        )

        # ---- Preprocessing for batched rollout (Morton ordering + spring coloring) ----
        n_obj = int(self.num_all_points)
        n_ctrl = int(self.init_controller_points.shape[0])

        logger.info("Applying Morton ordering + spring coloring for batched rollout...")
        preproc = self._apply_morton_reordering_with_spring_coloring(
            self.init_vertices, self.init_springs, self.init_rest_lengths, n_obj
        )

        # Reorder spring_Y to match the base (non-colored) spring ordering
        spring_Y_reordered = spring_Y[preproc["spring_perm_base"]]

        self._single_transfer = {
            "action_num": int(action_num),
            # Raw data (before Morton reordering, for single-instance rollout)
            "init_vertices": self.init_vertices.detach().clone(),
            "init_springs": self.init_springs.detach().clone(),
            "init_rest_lengths": self.init_rest_lengths.detach().clone(),
            "init_masses": self.init_masses.detach().clone(),
            "base_spring_Y": spring_Y.detach().clone(),
            "collide_elas": collide_elas.detach().clone(),
            "collide_fric": collide_fric.detach().clone(),
            "collide_object_elas": collide_object_elas.detach().clone(),
            "collide_object_fric": collide_object_fric.detach().clone(),
            # Preprocessed data for batched rollout
            "morton_vertices": preproc["vertices"].detach().clone(),
            "springs_base": preproc["springs_base"].detach().clone(),
            "rest_lengths_base": preproc["rest_lengths_base"].detach().clone(),
            "spring_Y_base": spring_Y_reordered.detach().clone(),
            "num_spring_colors": preproc["num_spring_colors"],
            "spring_color_offsets": preproc["spring_color_offsets"],
            "massnode_coloring_map_flat": preproc["massnode_coloring_map_flat"].detach().clone(),
            "n_obj": n_obj,
            "n_ctrl": n_ctrl,
        }
        self.batch_simulator = None
        self.batch_size_loaded = None
    
    def load_model_transfer_no_acc(
        self, model_path, init_controller_points, final_points, action_num, dt=5e-5
    ):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        spring_Y = spring_Y[: self.num_object_springs]

        self.init_controller_points = init_controller_points.contiguous()

        # Reconnect the springs between the controller points and the object points
        controller_points = self.init_controller_points.cpu().numpy()
        springs = self.init_springs.cpu().numpy()[: self.num_object_springs]

        rest_lengths = np.linalg.norm(
            final_points[springs[:, 0]] - final_points[springs[:, 1]], axis=1
        )

        springs = springs.tolist()
        rest_lengths = rest_lengths.tolist()
        # Update the connection between the final points and the controller points
        first_frame_controller_points = controller_points
        points = np.concatenate([final_points, first_frame_controller_points], axis=0)
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(final_points)
        pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

        # Process to get the connection distance among the controller points and object points
        # Locate the nearest object point for each controller point
        kdtree = KDTree(final_points)
        _, idx = kdtree.query(first_frame_controller_points, k=1)
        # find the distances
        distances = np.linalg.norm(
            final_points[idx] - first_frame_controller_points, axis=1
        )
        # find the indices of the top 4 controller points that are close
        controller_radius = np.ones(first_frame_controller_points.shape[0]) * 0.01
        controller_radius = distances + 0.005

        for i in range(len(first_frame_controller_points)):
            [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                first_frame_controller_points[i],
                controller_radius[i],
                30,
            )
            for j in idx:
                springs.append([self.num_all_points + i, j])
                rest_lengths.append(
                    np.linalg.norm(first_frame_controller_points[i] - points[j])
                )

        self.init_springs = torch.tensor(
            np.array(springs), dtype=torch.int32, device=cfg.device
        )

        self.init_rest_lengths = torch.tensor(
            np.array(rest_lengths), dtype=torch.float32, device=cfg.device
        )
        self.init_masses = torch.tensor(
            np.ones(len(points)), dtype=torch.float32, device=cfg.device
        )

        self.init_vertices = torch.tensor(
            points,
            dtype=torch.float32,
            device=cfg.device,
        )
        self.controller_points = torch.tensor(
            [controller_points] * action_num,
            dtype=torch.float32,
            device=cfg.device,
        )

        cfg.dt = dt
        cfg.num_substeps = round(1.0 / cfg.FPS / cfg.dt)
        cfg.collision_dist = 0.005

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
        )

        spring_Y = torch.cat(
            [
                spring_Y,
                3e4
                * torch.ones(
                    self.simulator.n_springs - self.num_object_springs,
                    dtype=torch.float32,
                    device=cfg.device,
                ),
            ]
        )

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(), collide_object_fric.detach().clone()
        )

    def _build_batched_simulator_for_rollout(self, batch_size):
        """
        Build a batched simulator using SpringMassSystemWarpBatched.
        
        Following boba's pattern:
        - Template (shared) data: springs, masses, masks, coloring map (single-instance size)
        - Batched data: positions, velocities, controller points (all instances at SAME position)
        - No spatial offset between instances
        """
        if self._single_transfer is None:
            raise RuntimeError("load_model_transfer must be called before rollout_batch")

        info = self._single_transfer
        n_obj = info["n_obj"]
        n_ctrl = info["n_ctrl"]

        # Use Morton-reordered vertices
        morton_vertices = info["morton_vertices"]
        obj_init_vertices = morton_vertices[:n_obj]
        ctrl_init_vertices = morton_vertices[n_obj:]

        # All instances start at the SAME position (no offset)
        # Object vertices: [inst0_verts, inst1_verts, ..., instN_verts]
        batched_obj_vertices = obj_init_vertices.repeat(batch_size, 1)  # (B*n_obj, 3)

        # Controller rest location: [inst0_ctrl, inst1_ctrl, ..., instN_ctrl]
        batched_ctrl_rest = ctrl_init_vertices.repeat(batch_size, 1)  # (B*n_ctrl, 3)

        object_massnodes_total = batch_size * n_obj

        # Shared masses (single-instance sized)
        single_masses = info["init_masses"][:n_obj].contiguous()

        # Shared masks (single-instance sized)
        single_masks = None
        if self.init_masks is not None:
            single_masks = self.init_masks[:n_obj].contiguous()

        # Batched init velocities
        batched_init_velocities = None
        if self.init_velocities is not None:
            batched_init_velocities = self.init_velocities[:n_obj].repeat(batch_size, 1).contiguous()

        # Create the batched simulator
        simulator = SpringMassSystemWarpBatched(
            # Shared (template) data
            spring_base=info["springs_base"],
            rest_length_base=info["rest_lengths_base"],
            init_masses=single_masses,
            init_masks=single_masks,
            massnode_coloring_map=info["massnode_coloring_map_flat"],
            num_spring_colors=info["num_spring_colors"],
            # Batched data
            init_vertices=batched_obj_vertices,
            init_velocities=batched_init_velocities,
            controller_rest_location=batched_ctrl_rest,
            # Sizing
            object_massnodes_total=object_massnodes_total,
            object_massnodes_single=n_obj,
            controller_massnodes_single=n_ctrl,
            number_of_instance=batch_size,
            # Trained physics parameters
            spring_Y=info["spring_Y_base"],
            collide_elas=info["collide_elas"],
            collide_fric=info["collide_fric"],
            collide_object_elas=info["collide_object_elas"],
            collide_object_fric=info["collide_object_fric"],
            # Simulation config
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collision_dist=cfg.collision_dist,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            self_collision=cfg.self_collision,
        )

        # Initialize state
        simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)

        # Set up collision
        if simulator.object_collision_flag:
            simulator.create_resting_case()

        # Capture CUDA graph
        simulator.create_cuda_graph()

        self.batch_simulator = simulator
        self.batch_size_loaded = batch_size

    def rollout_batch(self, controller_points_array, visualize=False, return_trajectory=False):
        """
        Batched rollout using set_controller_interactive (boba pattern).

        Args:
            controller_points_array: [B, T, C, 3] - per-instance controller targets
                B = batch_size, T = action_num (timesteps), C = num_ctrl_points
            visualize: ignored for now
            return_trajectory: if True, return [B, T+1, N_obj, 3]; else [B, N_obj, 3]
        """
        if controller_points_array.ndim != 4:
            raise ValueError(
                f"controller_points_array should be [B, T, C, 3], got {tuple(controller_points_array.shape)}"
            )

        batch_size, action_num, num_ctrl, _ = controller_points_array.shape
        expected_ctrl = int(self.init_controller_points.shape[0])
        if num_ctrl != expected_ctrl:
            raise ValueError(f"Expected {expected_ctrl} control points, got {num_ctrl}")
        if self._single_transfer is None:
            raise RuntimeError("load_model_transfer must be called before rollout_batch")

        if visualize:
            logger.warning("rollout_batch currently ignores visualize=True")

        # Build batched simulator if needed
        if self.batch_simulator is None or self.batch_size_loaded != batch_size:
            self._build_batched_simulator_for_rollout(batch_size)

        simulator = self.batch_simulator
        n_obj = self._single_transfer["n_obj"]

        # Reset initial state (all instances at same position)
        simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)

        # Prepare batched controller targets: [T, B*C, 3]
        # controller_points_array is [B, T, C, 3]
        # We need to merge batch and ctrl dims: [T, B*C, 3]
        batched_targets = controller_points_array.permute(1, 0, 2, 3).reshape(
            action_num, batch_size * num_ctrl, 3
        ).contiguous()

        # Get initial controller rest location (same for all instances)
        # This is stored in the simulator as wp_original_control_point initially
        prev_target = wp.to_torch(
            simulator.wp_original_control_point, requires_grad=False
        ).clone()

        trajectory = None
        if return_trajectory:
            x0 = wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False)
            trajectory = [x0.view(batch_size, n_obj, 3).detach().clone()]

        for t in range(action_num):
            current_target = batched_targets[t]

            # Set interpolation endpoints: prev_target → current_target
            simulator.set_controller_interactive(prev_target, current_target)

            if simulator.object_collision_flag:
                simulator.update_collision_graph()

            # Execute one timestep via CUDA graph
            wp.capture_launch(simulator.forward_graph)
            wp.synchronize()

            x = wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
            if return_trajectory:
                trajectory.append(x.view(batch_size, n_obj, 3).detach().clone())

            # Advance state for next timestep
            simulator.set_init_state(
                simulator.wp_states[-1].wp_x,
                simulator.wp_states[-1].wp_v,
            )

            prev_target = current_target.clone()

        if return_trajectory:
            return torch.stack(trajectory, dim=1)  # [B, T+1, N_obj, 3]

        x = wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
        return x.view(batch_size, n_obj, 3)  # [B, N_obj, 3]

    def rollout(self, controller_points_array, visualize=False, return_trajectory=False):
        self.simulator.reset_idx()

        self.simulator.set_init_state(
            self.simulator.wp_init_vertices,
            self.simulator.wp_init_velocities,
            pure_inference=True,
        )

        self.simulator.controller_points = torch.cat(
            [self.init_controller_points.unsqueeze(0), controller_points_array], dim=0
        )
        self.simulator.set_controller_targets()

        if visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True)

            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis.add_geometry(coordinate)

            vis_vertices = (
                wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
                .cpu()
                .numpy()
            )

            vis_controller_points = self.simulator.controller_points[0].cpu().numpy()

            object_colors = self.object_colors.cpu().numpy()[0]
            if object_colors.shape[0] < vis_vertices.shape[0]:
                # If the object_colors is not the same as object_points, fill the colors with black
                object_colors = np.concatenate(
                    [
                        object_colors,
                        np.ones(
                            (
                                vis_vertices.shape[0] - object_colors.shape[0],
                                3,
                            )
                        )
                        * 0.3,
                    ],
                    axis=0,
                )

            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(vis_vertices)
            object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
            vis.add_geometry(object_pcd)

            target_pcd = o3d.io.read_point_cloud('experiments/log/qqtt/plan/target/target.pcd')
            target_pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * np.array(target_pcd.points).shape[0]))
            vis.add_geometry(target_pcd)

            controller_meshes = []
            prev_center = []
            if vis_controller_points is not None:
                # Use sphere mesh for each controller point
                for j in range(vis_controller_points.shape[0]):
                    origin = vis_controller_points[j]
                    origin_color = [1, 0, 0] if j in [0, 1, 4, 5, 8, 9, 12, 13] else [0, 1, 0]
                    controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.01
                    ).translate(origin)
                    controller_mesh.compute_vertex_normals()
                    controller_mesh.paint_uniform_color(origin_color)
                    controller_meshes.append(controller_mesh)
                    vis.add_geometry(controller_meshes[-1])
                    prev_center.append(origin)

            view_control = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            width, height = cfg.WH
            intrinsic = cfg.intrinsics[0]
            w2c = cfg.w2cs[0]
            intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                width * 2, height * 2, intrinsic
            )
            camera_params.intrinsic = intrinsic_parameter
            camera_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(
                camera_params, allow_arbitrary=True
            )

        action_num = controller_points_array.shape[0]
        trajectory = None
        if return_trajectory:
            x0 = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
            trajectory = [x0.detach().clone()]
        for i in range(action_num):

            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            wp.capture_launch(self.simulator.forward_graph)

            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            if return_trajectory:
                trajectory.append(x.detach().clone())
            if visualize:
                # add the visualization code here
                vis_vertices = x.cpu().numpy()

                object_pcd.points = o3d.utility.Vector3dVector(vis_vertices)
                vis.update_geometry(object_pcd)

                vis_controller_points = self.simulator.controller_points[i+1].cpu().numpy()
                if vis_controller_points is not None:
                    for j in range(vis_controller_points.shape[0]):
                        origin = vis_controller_points[j]
                        controller_meshes[j].translate(origin - prev_center[j])
                        vis.update_geometry(controller_meshes[j])
                        prev_center[j] = origin
                vis.poll_events()
                vis.update_renderer()

        x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        return x

    def rollout_no_acc(self, controller_points_array, visualize=False):
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices,
            self.simulator.wp_init_velocities,
            pure_inference=True,
        )

        self.simulator.controller_points = torch.cat(
            [self.init_controller_points.unsqueeze(0), controller_points_array], dim=0
        )

        if visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True)

            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis.add_geometry(coordinate)

            vis_vertices = (
                wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
                .cpu()
                .numpy()
            )

            vis_controller_points = self.simulator.controller_points[0].cpu().numpy()

            object_colors = self.object_colors.cpu().numpy()[0]
            if object_colors.shape[0] < vis_vertices.shape[0]:
                object_colors = np.concatenate(
                    [
                        object_colors,
                        np.ones(
                            (
                                vis_vertices.shape[0] - object_colors.shape[0],
                                3,
                            )
                        )
                        * 0.3,
                    ],
                    axis=0,
                )

            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(vis_vertices)
            object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
            vis.add_geometry(object_pcd)

            target_pcd = o3d.io.read_point_cloud('experiments/log/qqtt/plan/target/target.pcd')
            target_pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * np.array(target_pcd.points).shape[0]))
            vis.add_geometry(target_pcd)

            controller_meshes = []
            prev_center = []
            if vis_controller_points is not None:
                for j in range(vis_controller_points.shape[0]):
                    origin = vis_controller_points[j]
                    origin_color = [1, 0, 0] if j in [0, 1, 4, 5, 8, 9, 12, 13] else [0, 1, 0]
                    controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.01
                    ).translate(origin)
                    controller_mesh.compute_vertex_normals()
                    controller_mesh.paint_uniform_color(origin_color)
                    controller_meshes.append(controller_mesh)
                    vis.add_geometry(controller_meshes[-1])
                    prev_center.append(origin)

            view_control = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            width, height = cfg.WH
            intrinsic = cfg.intrinsics[0]
            w2c = cfg.w2cs[0]
            intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                width * 2, height * 2, intrinsic
            )
            camera_params.intrinsic = intrinsic_parameter
            camera_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(
                camera_params, allow_arbitrary=True
            )

        action_num = controller_points_array.shape[0]
        for i in range(action_num):
            self.simulator.set_controller_target(i + 1, pure_inference=True)

            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()

            wp.capture_launch(self.simulator.forward_graph)

            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)

            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
                pure_inference=True,
            )

            if visualize:
                vis_vertices = x.cpu().numpy()

                object_pcd.points = o3d.utility.Vector3dVector(vis_vertices)
                vis.update_geometry(object_pcd)

                vis_controller_points = self.simulator.controller_points[i + 1].cpu().numpy()
                if vis_controller_points is not None:
                    for j in range(vis_controller_points.shape[0]):
                        origin = vis_controller_points[j]
                        controller_meshes[j].translate(origin - prev_center[j])
                        vis.update_geometry(controller_meshes[j])
                        prev_center[j] = origin
                vis.poll_events()
                vis.update_renderer()

        x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
        return x
