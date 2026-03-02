"""
Batched Spring-Mass Simulator for MPC.

Ported from boba_phystwin/EdgePhystwin/qqtt/model/diff_simulator/spring_mass_warp.py
Implements MJX-style batching with:
  - Template (shared) spring connectivity, masses, masks, coloring map
  - Per-instance state arrays (positions, velocities, controller points)
  - Two-phase spring force: compute all → reduce via massnode coloring map (no atomics)
  - Instance-aware collision detection (skip cross-instance pairs)
  - CUDA graph capture for fast inference
"""

import torch
from ...utils import logger, cfg
import warp as wp

wp.init()
wp.set_device("cuda:0")


class State:
    def __init__(self, wp_init_vertices, num_control_points):
        self.wp_x = wp.zeros_like(wp_init_vertices, requires_grad=False)
        self.wp_v_before_collision = wp.zeros_like(wp_init_vertices, requires_grad=False)
        self.wp_v_before_ground = wp.zeros_like(wp_init_vertices, requires_grad=False)
        self.wp_v = wp.zeros_like(self.wp_x, requires_grad=False)
        self.wp_vertice_forces = wp.zeros_like(self.wp_x, requires_grad=False)
        # No need to compute the gradient for the control points
        self.wp_control_x = wp.zeros(
            (num_control_points,), dtype=wp.vec3, requires_grad=False
        )
        self.wp_control_v = wp.zeros_like(self.wp_control_x, requires_grad=False)

    def clear_forces(self):
        self.wp_vertice_forces.zero_()


@wp.kernel(enable_backward=False)
def copy_vec3(data: wp.array(dtype=wp.vec3), origin: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def set_control_points(
    num_substeps: int,
    original_control_point: wp.array(dtype=wp.vec3),
    target_control_point: wp.array(dtype=wp.vec3),
    step: int,
    control_x: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    t = float(step + 1) / float(num_substeps)
    control_x[tid] = (
        original_control_point[tid]
        + (target_control_point[tid] - original_control_point[tid]) * t
    )


@wp.kernel(enable_backward=False)
def eval_springs_batched_compute_all_base(
    # batched state
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),
    # BASE template springs (size = n_springs_single)
    springs: wp.array(dtype=wp.vec2i),
    inv_rest_lengths: wp.array(dtype=float),
    spring_Y_clamped: wp.array(dtype=float),
    dashpot_damping: float,
    # sizes
    object_massnode_single: int,
    controller_massnode_single: int,
    n_springs: int,
    number_of_instance: int,
    # output: per-instance per-base-spring force (size = number_of_instance * n_springs)
    spring_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    inst = tid // n_springs
    local_idx = tid - inst * n_springs  # BASE spring id in [0, n_springs)

    y = spring_Y_clamped[local_idx]

    local_idx1 = springs[local_idx][0]
    local_idx2 = springs[local_idx][1]

    # endpoint 1
    if local_idx1 >= object_massnode_single:
        ctrl1 = inst * controller_massnode_single + (local_idx1 - object_massnode_single)
        x1 = control_x[ctrl1]
        v1 = control_v[ctrl1]
    else:
        global_idx1 = inst * object_massnode_single + local_idx1
        x1 = x[global_idx1]
        v1 = v[global_idx1]

    # endpoint 2
    if local_idx2 >= object_massnode_single:
        ctrl2 = inst * controller_massnode_single + (local_idx2 - object_massnode_single)
        x2 = control_x[ctrl2]
        v2 = control_v[ctrl2]
    else:
        global_idx2 = inst * object_massnode_single + local_idx2
        x2 = x[global_idx2]
        v2 = v[global_idx2]

    inv_rest = inv_rest_lengths[local_idx]

    dis = x2 - x1
    dis_len = wp.length(dis)
    d = dis / wp.max(dis_len, 1e-6)

    spring_force = y * (dis_len * inv_rest - 1.0) * d
    v_rel = wp.dot(v2 - v1, d)
    dashpot_forces = dashpot_damping * v_rel * d

    spring_out[tid] = spring_force + dashpot_forces


@wp.kernel(enable_backward=False)
def reduce_massnode_force_from_map_instance(
    massnode_coloring: wp.array(dtype=wp.int32),
    spring_force_lookup: wp.array(dtype=wp.vec3),
    object_massnode_single: int,
    num_spring_colors: int,
    n_springs: int,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    inst = tid // object_massnode_single
    obj_node = tid - inst * object_massnode_single

    acc = wp.vec3(0.0, 0.0, 0.0)
    base_row = obj_node * num_spring_colors

    for c in range(num_spring_colors):
        sid = massnode_coloring[base_row + c]
        if sid != 0:
            sign = 1.0
            if sid < 0:
                sign = -1.0
                sid = -sid
            base_idx = sid - 1
            force = spring_force_lookup[inst * n_springs + base_idx]
            acc = acc + sign * force

    f[inst * object_massnode_single + obj_node] = acc


@wp.kernel(enable_backward=False)
def update_vel_from_force(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: float,
    reverse_factor: float,
    object_massnode_single: int,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    inst = tid // object_massnode_single
    local_idx = tid - inst * object_massnode_single

    v0 = v[tid]
    f0 = f[tid]
    m0 = masses[local_idx]

    drag_damping_factor = wp.exp(-dt * drag_damping)
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    a = all_force / m0
    v1 = v0 + a * dt
    v2 = v1 * drag_damping_factor

    v_new[tid] = v2


@wp.func
def loop(
    i: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    clamp_collide_object_elas: float,
    clamp_collide_object_fric: float,
    mass_i: float,
    offset: int,
):
    x1 = x[i]
    v1 = v[i]
    m1 = mass_i
    mask1 = masks[i - offset]

    valid_count = float(0.0)
    J_sum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(collision_number[i]):
        index = collision_indices[i][k]
        x2 = x[index]
        v2 = v[index]

        local_idx2 = index - offset
        m2 = masses[local_idx2]
        mask2 = masks[local_idx2]

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1
        if (
            mask1 != mask2
            and dis_len < collision_dist
            and wp.dot(dis, relative_v) < -1e-4
        ):
            valid_count += 1.0

            collision_normal = dis / wp.max(dis_len, 1e-6)
            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal
            impulse_n = (-(1.0 + clamp_collide_object_elas) * v_rel_n) / (
                1.0 / m1 + 1.0 / m2
            )
            v_rel_n_length = wp.length(v_rel_n)

            v_rel_t = relative_v - v_rel_n
            v_rel_t_length = wp.max(wp.length(v_rel_t), 1e-6)
            a = wp.max(
                0.0,
                1.0
                - clamp_collide_object_fric
                * (1.0 + clamp_collide_object_elas)
                * v_rel_n_length
                / v_rel_t_length,
            )
            impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)

            J = impulse_n + impulse_t
            J_sum += J

    return valid_count, J_sum


@wp.kernel(enable_backward=False)
def update_potential_collision_restmap(
    x: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    grid: wp.uint64,
    resting_collision_pairs: wp.array2d(dtype=wp.bool),
    object_massnode_single: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    inst_i = i // object_massnode_single
    local_i = i - inst_i * object_massnode_single

    x1 = x[i]
    mask1 = masks[local_i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)

    for neighbor_index in neighbors:
        if neighbor_index != i:
            inst_neighbor = neighbor_index // object_massnode_single
            # skip cross-instance pairs
            if inst_neighbor != inst_i:
                continue
            local_neighbor = neighbor_index - inst_neighbor * object_massnode_single

            if (
                resting_collision_pairs[local_i][local_neighbor] == True
                or resting_collision_pairs[local_neighbor][local_i] == True
            ):
                continue
            x2 = x[neighbor_index]
            mask2 = masks[local_neighbor]

            dis = x2 - x1
            dis_len = wp.length(dis)
            if mask1 != mask2 and dis_len < collision_dist:
                collision_indices[i][collision_number[i]] = neighbor_index
                collision_number[i] += 1


@wp.kernel(enable_backward=False)
def object_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collide_object_elas: wp.array(dtype=float),
    collide_object_fric: wp.array(dtype=float),
    collision_dist: float,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    object_massnode_single: int,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    inst = tid // object_massnode_single
    offset = inst * object_massnode_single
    local_idx = tid - offset

    v1 = v[tid]
    m1 = masses[local_idx]

    clamp_collide_object_elas = wp.clamp(collide_object_elas[0], low=0.0, high=1.0)
    clamp_collide_object_fric = wp.clamp(collide_object_fric[0], low=0.0, high=2.0)

    valid_count, J_sum = loop(
        tid,
        collision_indices,
        collision_number,
        x,
        v,
        masses,
        masks,
        collision_dist,
        clamp_collide_object_elas,
        clamp_collide_object_fric,
        m1,
        offset,
    )

    if valid_count > 0:
        J_average = J_sum / valid_count
        v_new[tid] = v1 - J_average / m1
    else:
        v_new[tid] = v1


@wp.kernel(enable_backward=False)
def build_resting_collision_pairs(
    x: wp.array(dtype=wp.vec3),
    collision_dist: float,
    grid: wp.uint64,
    resting_collision_pairs: wp.array2d(dtype=wp.bool),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]
    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    for index in neighbors:
        if index < i:
            resting_collision_pairs[i][index] = wp.bool(1)
            resting_collision_pairs[index][i] = wp.bool(1)


@wp.kernel(enable_backward=False)
def integrate_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    collide_elas: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    dt: float,
    reverse_factor: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor

    x_z = x0[2]
    v_z = v0[2]
    next_x_z = (x_z + v_z * dt) * reverse_factor

    if next_x_z < 0.0 and v_z * reverse_factor < -1e-4:
        v_normal = wp.dot(v0, normal) * normal
        v_tao = v0 - v_normal
        v_normal_length = wp.length(v_normal)
        v_tao_length = wp.max(wp.length(v_tao), 1e-6)
        clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
        clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)

        v_normal_new = -clamp_collide_elas * v_normal
        a = wp.max(
            0.0,
            1.0
            - clamp_collide_fric
            * (1.0 + clamp_collide_elas)
            * v_normal_length
            / v_tao_length,
        )
        v_tao_new = a * v_tao

        v1 = v_normal_new + v_tao_new
        toi = -x_z / v_z
    else:
        v1 = v0
        toi = 0.0

    x_new[tid] = x0 + v0 * toi + v1 * (dt - toi)
    v_new[tid] = v1


class SpringMassSystemWarpBatched:
    """
    Batched spring-mass simulator for MPC inference.

    Memory layout:
      Object state:     [inst0_node0 .. inst0_nodeN, inst1_node0 .. inst1_nodeN, ...]
      Controller state: [inst0_ctrl0 .. inst0_ctrlM, inst1_ctrl0 .. inst1_ctrlM, ...]

    Template (shared) data: springs, inv_rest_lengths, spring_Y_clamped, masses, masks,
                            massnode_coloring_map (all single-instance sized).
    Batched data: positions, velocities, controller positions, spring_force_lookup.
    """

    def __init__(
        self,
        # Shared (template) data
        spring_base,                   # (n_springs_single, 2) int32 - template springs using local indices
        rest_length_base,              # (n_springs_single,) float - template rest lengths
        init_masses,                   # (object_massnode_single,) float - shared masses
        init_masks,                    # (object_massnode_single,) int32 or None
        massnode_coloring_map,         # (object_massnode_single * num_spring_colors,) int32 flat
        num_spring_colors,             # int
        # Per-instance (already batched)
        init_vertices,                 # (object_massnode_total,3) float - batched object vertices
        init_velocities,               # (object_massnode_total,3) float or None
        controller_rest_location,      # (number_of_instance * controller_massnode_single, 3) float
        # Sizing
        object_massnodes_total,        # = number_of_instance * object_massnode_single
        object_massnodes_single,       # per-instance object node count
        controller_massnodes_single,   # per-instance controller node count
        number_of_instance,            # batch size
        # Trained physics parameters
        spring_Y,                      # (n_springs_single,) float - NON-log, raw stiffness values
        collide_elas,                  # (1,) float tensor
        collide_fric,                  # (1,) float tensor
        collide_object_elas,           # (1,) float tensor
        collide_object_fric,           # (1,) float tensor
        # Simulation config
        dt,
        num_substeps,
        dashpot_damping,
        drag_damping,
        collision_dist=0.005,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
        self_collision=False,
    ):
        logger.info(f"[SIMULATION]: Initialize Batched Spring-Mass System "
                     f"(instances={number_of_instance}, obj_nodes={object_massnodes_single}, "
                     f"ctrl_nodes={controller_massnodes_single})")
        self.device = cfg.device

        # Sizes
        self.object_massnode_single = object_massnodes_single
        self.object_massnode_total = object_massnodes_total
        self.controller_massnode_single = controller_massnodes_single
        self.number_of_instance = number_of_instance
        self.n_springs_single = int(spring_base.shape[0])
        self.n_springs_batched = self.n_springs_single * number_of_instance
        self.num_spring_colors = int(num_spring_colors)

        # Simulation parameters
        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.collision_dist = collision_dist
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        # -- Shared (template) warp arrays --
        self.wp_springs = wp.from_torch(
            spring_base.contiguous(), dtype=wp.vec2i, requires_grad=False
        )
        self.wp_inv_rest_length = wp.from_torch(
            (1.0 / rest_length_base).contiguous(), dtype=wp.float32, requires_grad=False
        )

        # Pre-clamp spring_Y (no exp at runtime, unlike the non-batched version)
        spring_Y_clamped = spring_Y.to(device=self.device, dtype=torch.float32).contiguous()
        spring_Y_clamped = spring_Y_clamped.clamp(min=self.spring_Y_min, max=self.spring_Y_max)
        self.wp_spring_Y_clamped = wp.from_torch(
            spring_Y_clamped, dtype=wp.float32, requires_grad=False
        )
        assert spring_Y_clamped.numel() == self.wp_springs.shape[0]

        self.wp_masses = wp.from_torch(
            init_masses.contiguous(), dtype=wp.float32, requires_grad=False
        )

        # Massnode coloring map for two-phase force reduction
        self.wp_massnode_coloring = wp.from_torch(
            massnode_coloring_map.to(dtype=torch.int32).contiguous(),
            dtype=wp.int32,
            requires_grad=False,
        )

        # Spring force lookup buffer (batched)
        self.wp_spring_force_lookup = wp.zeros(
            (self.n_springs_batched,), dtype=wp.vec3, requires_grad=False
        )

        # -- Collision --
        self.object_collision_flag = 0
        self.resting_collision_pairs = None
        self.wp_single_x = None
        self.collision_grid = None
        self.wp_collision_indices = None
        self.wp_collision_number = None

        self.wp_masks = None
        if self_collision:
            self.object_collision_flag = 1
            if init_masks is None:
                default_masks = torch.arange(
                    object_massnodes_single, dtype=torch.int32, device=self.device
                )
                self.wp_masks = wp.from_torch(default_masks, dtype=wp.int32, requires_grad=False)
            else:
                assert init_masks.shape[0] == object_massnodes_single
                self.wp_masks = wp.from_torch(
                    init_masks.to(dtype=torch.int32).contiguous(),
                    dtype=wp.int32,
                    requires_grad=False,
                )

            self.resting_collision_pairs = wp.zeros(
                (object_massnodes_single, object_massnodes_single),
                dtype=wp.bool,
                requires_grad=False,
            )
            self.wp_single_x = wp.empty(
                shape=(object_massnodes_single,),
                dtype=wp.vec3,
                device=self.device,
                requires_grad=False,
            )
            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.wp_collision_indices = wp.zeros(
                (self.object_massnode_total, 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.object_massnode_total,), dtype=wp.int32, requires_grad=False
            )
        elif init_masks is not None:
            # Check if there are multiple mask groups → enable collision
            if torch.unique(init_masks).shape[0] > 1:
                self.object_collision_flag = 1
                self.wp_masks = wp.from_torch(
                    init_masks.to(dtype=torch.int32).contiguous(),
                    dtype=wp.int32,
                    requires_grad=False,
                )
                self.resting_collision_pairs = wp.zeros(
                    (object_massnodes_single, object_massnodes_single),
                    dtype=wp.bool,
                    requires_grad=False,
                )
                self.wp_single_x = wp.empty(
                    shape=(object_massnodes_single,),
                    dtype=wp.vec3,
                    device=self.device,
                    requires_grad=False,
                )
                self.collision_grid = wp.HashGrid(128, 128, 128)
                self.wp_collision_indices = wp.zeros(
                    (self.object_massnode_total, 500),
                    dtype=wp.int32,
                    requires_grad=False,
                )
                self.wp_collision_number = wp.zeros(
                    (self.object_massnode_total,), dtype=wp.int32, requires_grad=False
                )

        # Collision parameters
        self.wp_collide_elas = wp.from_torch(
            collide_elas.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=False,
        )
        self.wp_collide_fric = wp.from_torch(
            collide_fric.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=False,
        )
        self.wp_collide_object_elas = wp.from_torch(
            collide_object_elas.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=False,
        )
        self.wp_collide_object_fric = wp.from_torch(
            collide_object_fric.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=False,
        )

        # -- Per-instance (batched) state --
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:object_massnodes_total].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )

        if init_velocities is None:
            self.wp_init_velocities = wp.zeros(
                shape=(self.object_massnode_total,),
                dtype=wp.vec3,
                device=self.device,
                requires_grad=False,
            )
        else:
            assert init_velocities.shape[0] == object_massnodes_total
            self.wp_init_velocities = wp.from_torch(
                init_velocities.contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        # Controller: interpolation between original and target
        self.num_controller_points = controller_rest_location.shape[0]
        self.wp_original_control_point = wp.from_torch(
            controller_rest_location.clone().contiguous(), dtype=wp.vec3, requires_grad=False
        )
        self.wp_target_control_point = wp.from_torch(
            controller_rest_location.clone().contiguous(), dtype=wp.vec3, requires_grad=False
        )

        # Preallocate substep states
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            state = State(self.wp_init_vertices, self.num_controller_points)
            self.wp_states.append(state)

    def create_cuda_graph(self):
        """Capture the step() into a CUDA graph for fast replay."""
        with wp.ScopedCapture() as forward_capture:
            self.step()
        self.forward_graph = forward_capture.graph

    def create_resting_case(self):
        """Build resting collision pairs from the initial state of a single instance."""
        # Copy first instance's positions into single_x
        wp.launch(
            copy_vec3,
            dim=self.object_massnode_single,
            inputs=[self.wp_states[0].wp_x],
            outputs=[self.wp_single_x],
        )
        self.collision_grid.build(self.wp_single_x, self.collision_dist * 5.0)
        wp.launch(
            build_resting_collision_pairs,
            dim=self.object_massnode_single,
            inputs=[
                self.wp_single_x,
                self.collision_dist,
                self.collision_grid.id,
            ],
            outputs=[self.resting_collision_pairs],
        )

    def set_controller_interactive(self, last_controller_points, current_controller_points):
        """
        Set the interpolation endpoints for controller points.

        Args:
            last_controller_points: warp array or torch tensor,
                shape (number_of_instance * controller_massnode_single, 3)
            current_controller_points: same shape
        """
        if isinstance(last_controller_points, torch.Tensor):
            last_controller_points = wp.from_torch(
                last_controller_points.contiguous(), dtype=wp.vec3, requires_grad=False
            )
        if isinstance(current_controller_points, torch.Tensor):
            current_controller_points = wp.from_torch(
                current_controller_points.contiguous(), dtype=wp.vec3, requires_grad=False
            )
        wp.launch(
            copy_vec3,
            dim=self.num_controller_points,
            inputs=[last_controller_points],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_controller_points,
            inputs=[current_controller_points],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v):
        """Set the initial state for all instances. Always writes to states[0]."""
        assert self.object_massnode_total == wp_x.shape[0]
        wp.launch(
            copy_vec3,
            dim=self.object_massnode_total,
            inputs=[wp_x],
            outputs=[self.wp_states[0].wp_x],
        )
        wp.launch(
            copy_vec3,
            dim=self.object_massnode_total,
            inputs=[wp_v],
            outputs=[self.wp_states[0].wp_v],
        )

    def update_collision_graph(self):
        """Build hash grid over all instances and find collision pairs (within-instance only)."""
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision_restmap,
            dim=self.object_massnode_total,
            inputs=[
                self.wp_states[0].wp_x,
                self.wp_masks,
                self.collision_dist,
                self.collision_grid.id,
                self.resting_collision_pairs,
                self.object_massnode_single,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        """Execute one timestep (num_substeps sub-steps)."""
        for i in range(self.num_substeps):
            self.wp_states[i].clear_forces()

            # Interpolate controller positions for this substep
            wp.launch(
                set_control_points,
                dim=self.num_controller_points,
                inputs=[
                    self.num_substeps,
                    self.wp_original_control_point,
                    self.wp_target_control_point,
                    i,
                ],
                outputs=[self.wp_states[i].wp_control_x],
            )

            # Phase 1: Compute per-spring forces
            wp.launch(
                kernel=eval_springs_batched_compute_all_base,
                dim=self.n_springs_batched,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_control_x,
                    self.wp_states[i].wp_control_v,
                    self.wp_springs,
                    self.wp_inv_rest_length,
                    self.wp_spring_Y_clamped,
                    self.dashpot_damping,
                    self.object_massnode_single,
                    self.controller_massnode_single,
                    self.n_springs_single,
                    self.number_of_instance,
                ],
                outputs=[self.wp_spring_force_lookup],
            )

            # Phase 2: Reduce per-spring forces to per-node forces
            wp.launch(
                kernel=reduce_massnode_force_from_map_instance,
                dim=self.number_of_instance * self.object_massnode_single,
                inputs=[
                    self.wp_massnode_coloring,
                    self.wp_spring_force_lookup,
                    self.object_massnode_single,
                    self.num_spring_colors,
                    self.n_springs_single,
                ],
                outputs=[self.wp_states[i].wp_vertice_forces],
            )

            if self.object_collision_flag:
                output_v = self.wp_states[i].wp_v_before_collision
            else:
                output_v = self.wp_states[i].wp_v_before_ground

            # Update velocity from forces
            wp.launch(
                kernel=update_vel_from_force,
                dim=self.object_massnode_total,
                inputs=[
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_vertice_forces,
                    self.wp_masses,
                    self.dt,
                    self.drag_damping,
                    self.reverse_factor,
                    self.object_massnode_single,
                ],
                outputs=[output_v],
            )

            if self.object_collision_flag:
                wp.launch(
                    kernel=object_collision,
                    dim=self.object_massnode_total,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v_before_collision,
                        self.wp_masses,
                        self.wp_masks,
                        self.wp_collide_object_elas,
                        self.wp_collide_object_fric,
                        self.collision_dist,
                        self.wp_collision_indices,
                        self.wp_collision_number,
                        self.object_massnode_single,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )

            # Integrate with ground collision
            wp.launch(
                kernel=integrate_ground_collision,
                dim=self.object_massnode_total,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v_before_ground,
                    self.wp_collide_elas,
                    self.wp_collide_fric,
                    self.dt,
                    self.reverse_factor,
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )
