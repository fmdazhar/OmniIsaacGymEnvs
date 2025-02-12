# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

import numpy as np
import torch
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from deformableRL.tasks.base.rl_task import RLTask
from deformableRL.robots.articulations.anymal import Anymal
from deformableRL.robots.articulations.views.anymal_view import AnymalView
from deformableRL.tasks.utils.anymal_terrain_generator import *
from deformableRL.utils.terrain_utils.terrain_utils import *
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdLux, Sdf, Gf, UsdShade, Vt

from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.physx.scripts import physicsUtils, particleUtils


class AnymalTerrainTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.height_samples = None
        self.depression_details = None
        self.custom_origins = False
        self.init_done = False
        self._env_spacing = 0.0

        self._num_observations = 188
        self._num_actions = 12

        self.update_config(sim_config)

        RLTask.__init__(self, name, env)

        self.height_points = self.init_height_points()
        self.measured_heights = None
        # joint positions offsets
        self.default_dof_pos = torch.zeros(
            (self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False
        )
        # reward episode sums
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {
            "lin_vel_xy": torch_zeros(),
            "lin_vel_z": torch_zeros(),
            "ang_vel_z": torch_zeros(),
            "ang_vel_xy": torch_zeros(),
            "orient": torch_zeros(),
            "torques": torch_zeros(),
            "joint_acc": torch_zeros(),
            "base_height": torch_zeros(),
            "air_time": torch_zeros(),
            "collision": torch_zeros(),
            "action_rate": torch_zeros(),
            "hip": torch_zeros(),
            "fallen_over": torch_zeros(),
            "dof_pos_limits": torch_zeros(),
            "termination": torch_zeros(),
        }

        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self._task_cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        #reward
        self.base_height_target = self._task_cfg["env"]["learn"]["baseHeightTarget"]
        self.soft_dof_pos_limit = self._task_cfg["env"]["learn"]["softDofPositionLimit"]
        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["terminalReward"]
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["ang_vel_xy"] = self._task_cfg["env"]["learn"]["angularVelocityXYRewardScale"]
        self.rew_scales["orient"] = self._task_cfg["env"]["learn"]["orientationRewardScale"]
        self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["joint_vel"] = self._task_cfg["env"]["learn"]["jointVelRewardScale"]
        self.rew_scales["base_height"] = self._task_cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["hip"] = self._task_cfg["env"]["learn"]["hipRewardScale"]
        self.rew_scales["fallen_over"] = self._task_cfg["env"]["learn"]["fallenOverRewardScale"]
        self.rew_scales["collision"] = self._task_cfg["env"]["learn"]["collisionRewardScale"]
        self.rew_scales["air_time"] = self._task_cfg["env"]["learn"]["airTimeRewardScale"]
        self.rew_scales["dof_pos_limits"] = self._task_cfg["env"]["learn"]["dofPosLimitsRewardScale"]

        # command ranges
        self.vel_curriculum = self._task_cfg["env"]["terrain"]["VelocityCurriculum"]
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        self.limit_vel_x = self._task_cfg["env"]["limitCommandVelocityRanges"]["linear_x"]
        self.limit_vel_y = self._task_cfg["env"]["limitCommandVelocityRanges"]["linear_y"]
        self.limit_vel_yaw = self._task_cfg["env"]["limitCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self.base_threshold = 0.2
        self.thigh_threshold = 0.1

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]

        self._task_cfg["sim"]["default_physics_material"]["static_friction"] = self._task_cfg["env"]["terrain"][
            "staticFriction"
        ]
        self._task_cfg["sim"]["default_physics_material"]["dynamic_friction"] = self._task_cfg["env"]["terrain"][
            "dynamicFriction"
        ]
        self._task_cfg["sim"]["default_physics_material"]["restitution"] = self._task_cfg["env"]["terrain"][
            "restitution"
        ]

        self._task_cfg["sim"]["add_ground_plane"] = False

        self._particle_cfg = self._task_cfg["env"]["particles"]

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self._task_cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self._task_cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self._task_cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0.0  # commands
        noise_vec[12:24] = self._task_cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[24:36] = self._task_cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[36:176] = (
            self._task_cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale
        )
        noise_vec[176:188] = 0.0  # previous actions
        return noise_vec

    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor(
            [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device, requires_grad=False
        )  # 10-50cm on each side
        x = 0.1 * torch.tensor(
            [-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False
        )  # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _create_trimesh(self, create_mesh=True):
        self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([-self.terrain.border_size, -self.terrain.border_size, 0.0])
        if create_mesh:
            add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["lin_vel_xy"][env_ids]) / self.max_episode_length_s > 0.8 * self.rew_scales["lin_vel_xy"]:
            self.command_x_range[0] = np.clip(self.command_x_range[0] - 0.2, -self.limit_vel_x[0], 0.).item()
            self.command_x_range[1] = np.clip(self.command_x_range[1] + 0.2, 0., self.limit_vel_x[1]).item()

            # Increase the range of commands for y
            self.command_y_range[0] = np.clip(self.command_y_range[0] - 0.2, -self.limit_vel_y[0], 0.).item()
            self.command_y_range[1] = np.clip(self.command_y_range[1] + 0.2, 0., self.limit_vel_y[1]).item()
        
        if torch.mean(self.episode_sums["ang_vel_z"][env_ids]) / self.max_episode_length_s > 0.8 * self.rew_scales["ang_vel_z"]:
        # Increase the range of commands for yaw
            self.command_yaw_range[0] = np.clip(self.command_yaw_range[0] - 0.2, -self.limit_vel_yaw[0], 0.).item()
            self.command_yaw_range[1] = np.clip(self.command_yaw_range[1] + 0.2, 0., self.limit_vel_yaw[1]).item()

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        simulation_context = SimulationContext.instance()
        simulation_context.get_physics_context().enable_gpu_dynamics(True)
        simulation_context.get_physics_context().set_broadphase_type("GPU")
        self.get_terrain()
        self.get_anymal()
        super().set_up_scene(scene, collision_filter_global_paths=["/World/terrain"])
        self._anymals = AnymalView(
            prim_paths_expr="/World/envs/.*/anymal", name="anymal_view", track_contact_forces=True
        )
        if self._particle_cfg["enabled"]:
            self.create_particle_system()
            scene.add(self._particle_system)
        scene.add(self._anymals)
        scene.add(self._anymals._thigh)
        scene.add(self._anymals._shank)
        scene.add(self._anymals._foot)
        scene.add(self._anymals._base)

    def initialize_views(self, scene):
        # initialize terrain variables even if we do not need to re-create the terrain mesh
        self.get_terrain(create_mesh=False)

        super().initialize_views(scene)
        if scene.object_exists("anymal_view"):
            scene.remove_object("anymal_view", registry_only=True)
        if scene.object_exists("thigh_view"):
            scene.remove_object("thigh_view", registry_only=True)
        if scene.object_exists("shank_view"):
            scene.remove_object("shank_view", registry_only=True)
        if scene.object_exists("foot_view"):
            scene.remove_object("foot_view", registry_only=True)
        if scene.object_exists("base_view"):
            scene.remove_object("base_view", registry_only=True)
        if scene.object_exists("particle_view"):
            scene.remove_object("particle_view", registry_only=True)

        self._anymals = AnymalView(
            prim_paths_expr="/World/envs/.*/anymal", name="anymal_view", track_contact_forces=True
        )

        if self._particle_cfg["enabled"]:
            self.create_particle_system()
            scene.add(self._particle_systems)
        scene.add(self._anymals)
        scene.add(self._anymals._thigh)
        scene.add(self._anymals._shank)
        scene.add(self._anymals._foot)
        scene.add(self._anymals._base)

    def get_terrain(self, create_mesh=True):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        if not self.curriculum:
            self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(
            0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"] + 1, (self.num_envs,), device=self.device
        )
        self.terrain_types = torch.randint(
            0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device
        )
        self._create_trimesh(create_mesh=create_mesh)
        if self._particle_cfg["enabled"]:
            self.depression_details = self.terrain.depression_details
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

    def get_anymal(self):
        anymal_translation = torch.tensor([0.0, 0.0, 0.66])
        anymal_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        anymal = Anymal(
            prim_path=self.default_zero_env_path + "/anymal",
            name="anymal",
            translation=anymal_translation,
            orientation=anymal_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "anymal", get_prim_at_path(anymal.prim_path), self._sim_config.parse_actor_config("anymal")
        )
        anymal.set_anymal_properties(self._stage, anymal.prim)
        anymal.prepare_contacts(self._stage, anymal.prim)

        self.dof_names = anymal.dof_names
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle


    def post_reset(self):
        self.base_init_state = torch.tensor(
            self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # initialize some data used later on
        self.up_axis_idx = 2
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self._task_cfg)
        self.commands = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            device=self.device,
            requires_grad=False,
        )
        self.gravity_vec = torch.tensor(
            get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )

        feet_names = ['left_front','left_rear', 'right_front', 'right_rear']  # Example, adjust as needed
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)

        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.last_dof_vel = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
        self.num_dof = self._anymals.num_dof
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

        self.thigh_pos = torch.zeros((self.num_envs * 4, 3), dtype=torch.float, device=self.device)
        self.thigh_quat = torch.zeros((self.num_envs * 4, 4), dtype=torch.float, device=self.device)
        self.foot_contact_forces = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.thigh_contact_forces = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.shank_contact_forces = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)


        # Get joint limits
        dof_limits = self._anymals.get_dof_limits()
        lower_limits = dof_limits[0, :, 0]    
        upper_limits = dof_limits[0, :, 1]    
        midpoint = 0.5 * (lower_limits + upper_limits)
        limit_range = upper_limits - lower_limits
        soft_lower_limits = midpoint - 0.5 * limit_range * self.soft_dof_pos_limit
        soft_upper_limits = midpoint + 0.5 * limit_range * self.soft_dof_pos_limit
        self.anymal_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.anymal_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.anymal_dof_soft_lower_limits = soft_lower_limits.to(device=self._device)
        self.anymal_dof_soft_upper_limits = soft_upper_limits.to(device=self._device)

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        self.init_done = True

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        if self.vel_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        self.update_terrain_level(env_ids)
        self.base_pos[env_ids] = self.base_init_state[0:3]
        self.base_pos[env_ids, 0:3] += self.env_origins[env_ids]
        self.base_pos[env_ids, 0:2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        self.base_quat[env_ids] = self.base_init_state[3:7]
        self.base_velocities[env_ids] = self.base_init_state[7:]

        self._anymals.set_world_poses(
            positions=self.base_pos[env_ids].clone(), orientations=self.base_quat[env_ids].clone(), indices=indices
        )
        self._anymals.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)
        self._anymals.set_joint_positions(positions=self.dof_pos[env_ids].clone(), indices=indices)
        self._anymals.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), indices=indices)

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(
            1
        )  # set small commands to zero

        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        self.extras["episode"]["max_command_x"] = self.command_x_range[1]
        self.extras["episode"]["max_command_y"] = self.command_y_range[1]
        self.extras["episode"]["max_command_yaw"] = self.command_yaw_range[1]

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # do not change on initial reset
            return
        root_pos, _ = self._anymals.get_world_poses(clone=False)
        distance = torch.norm(root_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (
            distance < torch.norm(self.commands[env_ids, :2]) * self.max_episode_length_s * 0.25
        )
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def refresh_dof_state_tensors(self):
        self.dof_pos = self._anymals.get_joint_positions(clone=False)
        self.dof_vel = self._anymals.get_joint_velocities(clone=False)

    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._anymals.get_world_poses(clone=False)
        self.base_velocities = self._anymals.get_velocities(clone=False)
        self.thigh_pos, self.thigh_quat = self._anymals._thigh.get_world_poses(clone=False)
    
    def refresh_net_contact_force_tensors(self):
        self.foot_contact_forces = self._anymals._foot.get_net_contact_forces(dt=self.dt,clone=False).view(self._num_envs, 4, 3)
        self.thigh_contact_forces = self._anymals._thigh.get_net_contact_forces(dt=self.dt,clone=False).view(self._num_envs, 4, 3)
        self.shank_contact_forces = self._anymals._shank.get_net_contact_forces(dt=self.dt,clone=False).view(self._num_envs, 4, 3)


    def pre_physics_step(self, actions):
        if not self.world.is_playing():
            return

        self.actions = actions.clone().to(self.device)
        for i in range(self.decimation):
            if self.world.is_playing():
                torques = torch.clip(
                    self.Kp * (self.action_scale * self.actions + self.default_dof_pos - self.dof_pos)
                    - self.Kd * self.dof_vel,
                    -80.0,
                    80.0,
                )
                self._anymals.set_joint_efforts(torques)
                self.torques = torques
                SimulationContext.step(self.world, render=False)
                # simulation_context = SimulationContext.instance()
                # print("Rendering dt:", simulation_context.get_rendering_dt())
                # print("Physics dt:", simulation_context.get_physics_dt())                
                self.refresh_dof_state_tensors()

    def post_physics_step(self):
        self.progress_buf[:] += 1

        if self.world.is_playing():

            self.refresh_dof_state_tensors()
            self.refresh_body_state_tensors()
            self.refresh_net_contact_force_tensors()

            self.common_step_counter += 1
            if self.common_step_counter % self.push_interval == 0:
                self.push_robots()

            # prepare quantities
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

            self.check_termination()
            self.get_states()
            self.calculate_metrics()

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()
            if self.add_noise:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

            self.last_actions[:] = self.actions[:]
            self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def push_robots(self):
        self.base_velocities[:, 0:2] = torch_rand_float(
            -1.0, 1.0, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self._anymals.set_velocities(self.base_velocities)

    def check_termination(self):
        self.timeout_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.timeout_buf),
            torch.zeros_like(self.timeout_buf),
        )
        
        self.has_fallen = (torch.norm(self._anymals._base.get_net_contact_forces(clone=False), dim=1) > 1.0) 
        self.reset_buf = self.has_fallen.clone()
        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)

    def calculate_metrics(self):
        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        # other base velocity penalties
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # orientation penalty
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        # base height penalty
        rew_base_height = torch.square(self.base_pos[:, 2] - self.base_height_target) * self.rew_scales["base_height"]

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]
        
        # Penalize dof velocities
        rew_dof_vel = torch.sum(torch.square(self.dof_vel), dim=1) * self.rew_scales["joint_vel"]

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel) / self.dt, dim=1) * self.rew_scales["joint_acc"]

        # Penalize collisions on selected bodies
        thigh_contact = (
            torch.norm(self.thigh_contact_forces, dim=-1)
            > 0.1
        )
        shank_contact = (torch.norm(self.shank_contact_forces, dim=-1) > 0.1)
        total_contact = thigh_contact + shank_contact
        rew_collision = torch.sum(total_contact, dim=-1) * self.rew_scales["collision"]
            
        # action rate penalty
        rew_action_rate = (
            torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        )

        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.anymal_dof_soft_lower_limits).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.anymal_dof_soft_upper_limits).clip(min=0.)
        rew_dof_pos_limits = torch.sum(out_of_limits, dim=1) * self.rew_scales["dof_pos_limits"]

        # Increment feet_air_time for each step the foot is not in contact
        contact = self.foot_contact_forces[:, self.feet_indices, 2] > 1.0  # Placeholder for contact detection, adjust threshold as needed
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt  # Assuming self.dt is the timestep duration
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        rew_airTime *= self.rew_scales["air_time"]
        self.feet_air_time *= ~self.contact_filt

        # cosmetic penalty for hip motion
        rew_hip = (
            torch.sum(torch.abs(self.dof_pos[:, 0:4] - self.default_dof_pos[:, 0:4]), dim=1) * self.rew_scales["hip"]
        )

        # fallen over penalty
        rew_fallen_over = self.has_fallen * self.rew_scales["fallen_over"]

        # total reward
        self.rew_buf = (
            rew_lin_vel_xy
            + rew_ang_vel_z
            + rew_lin_vel_z
            + rew_ang_vel_xy
            + rew_orient
            + rew_base_height
            + rew_torque
            + rew_joint_acc
            + rew_action_rate
            + rew_dof_pos_limits
            + rew_collision
            + rew_airTime
        )
        self.rew_buf = torch.clip(self.rew_buf, min=0.0, max=None)

        # add termination reward
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["base_height"] += rew_base_height
        self.episode_sums["dof_pos_limits"] += rew_dof_pos_limits
        self.episode_sums["collision"] += rew_collision
        self.episode_sums["air_time"] += rew_airTime


    def get_observations(self):
        self.measured_heights = self.get_heights()
        heights = (
            torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.height_meas_scale
        )
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.lin_vel_scale,
                self.base_ang_vel * self.ang_vel_scale,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.dof_pos * self.dof_pos_scale,
                self.dof_vel * self.dof_vel_scale,
                heights,
                self.actions,
            ),
            dim=-1,
        )

    def get_ground_heights_below_thigh(self):
        points = self.thigh_pos.reshape(self.num_envs, 4, 3)
        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

    def get_ground_heights_below_base(self):
        points = self.base_pos.reshape(self.num_envs, 1, 3)
        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

    def get_heights(self, env_ids=None):
        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]
            ) + (self.base_pos[env_ids, 0:3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.base_pos[:, 0:3]
            ).unsqueeze(1)

        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

    def create_particle_system(self):
        # Define paths
        default_prim_path = "/World"
        particle_system_path = default_prim_path + "/particleSystem"

        # Create the particle system
        self._particle_system = ParticleSystem(
            prim_path=particle_system_path,
            particle_system_enabled=True,
            simulation_owner="/physicsScene",
            rest_offset=self._particle_cfg["system1"]["particle_system_rest_offset"],
            contact_offset=self._particle_cfg["system1"]["particle_system_contact_offset"],
            solid_rest_offset=self._particle_cfg["system1"]["particle_system_solid_rest_offset"],
            particle_contact_offset=self._particle_cfg["system1"]["particle_system_particle_contact_offset"],
            max_velocity=self._particle_cfg["system1"]["particle_system_max_velocity"],
            max_neighborhood=self._particle_cfg["system1"]["particle_system_max_neighborhood"],
            solver_position_iteration_count=self._particle_cfg["system1"]["particle_system_solver_position_iteration_count"],
            enable_ccd=self._particle_cfg["system1"]["particle_system_enable_ccd"],
            # max_depenetration_velocity=self._particle_cfg[particle_system_max_depenetration_velocity],
        )

        # Create the particle prototype
        self.create_pbd_material()

        # Create particle grid
        self.create_particle_grid()

        # if self._particle_cfg["use_mesh_sampler"]:
        #     # Create particles from mesh
        #     self.create_particles_from_mesh()
        # else:
        #     # Create particle grid
        #     self.create_particle_grid()
            
    def create_pbd_material(self):
        ps = PhysxSchema.PhysxParticleSystem.Get(self._stage, Sdf.Path("/World/particleSystem"))
        # Setting up a material density, will be used by both ref & cand because of shared particle system
        pbd_material_path = Sdf.Path("/World/pbdmaterial")
        particleUtils.add_pbd_particle_material(
            self._stage,
            pbd_material_path,
            friction=self._particle_cfg["system1"]["pbd_material_friction"],
            particle_friction_scale=self._particle_cfg["system1"]["pbd_material_particle_friction_scale"],
            adhesion=self._particle_cfg["system1"]["pbd_material_adhesion"],
            particle_adhesion_scale=self._particle_cfg["system1"]["pbd_material_particle_adhesion_scale"],
            adhesion_offset_scale=self._particle_cfg["system1"]["pbd_material_adhesion_offset_scale"],
            density=self._particle_cfg["system1"]["pbd_material_density"],
        )
        physicsUtils.add_physics_material_to_prim(self._stage, ps.GetPrim(), pbd_material_path)

    def create_particle_grid(self):

        for index, depression in enumerate(self.depression_details):

            print("Creating particle grid for depression: ", index)
            # # Define paths
            # default_prim_path = "/World"
            # default_prim_path = Sdf.Path(default_prim_path)
            # particle_system_path = default_prim_path.AppendChild("particleSystem")

            # Define paths
            default_prim_path = "/World"
            particle_system_path = default_prim_path + "/particleSystem"
            # Define the position and size of the particle grid from config
            print("Depression details: ", depression)
            x_position, y_position, z_position, size , depth, type = depression

            lower = Gf.Vec3f(x_position, y_position, z_position + 0.1)

            solid_rest_offset = self._particle_cfg["system1"]["particle_system_solid_rest_offset"]
            particle_spacing = 2.5 * solid_rest_offset

            num_samples_x = int(size / particle_spacing) + 1
            num_samples_y = int(size / particle_spacing) + 1
            num_samples_z = int(-depth / particle_spacing) + 1

            # Jitter factor from config (as a fraction of particle_spacing)
            jitter_factor = self._particle_cfg["system1"]["particle_grid_jitter_factor"] * particle_spacing

            position = [Gf.Vec3f(0.0)] * num_samples_x * num_samples_y * num_samples_z
            uniform_particle_velocity = Gf.Vec3f(0.0)
            ind = 0
            x = lower[0]
            y = lower[1]
            z = lower[2]
            for i in range(num_samples_x):
                for j in range(num_samples_y):
                    for k in range(num_samples_z):
                        jitter_x = random.uniform(-jitter_factor, jitter_factor)
                        jitter_y = random.uniform(-jitter_factor, jitter_factor)
                        jitter_z = random.uniform(-jitter_factor, jitter_factor)

                        # Apply jitter to the position
                        jittered_x = x + jitter_x
                        jittered_y = y + jitter_y
                        jittered_z = z + jitter_z
                        position[ind] = Gf.Vec3f(jittered_x, jittered_y, jittered_z)
                        ind += 1
                        z = z + particle_spacing
                    z = lower[2]
                    y = y + particle_spacing
                y = lower[1]
                x = x + particle_spacing
            positions, velocities = (position, [uniform_particle_velocity] * len(position))
            widths = [2 * solid_rest_offset * 0.5] * len(position)
            print(f"Number of particles created: {len(position)}")


            # Define particle point instancer path
            particle_system_path = Sdf.Path(particle_system_path)
            particle_point_instancer_path = particle_system_path.AppendChild("particles_grid_{}".format(index))

            # # Define particle point instancer path
            # default_prim_path = "/World"
            # particle_system_path = default_prim_path + "/particleSystem"
            # particle_point_instancer_path = Sdf.Path(particle_system_path + "/particles1")

            # Add the particle set to the point instancer
            particleUtils.add_physx_particleset_pointinstancer(
                self._stage,
                particle_point_instancer_path,
                Vt.Vec3fArray(positions),
                Vt.Vec3fArray(velocities),
                particle_system_path,
                self_collision=self._particle_cfg["system1"]["particle_grid_self_collision"],
                fluid=self._particle_cfg["system1"]["particle_grid_fluid"],
                particle_group=self._particle_cfg["system1"]["particle_grid_particle_group"],
                particle_mass=self._particle_cfg["system1"]["particle_grid_particle_mass"],
                density=self._particle_cfg["system1"]["particle_grid_density"],
            )

            # Configure particle prototype
            particle_prototype_sphere = UsdGeom.Sphere.Get(
                self._stage, particle_point_instancer_path.AppendChild("particlePrototype0")
            )
            particle_prototype_sphere.CreateRadiusAttr().Set(solid_rest_offset)

    def create_particles_from_mesh(self):
        """
        Creates particles from the specified mesh.
        
        Args:

        """
        default_prim_path = "/World"
        particle_system_path = default_prim_path + "/particleSystem"
        particle_set_path = default_prim_path + "/particles"
        # create a cube mesh that shall be sampled:
        cube_mesh_path = Sdf.Path(omni.usd.get_stage_next_free_path(self._stage, "/Cube", True))
        cube_resolution = (
            2  # resolution can be low because we'll sample the surface / volume only irrespective of the vertex count
        )
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform", prim_type="Cube", u_patches=cube_resolution, v_patches=cube_resolution, select_new_prim=False
        )        
        cube_mesh = UsdGeom.Mesh.Get(self._stage, Sdf.Path(cube_mesh_path))

        physicsUtils.setup_transform_as_scale_orient_translate(cube_mesh)

        physicsUtils.set_or_add_translate_op(
        cube_mesh, 
        Gf.Vec3f(
            self._particle_cfg["system1"]["particle_x_position"], 
            self._particle_cfg["system1"]["particle_y_position"], 
            self._particle_cfg["system1"]["particle_z_position"]
            )
        )
        physicsUtils.set_or_add_scale_op(
            cube_mesh, 
            Gf.Vec3f(
                self._particle_cfg["system1"]["particle_scale_x"], 
                self._particle_cfg["system1"]["particle_scale_y"], 
                self._particle_cfg["system1"]["particle_scale_z"]
            )
        )
        
        # Calculate sampling distance based on particle system parameters
        solid_rest_offset =  self._particle_cfg["system1"]["particle_system_solid_rest_offset"]
        particle_sampler_distance = 2.5 * solid_rest_offset

        # Apply particle sampling on the mesh
        sampling_api = PhysxSchema.PhysxParticleSamplingAPI.Apply(cube_mesh.GetPrim())
        # sampling_api.CreateSamplingDistanceAttr().Set(particle_sampler_distance)
        sampling_api.CreateMaxSamplesAttr().Set(5e5)
        sampling_api.CreateVolumeAttr().Set(True)  # Set to True if sampling volume, False for surface

        cube_mesh.CreateVisibilityAttr("invisible")

        # create particle set
        points = UsdGeom.Points.Define(self._stage, particle_set_path)
        points.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(71.0 / 255.0, 125.0 / 255.0, 1.0)]))
        particleUtils.configure_particle_set(points.GetPrim(), particle_system_path, 
        self._particle_cfg["system1"]["particle_grid_self_collision"], self._particle_cfg["system1"]["particle_grid_fluid"], 
        self._particle_cfg["system1"]["particle_grid_particle_group"], self._particle_cfg["system1"]["particle_grid_particle_mass"], self._particle_cfg["system1"]["particle_grid_density"])

        # reference the particle set in the sampling api
        sampling_api.CreateParticlesRel().AddTarget(particle_set_path)


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def get_axis_params(value, axis_idx, x_value=0.0, dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index."""
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


