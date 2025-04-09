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
import omni
import carb

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.prims import get_prim_at_path, find_matching_prim_paths, is_prim_path_valid
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.a1 import A1
from omniisaacgymenvs.robots.articulations.views.a1_view import A1View
from omni.isaac.core.prims import RigidPrimView
from omniisaacgymenvs.tasks.utils.anymal_terrain_generator import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdLux, Sdf, Gf, UsdShade, Vt

from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.core.prims.soft.particle_system_view import ParticleSystemView
from omni.physx.scripts import physicsUtils, particleUtils
import omni.kit.commands
class AnymalTerrainTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.height_samples = None
        self.terrain_details = None
        self.custom_origins = False
        self.init_done = False
        self._env_spacing = 0.0
        self.update_config(sim_config)

        self._num_actions = 12
        self._num_proprio = 48 #188 #3 + 3 + 3 + 3 + 12 + 12 + 140 + 12
        self._num_privileged_observations = None
        self._num_priv = 28 #4 + 4 + 4 + 1 + 3 + 12 
        self._obs_history_length = 10  # e.g., 3, 5, etc.
        self._num_obs_history = self._obs_history_length * self._num_proprio
        # If measure_heights is True, we add that to the final observation dimension
        if self.measure_heights:
            self.num_height_points = 140
        else:
            self.num_height_points = 0

        # Then the final observation dimension is:
        self._num_observations = (self._obs_history_length * self._num_proprio) \
                                + self._num_priv \
                                + self._num_proprio \
                                + self.num_height_points

        RLTask.__init__(self, name, env)

        self._num_train_envs = self.num_envs

        if self.measure_heights:
            self.height_points = self.init_height_points()
        self.measured_heights = None
        self.debug_heights = False

        # Initialize dictionaries to track created particle systems and materials
        self.created_particle_systems = {}
        self.created_materials = {}
        self.particle_instancers_by_level = {}
        self._terrains_by_level = {}  # dictionary: level -> (tensor of row indices)
        self.total_particles = 0    # Initialize a counter for total particles

        # joint positions offsets
        self.default_dof_pos = torch.zeros(
            (self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False
        )

        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # normalization
        self.clip_obs = self._task_cfg["env"].get("clipObservations", np.Inf)
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self._task_cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward params
        self.base_height_target = self._task_cfg["env"]["learn"]["baseHeightTarget"]
        self.soft_dof_pos_limit = self._task_cfg["env"]["learn"]["softDofPositionLimit"]
        self.soft_dof_vel_limit = self._task_cfg["env"]["learn"]["softDofVelLimit"]
        self.soft_torque_limit = self._task_cfg["env"]["learn"]["softTorqueLimit"]
        self.tracking_sigma = self._task_cfg["env"]["learn"]["trackingSigma"]
        self.max_contact_force = self._task_cfg["env"]["learn"]["maxContactForce"]
        self.only_positive_rewards = self._task_cfg["env"]["learn"]["onlyPositiveRewards"]
        
        # reward scales
        self.reward_scales = self._task_cfg["env"]["learn"]["scales"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        self.limit_vel_x = self._task_cfg["env"]["limitCommandVelocityRanges"]["linear_x"]
        self.limit_vel_y = self._task_cfg["env"]["limitCommandVelocityRanges"]["linear_y"]
        self.limit_vel_yaw = self._task_cfg["env"]["limitCommandVelocityRanges"]["yaw"]
        self.vel_curriculum = self._task_cfg["env"]["terrain"]["VelocityCurriculum"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        #randomization
        self.friction_range = self._task_cfg["env"]["randomizationRanges"]["frictionRange"]
        self.restitution_range = self._task_cfg["env"]["randomizationRanges"]["restitutionRange"]
        self.added_mass_range = self._task_cfg["env"]["randomizationRanges"]["addedMassRange"]
        self.com_displacement_range = self._task_cfg["env"]["randomizationRanges"]["comDisplacementRange"]
        self.motor_strength_range = self._task_cfg["env"]["randomizationRanges"]["motorStrengthRange"]
        self.motor_offset_range = self._task_cfg["env"]["randomizationRanges"]["motorOffsetRange"]
        self.Kp_factor_range = self._task_cfg["env"]["randomizationRanges"]["KpFactorRange"]
        self.Kd_factor_range = self._task_cfg["env"]["randomizationRanges"]["KdFactorRange"]
        self.gravity_range = self._task_cfg["env"]["randomizationRanges"]["gravityRange"]

        # other
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.pbd_randomize_interval = int(self._task_cfg["env"]["randomizationRanges"]["material_randomization"]["particles"]["interval"] / self.dt + 0.5)
        self.gravity_randomize_interval = int(self._task_cfg["env"]["randomizationRanges"]["gravityRandIntervalSecs"] / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self.measure_heights = self._task_cfg["env"]["terrain"]["measureHeights"]

        self.base_threshold = 0.2
        self.thigh_threshold = 0.1

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
        terrain_types = self._task_cfg["env"]["terrain"].get("terrain_types", [])
        has_particles = any(tt.get("particle_present", False) for tt in terrain_types)
        self._particles_active = has_particles and self._particle_cfg.get("enabled", False) and not self.curriculum

    def _get_noise_scale_vec(self, cfg):

        noise_vec = torch.zeros(self._num_proprio, device=self.device, dtype=torch.float)
        self.add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self._task_cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self._task_cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self._task_cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0.0  # commands
        noise_vec[12:24] = self._task_cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[24:36] = self._task_cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[36:48] = 0.0  # previous actions
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

    def init_particle_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor(
            [-7,-6,-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], device=self.device, requires_grad=False
        )  # 10-50cm on each side
        x = 0.1 * torch.tensor(
            [-10,-9,-8, -7, -6, -5, -4, -3, -2,-1, 0, 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10], device=self.device, requires_grad=False
        )  # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.num_particle_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_particle_height_points, 3, device=self.device, requires_grad=False)
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
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length_s > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_x_range[0] = np.clip(self.command_x_range[0] - 0.2, -self.limit_vel_x[0], 0.).item()
            self.command_x_range[1] = np.clip(self.command_x_range[1] + 0.2, 0., self.limit_vel_x[1]).item()

            # Increase the range of commands for y
            self.command_y_range[0] = np.clip(self.command_y_range[0] - 0.2, -self.limit_vel_y[0], 0.).item()
            self.command_y_range[1] = np.clip(self.command_y_range[1] + 0.2, 0., self.limit_vel_y[1]).item()
        
        if torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length_s > 0.8 * self.reward_scales["tracking_ang_vel"]:
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
        self._anymals = A1View(
            prim_paths_expr="/World/envs/.*/a1", name="a1_view", track_contact_forces=True
        )
        if self._particles_active:
            self.create_particle_systems()
            self.particle_system_view = ParticleSystemView(prim_paths_expr="/World/particleSystem/*")
            scene.add(self.particle_system_view)
        scene.add(self._anymals)
        scene.add(self._anymals._thigh)
        scene.add(self._anymals._base)
        scene.add(self._anymals._foot)
        scene.add(self._anymals._calf)


    def initialize_views(self, scene):
        # initialize terrain variables even if we do not need to re-create the terrain mesh
        self.get_terrain(create_mesh=False)

        super().initialize_views(scene)
        if scene.object_exists("a1_view"):
            scene.remove_object("a1_view", registry_only=True)
        if scene.object_exists("thigh_view"):
            scene.remove_object("thigh_view", registry_only=True)
        if scene.object_exists("base_view"):
            scene.remove_object("base_view", registry_only=True)
        if scene.object_exists("foot_view"):
            scene.remove_object("foot_view", registry_only=True)
        if scene.object_exists("calf_view"):
            scene.remove_object("calf_view", registry_only=True)
        if scene.object_exists("particle_system_view"):
            scene.remove_object("particle_system_view", registry_only=True)
        self._anymals = A1(
            prim_paths_expr="/World/envs/.*/a1", name="a1_view", track_contact_forces=True
        )
        if self._particles_active:
            self.create_particle_systems()
            self.particle_system_view = ParticleSystemView(prim_paths_expr="/World/particleSystem/*")
            scene.add(self.particle_system_view)  
        scene.add(self._anymals)
        scene.add(self._anymals._thigh)
        scene.add(self._anymals._base)
        scene.add(self._anymals._foot)
        scene.add(self._anymals._calf)

    def get_terrain(self, create_mesh=True):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._create_trimesh(create_mesh=create_mesh)
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).float().to(self.device)
        self.terrain_details = torch.tensor(self.terrain.terrain_details, dtype=torch.float).to(self.device)

        levels = self.terrain_details[:, 1].long()  # shape: (N, ) where N=# of terrain blocks
        unique_levels = torch.unique(levels)
        for lvl in unique_levels:
            mask = (levels == lvl)
            row_indices = torch.nonzero(mask, as_tuple=False).flatten()
            self._terrains_by_level[lvl.item()] = row_indices

    def get_anymal(self):
        anymal_translation = torch.tensor([0.0, 0.0, 0.42])
        anymal_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        anymal = A1(
            prim_path=self.default_zero_env_path + "/a1",
            name="a1",
            translation=anymal_translation,
            orientation=anymal_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "a1", get_prim_at_path(anymal.prim_path), self._sim_config.parse_actor_config("a1")
        )
        anymal.set_a1_properties(self._stage, anymal.prim)
        anymal.prepare_contacts(self._stage, anymal.prim)

        self.dof_names = anymal.dof_names
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

    def _randomize_dof_props(self, env_ids):

        if self._task_cfg["env"]["randomizationRanges"]["randomizeMotorStrength"]:
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  self.motor_strength_range[1] - self.motor_strength_range[0]) + self.motor_strength_range[0]
        if self._task_cfg["env"]["randomizationRanges"]["randomizeMotorOffset"]:
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     self.motor_offset_range[1] - self.motor_offset_range[0]) + self.motor_offset_range[0]
        if self._task_cfg["env"]["randomizationRanges"]["randomizeKpFactor"]:
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  self.Kp_factor_range[1] - self.Kp_factor_range[0]) + self.Kp_factor_range[0]
        if self._task_cfg["env"]["randomizationRanges"]["randomizeKdFactor"]:
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  self.Kd_factor_range[1] - self.Kd_factor_range[0]) + self.Kd_factor_range[0]

    
    
    def _randomize_gravity(self):
        if self._task_cfg["env"]["randomizationRanges"]["randomizeGravity"]:
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                    requires_grad=False) * (self.gravity_range[1] - self.gravity_range[0]) + self.gravity_range[0]
            self.gravities[:, :] = external_force.unsqueeze(0)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        self.world._physics_sim_view.set_gravity(
            carb.Float3(gravity[0], gravity[1], gravity[2])
        )

    def _set_mass(self, view, env_ids, distribution="uniform" , operation="additive"):
        """Update material properties for a given asset."""

        masses = self.default_base_masses
        distribution_parameters = self.added_mass_range
        set_masses = view.set_masses
        self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (distribution_parameters[1] - distribution_parameters[0]) + distribution_parameters[0]
        masses += self.payloads
        set_masses(masses)
        print(f"Masses updated: {masses}")
        print(f"default_inertia: {self.default_inertias}")
        # Compute the ratios of the new masses to the default masses.
        ratios = masses / self.default_base_masses
        # The default_inertia is scaled by these ratios.
        # Note: The multiplication below assumes broadcasting works correctly for your inertia tensor shape.
        new_inertias = self.default_inertias * ratios.unsqueeze(-1)
        view.set_inertias(new_inertias)
        print(f"Inertias updated: {new_inertias}")

    def _set_friction(self ,asset, env_ids, device="cpu"):
        """Update material properties for a given asset."""
                # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, dtype=torch.int64, device=device)
        else:
            env_ids = env_ids.cpu()
        materials = asset._physics_view.get_material_properties().to(device)
        
        print(f"Current materials: {materials}")

        # obtain parameters for sampling friction and restitution values
        static_friction_range = self.friction_range
        dynamic_friction_range = self.friction_range
        restitution_range = self.restitution_range
        num_buckets = 64
        # sample material properties from the given ranges
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device=device)
        material_buckets = torch.rand(*(num_buckets, 3), device=device) * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        material_buckets[:, 1] = torch.min(material_buckets[:, 0], material_buckets[:, 1])

        # randomly assign material IDs to the geometries
        shapes_per_env = 4

        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), shapes_per_env), device=device)
        material_samples = material_buckets[bucket_ids]
        print(f"Material samples: {material_samples}")
        print(f"Material samples shape: {material_samples.shape}")
        # material_samples = material_samples.repeat(1, shapes_per_env, 1)
        # print(f"Material samples shape: {material_samples.shape}")
        new_materials = material_samples.view(len(env_ids)*shapes_per_env, 1, 3)
        print(f"New Material samples shape: {new_materials.shape}")
        #update material buffer with new samples
        materials[:] = new_materials

        # apply to simulation
        asset._physics_view.set_material_properties(materials, env_ids)
        print(f"Updated materials: {materials}")
        print(f"Updated materials shape: {materials.shape}")
        self.static_friction_coeffs = material_samples[:, :, 0].clone().to(self.device)  # shape: (num_envs, shapes_per_env)
        self.dynamic_friction_coeffs = material_samples[:, :, 1].clone().to(self.device)  # shape: (num_envs, shapes_per_env)
        self.restitutions = material_samples[:, :, 2].clone().to(self.device)             # shape: (num_envs, shapes_per_env)
        print("Static friction coefficients:", self.static_friction_coeffs)
        print("Dynamic friction coefficients:", self.dynamic_friction_coeffs)
        print("Restitutions:", self.restitutions)

    def _set_coms(self, view, env_ids, distribution="uniform" , operation="additive"):
        """Update material properties for a given view."""

        coms, ori = view.get_coms()
        print(f"Current coms: {coms}")

        distribution_parameters = self.com_displacement_range
        self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                            requires_grad=False) * ( distribution_parameters[1] - distribution_parameters[0]) + distribution_parameters[0]
        print(f"Displacements: {self.com_displacements.unsqueeze(1)}")
        coms += self.com_displacements.unsqueeze(1)
        set_coms = view.set_coms
        print(f"New coms: {coms}")
        set_coms(coms, ori)
        print(f"Coms updated: {coms}")


    def set_compliance(self, env_ids=None, device="cpu"):
        # If no env_ids are provided, do nothing.
        if env_ids is None:
            env_ids = torch.nonzero(self.compliance, as_tuple=False).flatten()

        if len(env_ids) == 0:
            return
        # Ensure env_ids is on the correct device.
        env_ids = env_ids.to(device)

        # Retrieve deformation bounds from the configuration.
        # For example, from: self._task_cfg["env"]["randomizationRanges"]["material_randomization"]["compliance"]["deformation"]
        deformation_bounds = self._task_cfg["env"]["randomizationRanges"]["material_randomization"]["compliance"]["deformation"]
        deformation_low, deformation_high = deformation_bounds[0], deformation_bounds[1]

        # Sample a deflection value for each environment in env_ids.
        num_envs = len(env_ids)
        deformations = torch.rand(num_envs, device=device) * (deformation_high - deformation_low) + deformation_low

        # Compute stiffness and damping per environment.
        stiffness_values = (self.total_masses[env_ids] * 9.81) / deformations
        damping_values = 0.1 * stiffness_values

        # Initialize (or update) the compliance tensors.
        self.stiffness[env_ids, 0] = stiffness_values.to(self.device)
        self.damping[env_ids, 0] = damping_values.to(self.device)
        # print(f"Stiffness set to: {self.stiffness[env_ids]}")
        # print(f"Damping set to: {self.damping[env_ids]}")

        # Collect prim references (once)
        _prim_paths = find_matching_prim_paths(self._anymals._foot_material_path)
        self._prims = [get_prim_at_path(path) for path in _prim_paths]
        self._material_apis = [None] * self._num_envs

        # Ensure we have PhysxMaterialAPI for each environment exactly once.
        for i in range(self._num_envs):
            if self._material_apis[i] is None:
                if self._prims[i].HasAPI(PhysxSchema.PhysxMaterialAPI):
                    self._material_apis[i] = PhysxSchema.PhysxMaterialAPI(self._prims[i])
                else:
                    self._material_apis[i] = PhysxSchema.PhysxMaterialAPI.Apply(self._prims[i])
            # Set the compliance values for the material.
            self._material_apis[i].CreateCompliantContactStiffnessAttr().Set(float(self.stiffness[i, 0]))
            self._material_apis[i].CreateCompliantContactDampingAttr().Set(float(self.damping[i, 0]))

    def store_pbd_params(self):
        """
        Populates self.pbd_parameters for each env that has a non-zero system_idx.
        Each row in self.pbd_parameters corresponds to one environment.
        In this example, we store 8 different PBD material parameters per row:
        [friction, damping, viscosity, density, surface_tension, cohesion, adhesion, cfl_coefficient].
        """
        # Make sure self.pbd_parameters has the right shape:
        # e.g. self.pbd_parameters = torch.zeros((self.num_envs, 8), device=self.device)
        self.pbd_parameters[:] = 0.0  # Reset to 0 for all envs first

        # Grab unique systems in the batch
        unique_systems = torch.unique(self.system_idx)

        for sid in unique_systems:
            # Skip system_idx <= 0, which we treat as "no system" or invalid system
            if sid <= 0:
                continue

            # Look up the corresponding "systemX" config in _particle_cfg
            system_str = f"system{sid.item()}"
            if system_str not in self._particle_cfg:
                # If your config doesn't have this key, skip
                continue

            # Pull out the relevant PBD parameters from the config
            # (Adjust the defaults and parameter names to match your usage)
            mat_cfg = self._particle_cfg[system_str]
            friction         = mat_cfg.get("pbd_material_friction", 0.0)
            particle_friction_scale   = mat_cfg.get("pbd_material_particle_friction_scale", 0.0)
            adhesion          = mat_cfg.get("pbd_material_adhesion", 0.0)
            particle_adhesion_scale  = mat_cfg.get("pbd_material_particle_adhesion_scale", 0.0)
            damping          = mat_cfg.get("pbd_material_damping", 0.0)
            density         = mat_cfg.get("pbd_material_density", 0.0)

            # Find which environments belong to this system
            env_ids = torch.nonzero(self.system_idx == sid, as_tuple=True)[0]

            # Write the parameters into self.pbd_parameters for these envs
            self.pbd_parameters[env_ids, 0] = friction
            self.pbd_parameters[env_ids, 1] = particle_friction_scale
            self.pbd_parameters[env_ids, 2] = adhesion
            self.pbd_parameters[env_ids, 3] = particle_adhesion_scale
            self.pbd_parameters[env_ids, 4] = damping
            self.pbd_parameters[env_ids, 5] = density

    def randomize_pbd_material(self):
        """
        Retrieves an existing PBD material via PhysxSchema and updates its parameters using randomization.
        
        Args:
            system_name: A string (e.g. "system1") identifying the system.
            config: A dict of fixed values and optional range keys. For example:
                {
                    "pbd_material_friction": 0.8,
                    "pbd_material_friction_range": [0.5, 1.0],
                    ...
                }
        """
        # Get the particles material randomization config from randomizationRanges.
        mat_rand_cfg = self._task_cfg["env"]["randomizationRanges"].get("material_randomization", {})
        if not mat_rand_cfg.get("enabled", False):
            # Material randomization is disabled.
            return

        particles_rand_cfg = mat_rand_cfg.get("particles", {}).get("systems", {})

        for system_name, config in particles_rand_cfg.items():
            if not config.get("enabled", False):
                # System Material randomization is disabled.
                return
            material_key = f"pbd_material_{system_name}"
            # Check if the material has been created already.
            if material_key in self.created_materials:
                # Randomize the material parameters for this system.
                pbd_material_path = f"/World/pbdmaterial_{system_name}"
                material_api = PhysxSchema.PhysxPBDMaterialAPI.Get(self._stage, pbd_material_path)
                if not material_api:
                    print(f"[ERROR] Could not find PBD material at {pbd_material_path}")
                    return False

                def sample_param(param_name):
                    # If a range key is provided, sample a value between low and high.
                    range_key = f"{param_name}_range"
                    if range_key in config:
                        low, high = config[range_key]
                        return random.uniform(low, high)
                    # Otherwise, return the fixed value from config, if provided.
                    return config.get(param_name, None)

                # List the parameters to update.
                parameters = [
                    "pbd_material_friction",
                    "pbd_material_particle_friction_scale",
                    "pbd_material_damping",
                    "pbd_material_viscosity",
                    "pbd_material_vorticity_confinement",
                    "pbd_material_surface_tension",
                    "pbd_material_cohesion",
                    "pbd_material_adhesion",
                    "pbd_material_particle_adhesion_scale",
                    "pbd_material_adhesion_offset_scale",
                    "pbd_material_gravity_scale",
                    "pbd_material_lift",
                    "pbd_material_drag",
                    "pbd_material_density",
                    "pbd_material_cfl_coefficient",
                ]

                for param in parameters:
                    value = sample_param(param)
                    if value is not None:
                        if param == "pbd_material_friction":
                            material_api.CreateFrictionAttr().Set(value)
                        elif param == "pbd_material_particle_friction_scale":
                            material_api.CreateParticleFrictionScaleAttr().Set(value)
                        elif param == "pbd_material_damping":
                            material_api.CreateDampingAttr().Set(value)
                        elif param == "pbd_material_viscosity":
                            material_api.CreateViscosityAttr().Set(value)
                        elif param == "pbd_material_vorticity_confinement":
                            material_api.CreateVorticityConfinementAttr().Set(value)
                        elif param == "pbd_material_surface_tension":
                            material_api.CreateSurfaceTensionAttr().Set(value)
                        elif param == "pbd_material_cohesion":
                            material_api.CreateCohesionAttr().Set(value)
                        elif param == "pbd_material_adhesion":
                            material_api.CreateAdhesionAttr().Set(value)
                        elif param == "pbd_material_particle_adhesion_scale":
                            material_api.CreateParticleAdhesionScaleAttr().Set(value)
                        elif param == "pbd_material_adhesion_offset_scale":
                            material_api.CreateAdhesionOffsetScaleAttr().Set(value)
                        elif param == "pbd_material_gravity_scale":
                            material_api.CreateGravityScaleAttr().Set(value)
                        elif param == "pbd_material_lift":
                            material_api.CreateLiftAttr().Set(value)
                        elif param == "pbd_material_drag":
                            material_api.CreateDragAttr().Set(value)
                        elif param == "pbd_material_density":
                            material_api.CreateDensityAttr().Set(value)
                        elif param == "pbd_material_cfl_coefficient":
                            material_api.CreateCflCoefficientAttr().Set(value)
                
                print(f"[INFO] Updated PBD material at {pbd_material_path} with randomized parameters.")
                
            else:
                print(f"[WARN] Material {material_key} not found; skipping randomization.")


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
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                requires_grad=False)
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
        feet_names = ["FL", "FR", "RL", "RR"]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)

        self.last_dof_vel = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)
        
        self.compliance = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.system_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

        self.bx_start = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.bx_end   = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.by_start = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.by_end   = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        env_ids = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)

        self.num_dof = self._anymals.num_dof
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

        self.thigh_pos = torch.zeros((self.num_envs * 4, 3), dtype=torch.float, device=self.device)
        self.thigh_quat = torch.zeros((self.num_envs * 4, 4), dtype=torch.float, device=self.device)
        
        self.thigh_contact_forces = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.calf_contact_forces = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_contact_forces = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)

        # self.torque_limits = self._anymals._physics_view.get_dof_max_forces()[0].tolist()        
        # print(f"Anymal torque limits: {self.torque_limits}")


        self.static_friction_coeffs = torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.dynamic_friction_coeffs = torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs,  dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                            requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.stiffness = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.damping = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.pbd_parameters = torch.zeros((self.num_envs, 6), device=self.device)

        self.friction_scale, self.friction_shift = self.get_scale_shift(self.friction_range)
        self.restitution_scale, self.restitution_shift = self.get_scale_shift(self.restitution_range)
        self.payload_scale, self.payload_shift = self.get_scale_shift(self.added_mass_range)
        self.com_scale, self.com_shift = self.get_scale_shift(self.com_displacement_range)
        self.motor_strength_scale, self.motor_strength_shift = self.get_scale_shift(self.motor_strength_range)
        
        self.default_base_masses = self._anymals._base.get_masses().clone()
        self.default_inertias = self._anymals._base.get_inertias().clone()
        # self.default_materials = self._anymals._foot._physics_view.get_material_properties().to(self.device)
        body_masses = self._anymals.get_body_masses().clone()  # already a torch tensor
        self.total_masses = torch.sum(body_masses, dim=1).to(self.device)

        # Determine the highest terrain level from the terrain details.
        self.highest_level = int(self.terrain_details[:, 1].max().item())
        # Track how many levels we have unlocked so far. Start with 0 (or 1, depending on your preference).
        self.current_unlocked_level = 0

        # Get joint limits
        dof_limits = self._anymals.get_dof_limits()
        lower_limits = dof_limits[0, :, 0]    
        upper_limits = dof_limits[0, :, 1]    
        midpoint = 0.5 * (lower_limits + upper_limits)
        limit_range = upper_limits - lower_limits
        soft_lower_limits = midpoint - 0.5 * limit_range * self.soft_dof_pos_limit
        soft_upper_limits = midpoint + 0.5 * limit_range * self.soft_dof_pos_limit
        self.a1_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.a1_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.a1_dof_soft_lower_limits = soft_lower_limits.to(device=self._device)
        self.a1_dof_soft_upper_limits = soft_upper_limits.to(device=self._device)


        if self._task_cfg["env"]["randomizationRanges"]["randomizeAddedMass"]:
            self._set_mass(self._anymals._base, env_ids=env_ids)
        if self._task_cfg["env"]["randomizationRanges"]["randomizeCOM"]:
            self._set_coms(self._anymals._base, env_ids=env_ids)
        if self._task_cfg["env"]["randomizationRanges"]["randomizeFriction"]:
            self._set_friction(self._anymals._foot, env_ids=env_ids)

        self._prepare_reward_function()

        self.reset_idx(env_ids)
        
        self.init_done = True

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        if self.vel_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        self.update_terrain_level(env_ids)
        self._randomize_dof_props(env_ids)

        self.base_pos[env_ids] = self.base_init_state[0:3]
        self.base_pos[env_ids, 0:3] += self.env_origins[env_ids]
        self.base_pos[env_ids, 0:2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
        rand_yaw = torch_rand_float(0, 2 * np.pi, (len(env_ids), 1), device=self.device)
        random_quat = torch.cat([
            torch.cos(rand_yaw / 2),
            torch.zeros(len(env_ids), 2, device=self.device),
            torch.sin(rand_yaw / 2)
        ], dim=1)
        self.base_quat[env_ids] = random_quat        
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
        self.obs_history_buf[env_ids, :, :] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        self.extras["time_outs"] = self.timeout_buf
        self.extras["episode"]["max_command_x"] = self.command_x_range[1]


    def update_terrain_level(self, env_ids):
    """
    Example version that:
      - If we've just unlocked a new level (i.e., self.current_unlocked_level < self.highest_level),
        then half of the env_ids go to the new level, half remain among lower levels.
      - Once we have reached all levels unlocked, distribution among all 0..highest_level is random.
    """

        if not self.init_done:
            # Skip terrain update on the very first reset if you like, or keep minimal logic
            return

        # Shuffle env_ids so the first half for "new level" is random
        rand_perm = torch.randperm(len(env_ids), device=self.device)
        mid_index = len(env_ids) // 2
        new_level_ids = env_ids[rand_perm[:mid_index]]
        old_level_ids = env_ids[rand_perm[mid_index:]]

        # If we haven't yet unlocked all levels, we "push" to the next level
        if self.current_unlocked_level < self.highest_level:
            # Unlock the next level (increment by 1).
            self.current_unlocked_level += 1

            # 50% of the envs: assign them to the newly unlocked level
            self.terrain_levels[new_level_ids] = self.current_unlocked_level

            # 50% of the envs: random among the previously unlocked levels
            # (that is, between 0 .. (current_unlocked_level - 1))
            if self.current_unlocked_level > 0:
                self.terrain_levels[old_level_ids] = torch.randint(
                    low=0,
                    high=self.current_unlocked_level,  # does NOT include self.current_unlocked_level
                    size=(len(old_level_ids),),
                    device=self.device,
                )
            else:
                # if current_unlocked_level == 0, all old_level_ids are forced to 0
                self.terrain_levels[old_level_ids] = 0

        else:
            # Once all levels are unlocked, we just do random among ALL levels [0..highest_level]
            self.terrain_levels[env_ids] = torch.randint(
                low=0,
                high=self.highest_level + 1,  # +1 because torch.randint's high is exclusive
                size=(len(env_ids),),
                device=self.device,
            )

        levels = self.terrain_levels[env_ids]
        # Group env_ids by terrain level:
        unique_levels, inverse_idx = torch.unique(levels, return_inverse=True)

        for i, lvl in enumerate(unique_levels):
            # Get which envs in env_ids map to this terrain level
            group = env_ids[inverse_idx == i]
            # Rows for this level
            candidate_indices = self._terrains_by_level[lvl.item()]

            # Randomly sample from that subset
            row_count = candidate_indices.shape[0]
            rand_rows = torch.randint(
                low=0,
                high=row_count,
                size=(group.shape[0],),
                device=self.device,
            )
            chosen_rows = candidate_indices[rand_rows]

            # bounding boxes, environment origins, PBD or compliance, etc.
            self.bx_start[group] = self.terrain_details[chosen_rows, 10]
            self.bx_end[group]   = self.terrain_details[chosen_rows, 11]
            self.by_start[group] = self.terrain_details[chosen_rows, 12]
            self.by_end[group]   = self.terrain_details[chosen_rows, 13]

            self.compliance[group]  = self.terrain_details[chosen_rows, 6].bool()
            self.system_idx[group]  = self.terrain_details[chosen_rows, 7].long()

            # (row, col) -> environment origins
            rows = self.terrain_details[chosen_rows, 2].long()
            cols = self.terrain_details[chosen_rows, 3].long()
            self.env_origins[group] = self.terrain_origins[rows, cols]

        # Update compliance and stored PBD parameters for these newly changed envs
        self.set_compliance(env_ids)
        self.store_pbd_params()

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
        self.calf_contact_forces = self._anymals._calf.get_net_contact_forces(dt=self.dt,clone=False).view(self._num_envs, 4, 3)
        self.base_contact_forces = self._anymals._base.get_net_contact_forces(dt=self.dt,clone=False).view(self._num_envs, 3)

    def pre_physics_step(self, actions):
        if not self.world.is_playing():
            return

        self.actions = actions.clone().to(self.device)

        for i in range(self.decimation):
            if self.world.is_playing():
                self.joint_pos_target = self.action_scale * self.actions + self.default_dof_pos
                torques = self.Kp * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.Kd * self.Kd_factors * self.dof_vel
                torques = torques * self.motor_strengths
                torques = torch.clip(torques, -33.5, 33.5)
                self._anymals.set_joint_efforts(torques)
                self.torques = torques
                SimulationContext.step(self.world, render=False)
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
            if self.common_step_counter % self.gravity_randomize_interval == 0:
                self._randomize_gravity()
            

            # prepare quantities
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

            self.check_termination()
            self.get_states()
            self.compute_reward()

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()

            self.last_actions[:] = self.actions[:]
            self.last_dof_vel[:] = self.dof_vel[:]
            
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

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
        self.has_fallen = (torch.norm(self.base_contact_forces, dim=1) > 1.0) 
        self.reset_buf = self.has_fallen.clone()
        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)
        
        # Convert each robot's base (x,y) position into heightfield indices
        hf_x = (self.base_pos[:, 0] + self.terrain.border_size) / self.terrain.horizontal_scale
        hf_y = (self.base_pos[:, 1] + self.terrain.border_size) / self.terrain.horizontal_scale
        # Define a small distance buffer (in index units) for early reset
        buffer = 0.5  # adjust this value based on your setup
        # Check if the robot is outside the "safe" bounds with the buffer:
        out_of_bounds = (
            (hf_x < self.bx_start + buffer) |
            (hf_x > self.bx_end - buffer)  |
            (hf_y < self.by_start + buffer) |
            (hf_y > self.by_end - buffer)
        )
        self.reset_buf = torch.where(out_of_bounds, torch.ones_like(self.reset_buf), self.reset_buf)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def get_scale_shift(self, rng):
        rng_tensor = torch.tensor(rng, dtype=torch.float, device=self.device)
        scale = 2.0 / (rng_tensor[1] - rng_tensor[0])
        shift = (rng_tensor[1] + rng_tensor[0]) / 2.0
        return scale, shift

    def get_observations(self):
        """
        Build a 'proprio_obs' block and (optionally) a 'heights' block,
        then combine them in obs_buf. Only 'proprio_obs' is put into obs_history.
        """
        # 1) Collect all your normal (proprio) observations:
        proprio_obs = torch.cat((
            self.base_lin_vel * self.lin_vel_scale,
            self.base_ang_vel * self.ang_vel_scale,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            self.dof_pos * self.dof_pos_scale,
            self.dof_vel * self.dof_vel_scale,
            self.actions,
        ), dim=-1)  # this should match self._num_proprio in size

        # 2) Add noise (only on proprio)
        if self.add_noise:
            proprio_obs += (2.0 * torch.rand_like(proprio_obs) - 1.0) * self.noise_scale_vec
        proprio_obs = torch.clip(proprio_obs, -self.clip_obs, self.clip_obs)

        # 3) If measuring heights, compute them and concatenate AFTER the proprio block.
        if self.measure_heights:
            self.measured_heights = self.get_heights()
            if self.debug_heights:
                # self._visualize_terrain_heights() 
                # self._visualize_depression_indices()
                self.query_top_particle_positions(0)
                self._visualize_height_scans()

            heights = torch.clip(
                self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1.0, 1.0
            ) * self._task_cfg["env"]["learn"]["heightMeasurementNoise"] * self._task_cfg["env"]["learn"]["noiseLevel"] * self.height_meas_scale
            # No noise is added to heights:
            
            final_obs_no_history = torch.cat([proprio_obs, heights], dim=-1)
        else:
            final_obs_no_history = proprio_obs
        
        # 4) Add any privileged data after the base blocks:
        priv_buf = torch.cat((
            (self.static_friction_coeffs - self.friction_shift) * self.friction_scale,
            (self.dynamic_friction_coeffs - self.friction_shift) * self.friction_scale,
            (self.restitutions - self.restitution_shift) * self.restitution_scale,
            (self.payloads.unsqueeze(1) - self.payload_shift) * self.payload_scale,
            (self.com_displacements - self.com_shift) * self.com_scale,
            (self.motor_strengths - self.motor_strength_shift) * self.motor_strength_scale,
        ), dim=1)

        # 5) Concatenate everything: [ (proprio + maybe heights) + priv_buf + obs_history ]
        self.obs_buf = torch.cat([
            final_obs_no_history,
            priv_buf,
            self.obs_history_buf.view(self.num_envs, -1)
        ], dim=-1)

        # 6) Update the rolling history ONLY with the current proprio block:
        # shift old frames and store the new one
        self.obs_history_buf[:, :-1] = self.obs_history_buf[:, 1:].clone()
        self.obs_history_buf[:, -1] = proprio_obs

    def get_ground_heights_below_knees(self):
        points = self.knee_pos.reshape(self.num_envs, 4, 3)
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
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        thigh_contact = (
            torch.norm(self.thigh_contact_forces, dim=-1)
            > 0.1
        )
        calf_contact = (torch.norm(self.calf_contact_forces, dim=-1) > 0.1)
        total_contact = thigh_contact + calf_contact
        return torch.sum(total_contact, dim=-1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.has_fallen
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.a1_dof_soft_lower_limits).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.a1_dof_soft_upper_limits).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.only_positive_rewards).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.foot_contact_forces[:, self.feet_indices, 2] > 1.0  # Placeholder for contact detection, adjust threshold as needed
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt  # Assuming self.dt is the timestep duration
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~self.contact_filt
        return rew_airTime

    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.foot_contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.foot_contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.foot_contact_forces[:, self.feet_indices, :], dim=-1) -  self.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_hip_motion(self):
        # Penalize hip motion
        return torch.sum(torch.abs(self.dof_pos[:, :4] - self.default_dof_pos[:, :4]), dim=1)

    #------------ end reward functions----------------
    


    #------------ particle based functions----------------
    def create_particle_systems(self):
        for i in range(self.terrain_details.shape[0]):
            terrain_row = self.terrain_details[i]
            if not int(terrain_row[5].item()):
                continue  # Skip terrains without particles

            # Construct the system name from integer system_id
            system_id = int(terrain_row[7])                # e.g. 1, 2, ...
            system_name = f"system{system_id}"     # "system1", "system2", etc.            
            material_key = f"pbd_material_{system_name}"
            particle_system_path = f"/World/particleSystem/{system_name}"

            # **Create Particle System if not already created**
            if system_name not in self.created_particle_systems:
                if not self._stage.GetPrimAtPath(particle_system_path).IsValid():
                    particle_system = ParticleSystem(
                        prim_path=particle_system_path,
                        particle_system_enabled=True,
                        simulation_owner="/physicsScene",
                        rest_offset=self._particle_cfg[system_name].get("particle_system_rest_offset", None),
                        contact_offset=self._particle_cfg[system_name].get("particle_system_contact_offset", None),
                        solid_rest_offset=self._particle_cfg[system_name].get("particle_system_solid_rest_offset", None),
                        particle_contact_offset=self._particle_cfg[system_name].get("particle_system_particle_contact_offset", None),
                        max_velocity=self._particle_cfg[system_name].get("particle_system_max_velocity", None),
                        max_neighborhood=self._particle_cfg[system_name].get("particle_system_max_neighborhood", None),
                        solver_position_iteration_count=self._particle_cfg[system_name].get("particle_system_solver_position_iteration_count", None),
                        enable_ccd=self._particle_cfg[system_name].get("particle_system_enable_ccd", None),
                        max_depenetration_velocity=self._particle_cfg[system_name].get("particle_system_max_depenetration_velocity", None),
                    )
                    if self._particle_cfg[system_name].get("Anisotropy", False):
                        # apply api and use all defaults
                        PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.prim)

                    if self._particle_cfg[system_name].get("Smoothing", False):
                        # apply api and use all defaults
                        PhysxSchema.PhysxParticleSmoothingAPI.Apply(particle_system.prim)

                    if self._particle_cfg[system_name].get("Isosurface", False):
                        # apply api and use all defaults
                        PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particle_system.prim)
                        # tweak anisotropy min, max, and scale to work better with isosurface:
                        if self._particle_cfg[system_name].get("Anisotropy", False):
                            ani_api = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.prim)
                            ani_api.CreateScaleAttr().Set(5.0)
                            ani_api.CreateMinAttr().Set(1.0)  # avoids gaps in surface
                            ani_api.CreateMaxAttr().Set(2.0)  # avoids gaps in surface

                    print(f"[INFO] Created Particle System: {particle_system_path}")
                self.created_particle_systems[system_name] = particle_system_path

            # **Create PBD Material if not already created**
            if material_key not in self.created_materials:
                self.create_pbd_material(system_name)
                self.created_materials[material_key] = True

            # **Create Particle Grid under the existing system**
            self.create_particle_grid(i, terrain_row, system_name)
        print(f"[INFO] Created {len(self.created_materials)} PBD Materials.")
        print(f"[INFO] Created {self.total_particles} Particles.")


    def create_pbd_material(self, system_name):
        # Retrieve material parameters from config based on system_name
        material_cfg = self._particle_cfg[system_name]
        
        # Define unique material path
        pbd_material_path = f"/World/pbdmaterial_{system_name}"
        
        # Check if the material already exists
        if not self._stage.GetPrimAtPath(pbd_material_path).IsValid():
            # Create PBD Material
            particleUtils.add_pbd_particle_material(
                self._stage,
                Sdf.Path(pbd_material_path),
                friction=material_cfg.get("pbd_material_friction", None),
                particle_friction_scale=material_cfg.get("pbd_material_particle_friction_scale", None),
                damping=material_cfg.get("pbd_material_damping", None),
                viscosity=material_cfg.get("pbd_material_viscosity", None),
                vorticity_confinement=material_cfg.get("pbd_material_vorticity_confinement", None),
                surface_tension=material_cfg.get("pbd_material_surface_tension", None),
                cohesion=material_cfg.get("pbd_material_cohesion", None),
                adhesion=material_cfg.get("pbd_material_adhesion", None),
                particle_adhesion_scale=material_cfg.get("pbd_material_particle_adhesion_scale", None),
                adhesion_offset_scale=material_cfg.get("pbd_material_adhesion_offset_scale", None),
                gravity_scale=material_cfg.get("pbd_material_gravity_scale", None),
                lift=material_cfg.get("pbd_material_lift", None),
                drag=material_cfg.get("pbd_material_drag", None),
                density=material_cfg.get("pbd_material_density", None),
                cfl_coefficient=material_cfg.get("pbd_material_cfl_coefficient", None)
            )
            print(f"[INFO] Created PBD Material: {pbd_material_path}")

            # Assign material to particle system
            ps = PhysxSchema.PhysxParticleSystem.Get(self._stage, Sdf.Path(f"/World/particleSystem/{system_name}"))
            physicsUtils.add_physics_material_to_prim(self._stage, ps.GetPrim(), pbd_material_path)

            if self._particle_cfg[system_name].get("Looks", False):
                mtl_created = []
                omni.kit.commands.execute(
                    "CreateAndBindMdlMaterialFromLibrary",
                    mdl_name="OmniSurfacePresets.mdl",
                    mtl_name="OmniSurface_DeepWater",
                    mtl_created_list=mtl_created,
                    select_new_prim=False,
                )
                material_path = mtl_created[0]
                omni.kit.commands.execute(
                    "BindMaterial", prim_path=Sdf.Path(f"/World/particleSystem/{system_name}"), material_path=material_path
                )


    def create_particle_grid(self, i, terrain_row, system_name):
        # Define the particle system path
        particle_system_path = f"/World/particleSystem/{system_name}"    

        # Extract parameters from terrain_detail and config
        level = int(terrain_row[1])
        row_idx = int(terrain_row[2])
        col_idx = int(terrain_row[3])
        depth = float(terrain_row[8]) * 2.0
        size  = float(terrain_row[9])
    
        # If your environment origins are stored separately:
        env_origin = self.terrain_origins[row_idx, col_idx].float()
        env_origin_x = float(env_origin[0])
        env_origin_y = float(env_origin[1])
        env_origin_z = float(env_origin[2])
        
        x_position = env_origin_x - size / 2.0
        y_position = env_origin_y - size / 2.0
        z_position = env_origin_z + 0.05  # Align with environment origin
        lower = Gf.Vec3f(x_position, y_position, z_position)

        system_cfg = self._particle_cfg[system_name]
        solid_rest_offset = system_cfg.get("particle_system_solid_rest_offset", None)
        particle_spacing = system_cfg.get("particle_grid_spacing", None)
        fluid = system_cfg.get("particle_grid_fluid", None)

        if fluid:
            fluid_rest_offset = 0.99 * 0.6 * system_cfg.get("particle_system_particle_contact_offset", None)
            particle_spacing = 2.5 * fluid_rest_offset
        else:
            particle_spacing = 2.5 * solid_rest_offset

        num_samples_x = int(size / particle_spacing) + 1
        num_samples_y = int(size / particle_spacing) + 1
        num_samples_z = int(depth / particle_spacing) + 1

        jitter_factor = system_cfg["particle_grid_jitter_factor"] * particle_spacing

        positions = []
        velocities = []
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
                    positions.append(Gf.Vec3f(jittered_x, jittered_y, jittered_z))
                    velocities.append(uniform_particle_velocity)
                    ind += 1
                    z += particle_spacing
                z = lower[2]
                y += particle_spacing
            y = lower[1]
            x += particle_spacing

        # Define particle point instancer path (now grouped by level)
        particle_point_instancer_path = f"/World/particleSystem/{system_name}/level_{level}/particleInstancer"

        # Store instancer path in a dictionary grouped by level
        if level not in self.particle_instancers_by_level:
            self.particle_instancers_by_level[level] = []

        # Check if the PointInstancer already exists to prevent duplication
        if not self._stage.GetPrimAtPath(particle_point_instancer_path).IsValid():
            # Add the particle set to the point instancer
            particleUtils.add_physx_particleset_pointinstancer(
                self._stage,
                Sdf.Path(particle_point_instancer_path),
                Vt.Vec3fArray(positions),
                Vt.Vec3fArray(velocities),
                Sdf.Path(particle_system_path),
                self._particle_cfg[system_name]["particle_grid_self_collision"],
                self._particle_cfg[system_name]["particle_grid_fluid"],
                self._particle_cfg[system_name]["particle_grid_particle_group"],
                self._particle_cfg[system_name]["particle_grid_particle_mass"],
                self._particle_cfg[system_name]["particle_grid_density"],
                num_prototypes=1,  # Adjust if needed
                prototype_indices=None  # Adjust if needed
            )
            print(f"[INFO] Created Particle Grid at {particle_point_instancer_path}")
            self.particle_instancers_by_level[level] = particle_point_instancer_path
            # Increment the total_particles counter
            self.total_particles += len(positions)
        
            # Configure particle prototype
            particle_prototype_sphere = UsdGeom.Sphere.Get(
                self._stage, Sdf.Path(particle_point_instancer_path).AppendChild("particlePrototype0")
            )
            if fluid:
                radius = fluid_rest_offset 
            else:
                radius = solid_rest_offset
            particle_prototype_sphere.CreateRadiusAttr().Set(radius)
            # Increase counters, etc.
            self.total_particles += len(positions)
            print(f"[INFO] Created {len(positions)} Particles at {particle_point_instancer_path}")
        else:
            point_instancer = UsdGeom.PointInstancer.Get(self._stage, particle_point_instancer_path)            
            
            existing_positions = point_instancer.GetPositionsAttr().Get()
            existing_velocities = point_instancer.GetVelocitiesAttr().Get()

            # Convert Python lists -> Vt.Vec3fArray (new data)
            new_positions = Vt.Vec3fArray(positions)
            new_velocities = Vt.Vec3fArray(velocities)

            appended_positions = Vt.Vec3fArray(list(existing_positions) + list(new_positions))
            appended_velocities = Vt.Vec3fArray(list(existing_velocities) + list(new_velocities))

            # Re-set the attributes on the same instancer
            point_instancer.GetPositionsAttr().Set(appended_positions)
            point_instancer.GetVelocitiesAttr().Set(appended_velocities)

            # Also update the prototype indices if necessary.
            existing_proto = list(point_instancer.GetProtoIndicesAttr().Get() or [])
            new_proto = [0] * len(new_positions)
            point_instancer.GetProtoIndicesAttr().Set(existing_proto + new_proto)

            # IMPORTANT: Reconfigure the particle set so that the simulation recalculates
            # properties such as mass based on the updated number of particles.
            particleUtils.configure_particle_set(
                point_instancer.GetPrim(),
                particle_system_path,
                self._particle_cfg[system_name]["particle_grid_self_collision"],
                self._particle_cfg[system_name]["particle_grid_fluid"],
                self._particle_cfg[system_name]["particle_grid_particle_group"],
                self._particle_cfg[system_name]["particle_grid_particle_mass"] * len(appended_positions),  # update mass based on total count
                self._particle_cfg[system_name]["particle_grid_density"],
            )
            print(f"[INFO] Appended {len(new_positions)} Particles to {particle_point_instancer_path}")
            # Increment the total_particles counter
            self.total_particles += len(new_positions)


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


    def _visualize_terrain_heights(self):
        """
        Spawns (or updates) a PointInstancer of small spheres for every cell in self.height_samples,
        but only for the main terrain region (excluding the border).
        """
        import omni.kit.commands
        from pxr import UsdGeom, Sdf, Gf, Vt

        stage = self._stage  # or get_current_stage()

        # 1) Create a dedicated Scope for the debug instancer
        parent_scope_path = "/World/DebugTerrainHeights"
        parent_scope_prim = stage.GetPrimAtPath(parent_scope_path)
        if not parent_scope_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Scope",
                prim_path=parent_scope_path,
                attributes={}
            )

        # 2) Construct a PointInstancer prim if not already there
        point_instancer_path = f"{parent_scope_path}/terrain_points_instancer"
        point_instancer_prim = stage.GetPrimAtPath(point_instancer_path)
        if not point_instancer_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="PointInstancer",
                prim_path=point_instancer_path,
                attributes={}
            )

        point_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(point_instancer_path))

        # 3) Create/ensure we have a single prototype (Sphere) under the PointInstancer
        prototype_index = 0
        proto_path = f"{point_instancer_path}/prototype_Sphere"
        prototype_prim = stage.GetPrimAtPath(proto_path)
        if not prototype_prim.IsValid():
            # Create a sphere prototype
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Sphere",
                prim_path=proto_path,
                attributes={"radius": 0.02},  # adjust sphere size as you wish
            )
        # This step ensures the point-instancer references the prototype as well
        if len(point_instancer.GetPrototypesRel().GetTargets()) == 0:
            point_instancer.GetPrototypesRel().AddTarget(proto_path)

        # 4) Build up the positions (and protoIndices) for each cell of the *main* height field
        tot_rows = self.terrain.tot_rows   # i dimension
        tot_cols = self.terrain.tot_cols   # j dimension
        border   = self.terrain.border     # integer # of “cells” that define the border thickness

        positions = []
        proto_indices = []

        # Only iterate within the interior region [border, (tot_rows - border)) and [border, (tot_cols - border))
        for i in range(border, tot_rows - border):
            for j in range(border, tot_cols - border):
                # Convert row/col -> world coordinates
                px = i * self.terrain.horizontal_scale - self.terrain.border_size
                py = j * self.terrain.horizontal_scale - self.terrain.border_size
                pz = float(self.height_samples[i, j] * self.terrain.vertical_scale)

                positions.append(Gf.Vec3f(px, py, pz))
                proto_indices.append(prototype_index)

        positions_array = Vt.Vec3fArray(positions)
        proto_indices_array = Vt.IntArray(proto_indices)

        # 5) Assign the arrays to the PointInstancer
        point_instancer.CreatePositionsAttr().Set(positions_array)
        point_instancer.CreateProtoIndicesAttr().Set(proto_indices_array)

        # Optionally give these debug spheres a color by modifying the prototype itself:
        sphere_geom = UsdGeom.Sphere(stage.GetPrimAtPath(proto_path))
        sphere_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 1.0)])


    def _visualize_depression_indices(self):
        """
        Creates (or updates) a PointInstancer of small spheres at z=0 
        for each (x,y) entry
        """

        stage = self._stage  # Or get_current_stage()

        # 1) Create a dedicated Scope for debugging these indices
        debug_scope_path = "/World/DebugDepressionIndices"
        if not stage.GetPrimAtPath(debug_scope_path).IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Scope",
                prim_path=debug_scope_path,
                attributes={}
            )

        # 2) Create a PointInstancer for all depression indices
        instancer_path = f"{debug_scope_path}/DepressionIndicesPointInstancer"
        if not stage.GetPrimAtPath(instancer_path).IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="PointInstancer",
                prim_path=instancer_path,
                attributes={}
            )
        point_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(instancer_path))

        # 3) Make sure there's a prototype sphere
        sphere_proto_path = f"{instancer_path}/DepressionIndexSphere"
        if not stage.GetPrimAtPath(sphere_proto_path).IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Sphere",
                prim_path=sphere_proto_path,
                attributes={"radius": 0.02},  # adjust size as desired
            )
        # Ensure the instancer references the prototype
        if len(point_instancer.GetPrototypesRel().GetTargets()) == 0:
            point_instancer.GetPrototypesRel().AddTarget(sphere_proto_path)

        # 4) Collect positions and prototype indices
        positions = []
        proto_indices = []

        prototype_index = 0  # single prototype

        for terrain_entry in self.terrain.terrain_details:
            terrain_name = terrain_entry[4]
            if terrain_name == "central_depression_terrain":
                bx_start = int(terrain_entry[10])
                bx_end   = int(terrain_entry[11])
                by_start = int(terrain_entry[12])
                by_end   = int(terrain_entry[13])

                # For each (i, j) in that rectangle
                for i in range(bx_start, bx_end):
                    for j in range(by_start, by_end):
                        # Convert heightfield indices to world coordinates
                        px = i * self.terrain.horizontal_scale - self.terrain.border_size
                        py = j * self.terrain.horizontal_scale - self.terrain.border_size
                        pz = 0.0  # place at ground (z=0)
                        positions.append(Gf.Vec3f(px, py, pz))
                        proto_indices.append(prototype_index)

        positions_array = Vt.Vec3fArray(positions)
        proto_indices_array = Vt.IntArray(proto_indices)

        # 5) Assign to the PointInstancer
        point_instancer.CreatePositionsAttr().Set(positions_array)
        point_instancer.CreateProtoIndicesAttr().Set(proto_indices_array)

        # (Optional) Color the debug spheres differently
        sphere_geom = UsdGeom.Sphere(stage.GetPrimAtPath(sphere_proto_path))
        sphere_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])  # green for clarity


    def query_top_particle_positions(self, level=None):
        """
        Query all particle positions from the given level and, using the depression indices
        for that level, find for each grid cell the
        top (maximum z) particle position.

        Returns:
            A dictionary mapping cell indices (i, j) to a tuple:
            (cell_center_x, cell_center_y, top_z)
            Only cells where at least one particle was found are included.
        """
        if not self.particle_instancers_by_level:
            print("No particle instancers registered yet; skipping top particle query.")
            return
        stage = self._stage
        # Determine which levels to process: either the given level or all levels present
        levels_to_process = [level] if level is not None else list(self.particle_instancers_by_level.keys())

        for lvl in levels_to_process:
            if lvl not in self.particle_instancers_by_level:
                print(f"No particle instancers registered for level {lvl}")
                continue

            prim = self._stage.GetPrimAtPath(self.particle_instancers_by_level[lvl])
            point_instancer = UsdGeom.PointInstancer(prim)
            particle_positions = point_instancer.GetPositionsAttr().Get()  # Vt.Vec3fArray of positions
            particle_positions_np = np.array(particle_positions)

            # 1) Create a dedicated Scope for debugging these indices
            debug_scope_path = "/World/DebugDepressionIndices"
            if not stage.GetPrimAtPath(debug_scope_path).IsValid():
                omni.kit.commands.execute(
                    "CreatePrim",
                    prim_type="Scope",
                    prim_path=debug_scope_path,
                    attributes={}
                )

            # 2) Create a PointInstancer for all depression indices
            instancer_path = f"{debug_scope_path}/DepressionIndicesPointInstancer"
            if not stage.GetPrimAtPath(instancer_path).IsValid():
                omni.kit.commands.execute(
                    "CreatePrim",
                    prim_type="PointInstancer",
                    prim_path=instancer_path,
                    attributes={}
                )
            point_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(instancer_path))

            # 3) Make sure there's a prototype sphere
            sphere_proto_path = f"{instancer_path}/DepressionIndexSphere"
            if not stage.GetPrimAtPath(sphere_proto_path).IsValid():
                omni.kit.commands.execute(
                    "CreatePrim",
                    prim_type="Sphere",
                    prim_path=sphere_proto_path,
                    attributes={"radius": 0.02},  # adjust size as desired
                )
            # Ensure the instancer references the prototype
            if len(point_instancer.GetPrototypesRel().GetTargets()) == 0:
                point_instancer.GetPrototypesRel().AddTarget(sphere_proto_path)
            # Each region is assumed to be a dict with keys: "start_x", "end_x", "start_y", "end_y".
            prototype_index = 0  # single prototype
            positions = []
            proto_indices = []

            env_ids = (self.terrain_levels == lvl).nonzero(as_tuple=False).flatten()
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_particle_height_points), self.particle_height_points[env_ids]
            ) + (self.base_pos[env_ids, 0:3]).unsqueeze(1)
            points += self.terrain.border_size
            points = (points / self.terrain.horizontal_scale).long()
            px = points[:, :, 0].view(-1)
            py = points[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
            
            # Convert the tensors of indices to Python lists.
            px_list = px.tolist()
            py_list = py.tolist()

            for i, j in zip(px_list, py_list):
                # Convert (i,j) indices into world coordinates.
                # Here we assume that each cell spans a distance equal to terrain.horizontal_scale,
                # and that the terrain's origin offset is given by terrain.border_size.
                cell_x_min = (i-0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                cell_x_max = (i + 0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                cell_y_min = (j-0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                cell_y_max = (j + 0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                px = i * self.terrain.horizontal_scale - self.terrain.border_size
                py = j * self.terrain.horizontal_scale - self.terrain.border_size

                # For each cell defined by cell_x_min, cell_x_max, cell_y_min, cell_y_max:
                mask = (
                    (particle_positions_np[:, 0] >= cell_x_min) &
                    (particle_positions_np[:, 0] < cell_x_max) &
                    (particle_positions_np[:, 1] >= cell_y_min) &
                    (particle_positions_np[:, 1] < cell_y_max)
                )
                if np.any(mask):
                    top_z = float(np.max(particle_positions_np[mask, 2]))
                    top_z = min(top_z, 1*self.terrain.vertical_scale)
                    self.height_samples[i, j] = top_z 
                else:
                    top_z = self.height_samples[i, j] * self.terrain.vertical_scale
                positions.append(Gf.Vec3f(px, py, float(top_z)))
                proto_indices.append(prototype_index)

            positions_array = Vt.Vec3fArray(positions)
            proto_indices_array = Vt.IntArray(proto_indices)

            # 5) Assign to the PointInstancer
            point_instancer.CreatePositionsAttr().Set(positions_array)
            point_instancer.CreateProtoIndicesAttr().Set(proto_indices_array)

            # (Optional) Color the debug spheres differently
            sphere_geom = UsdGeom.Sphere(stage.GetPrimAtPath(sphere_proto_path))
            sphere_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])  # green for clarity


    def _visualize_height_scans(self):
        """
        Visualizes the height-scan points more efficiently by using a single PointInstancer
        to display all the debug spheres instead of creating/updating individual prims.
        """
        if not self.world.is_playing():
            return

        import omni.kit.commands
        from pxr import Sdf, Gf, UsdGeom, Vt

        # 1) Create/Get a dedicated DebugHeight scope
        parent_scope_path = "/World/DebugHeight"
        parent_scope_prim = self._stage.GetPrimAtPath(parent_scope_path)
        if not parent_scope_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Scope",
                prim_path=parent_scope_path,
                attributes={}
            )

        # 2) Create/Get a single PointInstancer for all height scan debug spheres
        point_instancer_path = f"{parent_scope_path}/HeightScanPointInstancer"
        point_instancer_prim = self._stage.GetPrimAtPath(point_instancer_path)
        if not point_instancer_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="PointInstancer",
                prim_path=point_instancer_path,
                attributes={}
            )
        point_instancer = UsdGeom.PointInstancer(self._stage.GetPrimAtPath(point_instancer_path))

        # 3) Create/ensure a single prototype sphere (with a small radius)
        prototype_path = f"{point_instancer_path}/prototype_Sphere"
        prototype_prim = self._stage.GetPrimAtPath(prototype_path)
        if not prototype_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Sphere",
                prim_path=prototype_path,
                attributes={"radius": 0.02},
            )
        # Make sure the PointInstancer references the prototype
        if len(point_instancer.GetPrototypesRel().GetTargets()) == 0:
            point_instancer.GetPrototypesRel().AddTarget(prototype_path)

        # 4) Accumulate the sphere positions (and assign a prototype index of 0 for all)
        positions = []
        proto_indices = []

        points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
            self.base_pos[:, 0:3]
        ).unsqueeze(1)
        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        num_points = self.num_envs * self.num_height_points
        for idx in range(num_points):
            # Compute world x and y from grid indices
            world_x = px[idx].item() * self.terrain.horizontal_scale - self.terrain.border_size
            world_y = py[idx].item() * self.terrain.horizontal_scale - self.terrain.border_size
            # Look up the measured height at the grid cell and convert to world units
            measured_z = self.height_samples[px[idx].item(), py[idx].item()].item() * self.terrain.vertical_scale
            positions.append(Gf.Vec3f(world_x, world_y, measured_z))
            proto_indices.append(0)


        positions_array = Vt.Vec3fArray(positions)
        proto_indices_array = Vt.IntArray(proto_indices)

        # 5) Update the PointInstancer with the positions and prototype indices
        point_instancer.CreatePositionsAttr().Set(positions_array)
        point_instancer.CreateProtoIndicesAttr().Set(proto_indices_array)

        # 6) (Optional) Set a debug color on the prototype sphere
        sphere_geom = UsdGeom.Sphere(self._stage.GetPrimAtPath(prototype_path))
        sphere_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

#------------ helper functions----------------

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
