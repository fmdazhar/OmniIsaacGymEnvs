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
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *


# terrain generator

class Terrain:
    def __init__(self, cfg, num_robots) -> None:
        self.cfg = cfg
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 20

        # Map dimensions in meters
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        terrain_types = self.cfg["terrain_types"]

        if not cfg["curriculum"]:
            self.env_rows = cfg["numLevels"]
            self.env_cols = cfg["numTerrains"]
        else:
            self.env_rows = len(terrain_types)                # number of rows = number of unique terrain "types"
            self.env_cols = max(t["count"] for t in terrain_types)  # total columns = max of 'count' across all types
        
        self.num_maps = self.env_rows * self.env_cols
        # Each sub-rectangle (sub-terrain) dimensions in "heightfield" pixels
        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        # Master heightfield storage
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.terrain_mapping = {
                                "flat": 0,
                                "rough": 1,
                                "compliant" : 2,
                                "central_depression_terrain": 3,
                                # add other terrain names if needed
                            }
        # We'll keep track of each sub-terrain's info in a list:
        self.terrain_details = []

        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        # Actually build the terrain
        if cfg["curriculum"]:
            self.deformable_curriculum()
        else:
            self.full_flat_terrain()
        
        self.heightsamples = self.height_field_raw

        # Convert to tri-mesh
        self.vertices, self.triangles = convert_heightfield_to_trimesh(
            self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"]
        )

    def deformable_curriculum(self):
        """
        Create sub-terrains in a deterministic 'in-order' fashion based on
        the `terrain_types` array from the config, repeating as needed.
        """

        # All possible terrain type definitions from config:
        terrain_type_list = self.cfg["terrain_types"]  # e.g. a list of dicts
        n_terrain_types = len(terrain_type_list)

        for i, terrain_type_info in enumerate(terrain_type_list):
            name = str(terrain_type_info["name"])
            count = terrain_type_info["count"]  # Number of terrains of this type
            level = terrain_type_info["level"]
            size = terrain_type_info.get("size", 0.0)
            depth = terrain_type_info.get("depth", 0.0)
            system = terrain_type_info.get("system", 0)
            particles = int(terrain_type_info.get("particle_present", "False"))
            compliant = int(terrain_type_info.get("compliant", "False"))

            for j in range(count):  # Generate `count` terrains for this type
                idx = len(self.terrain_details)  # Unique terrain index
                terrain = SubTerrain(
                    "terrain",
                    width=self.width_per_env_pixels,
                    length=self.length_per_env_pixels,
                    vertical_scale=self.vertical_scale,
                    horizontal_scale=self.horizontal_scale,
                )

                # Assign terrain heightfield based on type
                if name == "flat":
                    flat_terrain(terrain, height_meters=0.0)
                elif name == "rough":
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)                
                elif name == "central_depression_terrain":

                    central_depression_terrain(
                        terrain, depression_depth=-abs(depth), platform_height=0.0, depression_size=size
                    )
                else:
                    flat_terrain(terrain, height_meters=0.0)

                # Compute terrain placement in row i, col j
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels

                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw


                # bounding region in HF indices:
                if name == "central_depression_terrain":
                    lx0 = start_x + terrain.depression_indices["start_x"]
                    lx1 = start_x + terrain.depression_indices["end_x"]
                    ly0 = start_y + terrain.depression_indices["start_y"]
                    ly1 = start_y + terrain.depression_indices["end_y"]
                else:
                    lx0, lx1, ly0, ly1 = (start_x, end_x, start_y, end_y)

                # Store the origin of the terrain
                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                center_x1 = int((self.env_length / 2 - 1) / self.horizontal_scale)
                center_x2 = int((self.env_length / 2 + 1) / self.horizontal_scale)
                center_y1 = int((self.env_width / 2 - 1) / self.horizontal_scale)
                center_y2 = int((self.env_width / 2 + 1) / self.horizontal_scale)
                env_origin_z = np.max(
                    terrain.height_field_raw[center_x1:center_x2, center_y1:center_y2]
                ) * self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

                # Convert terrain name to a number using the mapping
                terrain_label = self.terrain_mapping.get(name, -1)  # default to -1 if not found

                # Store terrain details
                self.terrain_details.append((
                    idx,
                    level,
                    i,
                    j,
                    terrain_label,
                    particles,
                    compliant,
                    system,
                    depth,
                    size,
                    lx0,
                    lx1,
                    ly0,
                    ly1,
                ))

    def full_flat_terrain(self, height_meters=0.0):
            """
            Generate flat terrain for all sub-terrains instead of random obstacles.
            """
            for k in range(self.num_maps):
                # Env coordinates in the world
                (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

                # Heightfield coordinate system from now on
                start_x = self.border + i * self.length_per_env_pixels
                end_x   = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y   = self.border + (j + 1) * self.width_per_env_pixels

                # Create a SubTerrain for this environment
                terrain = SubTerrain(
                    "terrain",
                    width=self.width_per_env_pixels,
                    length=self.width_per_env_pixels,
                    vertical_scale=self.vertical_scale,
                    horizontal_scale=self.horizontal_scale,
                )

                # Call the flat_terrain function from terrain_utils
                flat_terrain(terrain, height_meters=height_meters)

                # Copy the new flat terrain into our global height_field_raw
                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

                # Compute the average origin height for placing robots
                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                
                # For a flat terrain, the terrain is uniform, but let's still compute
                x1 = int((self.env_length / 2.0 - 1) / self.horizontal_scale)
                x2 = int((self.env_length / 2.0 + 1) / self.horizontal_scale)
                y1 = int((self.env_width  / 2.0 - 1) / self.horizontal_scale)
                y2 = int((self.env_width  / 2.0 + 1) / self.horizontal_scale)
                
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

                terrain_label = self.terrain_mapping.get("flat", 0)
                self.terrain_details.append((
                    k,         # Unique terrain index
                    0,         # level (default)
                    i,         # row index
                    j,         # column index
                    terrain_label,
                    0,         # particles (default)
                    0,         # compliant (default)
                    0,         # system (default)
                    0,         # depth (default)
                    0,         # size (default)
                    start_x,   # lx0: start index in x
                    end_x,     # lx1: end index in x
                    start_y,   # ly0: start index in y
                    end_y,     # ly1: end index in y
                ))