# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional
import os

import numpy as np
import omni
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.sensor import _sensor
from pxr import PhysxSchema


class A1(Robot):
    """
    A minimal Unitree A1 robot class, referencing the A1 USD in Isaac Sim.
    No state-setting methods (e.g., set_state) are included here.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "a1",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the Unitree A1 robot in Isaac Sim.

        Args:
            prim_path (str): The prim path where the A1 robot is added in the stage.
            name (str, optional): The name of the robot.
            usd_path (str, optional): Path to a custom A1 USD file if desired.
            translation (np.ndarray, optional): Initial world position of the robot [x, y, z].
            orientation (np.ndarray, optional): Initial world orientation as [x, y, z, w] quaternion.
        """
        self._usd_path = usd_path
        self._name = name
        self._prim_path = prim_path


        if self._usd_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self._usd_path = os.path.join(script_dir, "../asset/Unitree/a1.usd")

        # Add the A1 reference to the stage at the specified prim path
        add_reference_to_stage(self._usd_path, prim_path)

        # The A1 has 12 actuated joints
        self._dof_names = [
        'FL_hip_joint',   'FR_hip_joint',   'RL_hip_joint',   'RR_hip_joint',
        'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
        'FL_calf_joint',  'FR_calf_joint',  'RL_calf_joint',  'RR_calf_joint'
        ]
        # contact sensor setup
        self.feet_path = [
            self._prim_path + "/FL_foot",
            self._prim_path + "/FR_foot",
            self._prim_path + "/RL_foot",
            self._prim_path + "/RR_foot",
        ]

        for i in range(4):
            omni.kit.commands.execute(
                "IsaacSensorCreateContactSensor",
                path="/sensor",
                parent=self.feet_path[i],
                min_threshold=0,
                max_threshold=1000000,
                radius=-1,
                sensor_period=-1,
            )

        self.foot_force = np.zeros(4)
        self.enable_foot_filter = True
        self._cs = _sensor.acquire_contact_sensor_interface()

        # Initialize the Robot base class
        super().__init__(
            prim_path=prim_path,
            name=self._name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

    @property
    def dof_names(self):
        """
        Returns:
            list of str: The list of DOF (joint) names for the A1 robot.
        """
        return self._dof_names

    @property
    def foot_forces(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: A 4-element array of the latest foot contact forces 
            [FL_force, FR_force, RL_force, RR_force].
        """
        return self._foot_forces

    def update_contact_sensor_data(self) -> None:
        """[summary]

        Updates processed contact sensor data from the robot feets, store them in member variable foot_force
        """
        # Order: FL, FR, BL, BR
        for i in range(len(self.feet_path)):
            sensor_reading = self._cs.get_sensor_reading(self.feet_path[i] + "/sensor")
            if sensor_reading.is_valid:
                if self.enable_foot_filter:
                    self.foot_force[i] = self.foot_force[i] * 0.9 + sensor_reading.value * 0.1
                else:
                    self.foot_force[i] = sensor_reading.value

    def set_a1_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(False)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetLinearDampingAttr().Set(0.0)
                rb.GetMaxLinearVelocityAttr().Set(1000.0)
                rb.GetAngularDampingAttr().Set(0.0)
                rb.GetMaxAngularVelocityAttr().Set(1000.0)
                rb.GetMaxDepenetrationVelocityAttr().Set(1.0)
                print("Setting A1 properties")


    def prepare_contacts(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                if "_hip" not in str(link_prim.GetPrimPath()):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0)
                    if not link_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                        print(f"Adding contact report API to prim: '{link_prim.GetPrimPath()}'")
                        cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    else:
                        print(f"Contact report API already exists on prim: '{link_prim.GetPrimPath()}'")
                        cr_api = PhysxSchema.PhysxContactReportAPI.Get(stage, link_prim.GetPrimPath())
                    # set threshold to zero
                    cr_api.CreateThresholdAttr().Set(0)