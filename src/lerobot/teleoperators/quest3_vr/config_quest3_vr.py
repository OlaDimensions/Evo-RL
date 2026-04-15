#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("quest3_vr")
@dataclass
class Quest3VRTeleopConfig(TeleoperatorConfig):
    """Configuration for Quest3 VR teleoperation."""

    # Quest3 input transport selection.
    # This config no longer depends on an external workspace package.
    use_wifi: bool = False
    ip_address: str = ""

    # Button mapping
    enable_button: str = "B"
    reset_button: str = "A"
    gripper_button: str = "rightTrig"

    # Control semantics

    # Input conditioning
    smooth_alpha: float = 0.2
    pos_dead: float = 0.01
    rot_dead: float = 0.026

    # Teleop action shaping
    pos_scale: float = 0.5
    rot_scale: float = 0.4

    # Gripper control
    gripper_open_value: float = 0.08
    gripper_close_value: float = 0.0
    trigger_threshold: float = 0.5

    # IK bridge enablement
    async_ik: bool = True
    reset_interp_steps: int = 25
    piper_urdf_path: str = "src/lerobot/assets/piper_description/urdf/piper_no_gripper_description.urdf"
    piper_package_dir: str = "src/lerobot/assets/piper_description"
    piper_ee_link_name: str = "ee"
    piper_ee_link_joint_name: str = "joint6"
    piper_locked_joint_names: tuple[str, str] = ()
    piper_ee_offset_xyzrpy: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.13, 0.0, -1.57, 0.0)
    arm_init_xyzrpy: tuple[float, float, float, float, float, float] = (0.19, 0.0, 0.2, 0.0, 0.0, 0.0)

    # Health & logging
    stale_input_timeout_s: float = 0.2
    safe_hold_on_stale: bool = True
    log_input_interval_s: float = 1.0
    log_action_interval_s: float = 0.5
    log_health_interval_s: float = 1.0
    log_only_on_enable: bool = True

    def __post_init__(self):
        if not (0.0 < self.smooth_alpha <= 1.0):
            raise ValueError("`smooth_alpha` must be in (0, 1].")
        if self.pos_dead < 0 or self.rot_dead < 0:
            raise ValueError("`pos_dead` and `rot_dead` must be >= 0.")
        if self.pos_scale <= 0:
            raise ValueError("`pos_scale` must be > 0.")
        if self.rot_scale <= 0:
            raise ValueError("`rot_scale` must be > 0.")
        if not isinstance(self.async_ik, bool):
            raise ValueError("`async_ik` must be true or false.")
        if self.reset_interp_steps < 1:
            raise ValueError("`reset_interp_steps` must be >= 1.")
        if not self.piper_urdf_path:
            raise ValueError("`piper_urdf_path` must be set for Quest3 VR IK.")
        if not self.piper_package_dir:
            raise ValueError("`piper_package_dir` must be set for Quest3 VR IK.")
        if self.piper_locked_joint_names and not isinstance(self.piper_locked_joint_names, tuple):
            raise ValueError("`piper_locked_joint_names` must be a tuple of joint names.")
        if not (0.0 <= self.trigger_threshold <= 1.0):
            raise ValueError("`trigger_threshold` must be in [0, 1].")
        if self.stale_input_timeout_s <= 0:
            raise ValueError("`stale_input_timeout_s` must be > 0.")
        if self.log_input_interval_s < 0:
            raise ValueError("`log_input_interval_s` must be >= 0.")
        if self.log_action_interval_s < 0:
            raise ValueError("`log_action_interval_s` must be >= 0.")
        if self.log_health_interval_s < 0:
            raise ValueError("`log_health_interval_s` must be >= 0.")
        if not isinstance(self.log_only_on_enable, bool):
            raise ValueError("`log_only_on_enable` must be true or false.")
