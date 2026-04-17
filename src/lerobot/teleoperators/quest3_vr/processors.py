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

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.teleoperators.bi_quest3_vr.config_bi_quest3_vr import BiQuest3VRTeleopConfig
from lerobot.teleoperators.quest3_vr.config_quest3_vr import Quest3VRTeleopConfig

from .ee_to_joint_ik import DualArmEEToJointIKProcessorStep, EEToJointIKProcessorStep
from .piper_pinocchio import PiperIKConfig, PiperPinocchioIKBackend


def _build_ik_cfg(cfg: Quest3VRTeleopConfig) -> PiperIKConfig:
    return PiperIKConfig(
        urdf_path=Path(cfg.piper_urdf_path),
        package_dir=Path(cfg.piper_package_dir),
        ee_link_name=cfg.piper_ee_link_name,
        ee_link_joint_name=cfg.piper_ee_link_joint_name,
        locked_joint_names=cfg.piper_locked_joint_names,
        ee_offset_xyzrpy=cfg.piper_ee_offset_xyzrpy,
        pose_error_mode=cfg.ik_pose_error_mode,
        max_position_error_m=cfg.ik_max_position_error_m,
        max_orientation_error_rad=math.radians(cfg.ik_max_orientation_error_deg),
        pose_error_log_interval_s=cfg.ik_pose_error_log_interval_s,
    )


def make_quest3_vr_robot_action_processor_from_config(cfg: Quest3VRTeleopConfig) -> RobotProcessorPipeline:
    """Build the single-arm Quest3 VR robot action processor chain."""
    ik_cfg = _build_ik_cfg(cfg)
    arm_init_T = _xyzrpy_to_matrix(*cfg.arm_init_xyzrpy)
    backend = PiperPinocchioIKBackend(ik_cfg)
    step = EEToJointIKProcessorStep(
        ik_backend=backend,
        arm_init_T=arm_init_T,
        dead_zone_pos=cfg.pos_dead,
        dead_zone_rot=cfg.rot_dead,
        async_solve=cfg.async_ik,
        joint_smooth_alpha=cfg.joint_smooth_alpha,
        max_joint_step_rad=math.radians(cfg.max_joint_step_deg),
        target_retry_segment_counts=cfg.target_retry_segment_counts,
        reset_interp_steps=cfg.reset_interp_steps,
        reset_target=cfg.reset_target,
        reset_joint_target_degrees=cfg.reset_joint_target_degrees,
        reset_joint_tolerance_rad=math.radians(cfg.reset_joint_tolerance_deg),
        reset_settle_ticks=cfg.reset_settle_ticks,
        reset_timeout_s=cfg.reset_timeout_s,
        diagnostics_interval_s=cfg.log_diagnostics_interval_s,
    )
    return RobotProcessorPipeline(
        steps=[step],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def make_bi_quest3_vr_robot_action_processor_from_config(cfg: BiQuest3VRTeleopConfig) -> RobotProcessorPipeline:
    """Build the bimanual Quest3 VR robot action processor chain."""
    ik_cfg = _build_ik_cfg(cfg)
    arm_init_T = _xyzrpy_to_matrix(*cfg.arm_init_xyzrpy)
    left_backend = PiperPinocchioIKBackend(ik_cfg)
    right_backend = PiperPinocchioIKBackend(ik_cfg)
    step = DualArmEEToJointIKProcessorStep(
        left_step=EEToJointIKProcessorStep(
            ik_backend=left_backend,
            arm_init_T=arm_init_T,
            dead_zone_pos=cfg.pos_dead,
            dead_zone_rot=cfg.rot_dead,
            input_prefix="left_",
            output_prefix="left_",
            enable_key="enabled",
            reset_key="reset",
            gripper_key="gripper.pos",
            async_solve=cfg.async_ik,
            joint_smooth_alpha=cfg.joint_smooth_alpha,
            max_joint_step_rad=math.radians(cfg.max_joint_step_deg),
            target_retry_segment_counts=cfg.target_retry_segment_counts,
            reset_interp_steps=cfg.reset_interp_steps,
            reset_target=cfg.reset_target,
            reset_joint_target_degrees=cfg.reset_joint_target_degrees,
            reset_joint_tolerance_rad=math.radians(cfg.reset_joint_tolerance_deg),
            reset_settle_ticks=cfg.reset_settle_ticks,
            reset_timeout_s=cfg.reset_timeout_s,
            diagnostics_interval_s=cfg.log_diagnostics_interval_s,
        ),
        right_step=EEToJointIKProcessorStep(
            ik_backend=right_backend,
            arm_init_T=arm_init_T,
            dead_zone_pos=cfg.pos_dead,
            dead_zone_rot=cfg.rot_dead,
            input_prefix="right_",
            output_prefix="right_",
            enable_key="enabled",
            reset_key="reset",
            gripper_key="gripper.pos",
            async_solve=cfg.async_ik,
            joint_smooth_alpha=cfg.joint_smooth_alpha,
            max_joint_step_rad=math.radians(cfg.max_joint_step_deg),
            target_retry_segment_counts=cfg.target_retry_segment_counts,
            reset_interp_steps=cfg.reset_interp_steps,
            reset_target=cfg.reset_target,
            reset_joint_target_degrees=cfg.reset_joint_target_degrees,
            reset_joint_tolerance_rad=math.radians(cfg.reset_joint_tolerance_deg),
            reset_settle_ticks=cfg.reset_settle_ticks,
            reset_timeout_s=cfg.reset_timeout_s,
            diagnostics_interval_s=cfg.log_diagnostics_interval_s,
        ),
    )
    return RobotProcessorPipeline(
        steps=[step],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def _xyzrpy_to_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    ca, sa = math.cos(yaw), math.sin(yaw)
    cb, sb = math.cos(pitch), math.sin(pitch)
    cc, sc = math.cos(roll), math.sin(roll)
    T[0] = [ca * cb, ca * sb * sc - sa * cc, sa * sc + ca * sb * cc, x]
    T[1] = [sa * cb, ca * cc + sa * sb * sc, sa * sb * cc - ca * sc, y]
    T[2] = [-sb, cb * sc, cb * cc, z]
    return T
