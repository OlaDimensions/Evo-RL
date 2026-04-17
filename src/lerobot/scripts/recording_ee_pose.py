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

"""Piper SDK end-effector pose helpers for recording storage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from lerobot.processor import RobotAction
from lerobot.utils.constants import ACTION, OBS_STATE

_SINGLE_EE_NAMES = ("ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos")
_LEFT_EE_NAMES = (
    "left_ee.x",
    "left_ee.y",
    "left_ee.z",
    "left_ee.qx",
    "left_ee.qy",
    "left_ee.qz",
    "left_ee.qw",
    "left_gripper.pos",
)
_RIGHT_EE_NAMES = (
    "right_ee.x",
    "right_ee.y",
    "right_ee.z",
    "right_ee.qx",
    "right_ee.qy",
    "right_ee.qz",
    "right_ee.qw",
    "right_gripper.pos",
)


def _milli_to_unit(value: float | int) -> float:
    return float(value) * 1e-3


def _rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw

    sr, cr = math.sin(half_roll), math.cos(half_roll)
    sp, cp = math.sin(half_pitch), math.cos(half_pitch)
    sy, cy = math.sin(half_yaw), math.cos(half_yaw)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm


def _sdk_pose_to_values(end_pose: Any, *, prefix: str = "") -> dict[str, float]:
    name_prefix = f"{prefix}_" if prefix else ""
    roll = math.radians(float(getattr(end_pose, "RX_axis")) * 1e-3)
    pitch = math.radians(float(getattr(end_pose, "RY_axis")) * 1e-3)
    yaw = math.radians(float(getattr(end_pose, "RZ_axis")) * 1e-3)
    qx, qy, qz, qw = _rpy_to_quat_xyzw(roll, pitch, yaw)
    return {
        f"{name_prefix}ee.x": float(getattr(end_pose, "X_axis")) * 1e-6,
        f"{name_prefix}ee.y": float(getattr(end_pose, "Y_axis")) * 1e-6,
        f"{name_prefix}ee.z": float(getattr(end_pose, "Z_axis")) * 1e-6,
        f"{name_prefix}ee.qx": qx,
        f"{name_prefix}ee.qy": qy,
        f"{name_prefix}ee.qz": qz,
        f"{name_prefix}ee.qw": qw,
    }


def _read_gripper_pos(arm_owner: Any) -> float:
    gripper_msg = arm_owner.arm.GetArmGripperMsgs()
    gripper_state = getattr(gripper_msg, "gripper_state", None)
    if gripper_state is None:
        raise RuntimeError("Piper SDK gripper feedback is unavailable.")
    return abs(_milli_to_unit(getattr(gripper_state, "grippers_angle", 0)))


def _read_arm_ee_pose(arm_owner: Any, *, prefix: str = "") -> dict[str, float]:
    pose_msg = arm_owner.arm.GetArmEndPoseMsgs()
    end_pose = getattr(pose_msg, "end_pose", None)
    if end_pose is None:
        raise RuntimeError("Piper SDK end-effector pose feedback is unavailable.")

    values = _sdk_pose_to_values(end_pose, prefix=prefix)
    gripper_key = f"{prefix}_gripper.pos" if prefix else "gripper.pos"
    values[gripper_key] = _read_gripper_pos(arm_owner)
    return values


def _ee_pose_feature(names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": (len(names),),
        "names": list(names),
    }


def _is_bimanual_piper(robot: Any) -> bool:
    return hasattr(robot, "left_arm") and hasattr(robot, "right_arm")


def _is_single_piper(robot: Any) -> bool:
    return hasattr(robot, "arm") and not _is_bimanual_piper(robot)


def get_piper_ee_pose_names(robot: Any) -> tuple[str, ...]:
    if _is_bimanual_piper(robot):
        return _LEFT_EE_NAMES + _RIGHT_EE_NAMES
    if _is_single_piper(robot):
        return _SINGLE_EE_NAMES
    raise ValueError(
        "`record_ee_pose=true` is only supported for Piper/PiperX follower robots, "
        f"got {type(robot).__name__}."
    )


def replace_low_dim_features_with_piper_ee_pose(dataset_features: dict[str, dict], robot: Any) -> None:
    names = get_piper_ee_pose_names(robot)
    ee_feature = _ee_pose_feature(names)

    if ACTION in dataset_features:
        dataset_features[ACTION] = dict(ee_feature)
    if OBS_STATE in dataset_features:
        dataset_features[OBS_STATE] = dict(ee_feature)
    if "complementary_info.policy_action" in dataset_features:
        dataset_features["complementary_info.policy_action"] = dict(ee_feature)


@dataclass
class PiperEEPoseStorage:
    """Read Piper SDK feedback poses in the same shape as the recording schema."""

    robot: Any
    _last_quats: dict[str, tuple[float, float, float, float]] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return get_piper_ee_pose_names(self.robot)

    def reset(self) -> None:
        self._last_quats = {}

    def _continuize_quaternion(self, values: RobotAction, *, prefix: str = "") -> RobotAction:
        if self._last_quats is None:
            self.reset()

        name_prefix = f"{prefix}_" if prefix else ""
        keys = (
            f"{name_prefix}ee.qx",
            f"{name_prefix}ee.qy",
            f"{name_prefix}ee.qz",
            f"{name_prefix}ee.qw",
        )
        quat = tuple(float(values[key]) for key in keys)
        previous = self._last_quats.get(prefix) if self._last_quats is not None else None
        if previous is not None:
            dot = sum(current * prev for current, prev in zip(quat, previous, strict=True))
            if dot < 0.0:
                quat = tuple(-value for value in quat)

        if self._last_quats is not None:
            self._last_quats[prefix] = quat
        for key, value in zip(keys, quat, strict=True):
            values[key] = value
        return values

    def read(self) -> RobotAction:
        if _is_bimanual_piper(self.robot):
            return {
                **self._continuize_quaternion(
                    _read_arm_ee_pose(self.robot.left_arm, prefix="left"), prefix="left"
                ),
                **self._continuize_quaternion(
                    _read_arm_ee_pose(self.robot.right_arm, prefix="right"), prefix="right"
                ),
            }
        if _is_single_piper(self.robot):
            return self._continuize_quaternion(_read_arm_ee_pose(self.robot))
        raise ValueError(
            "`record_ee_pose=true` is only supported for Piper/PiperX follower robots, "
            f"got {type(self.robot).__name__}."
        )

    def zero(self) -> RobotAction:
        return dict.fromkeys(self.names, 0.0)
