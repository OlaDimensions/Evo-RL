#!/usr/bin/env python

"""Helpers for end-effector pose actions."""

from __future__ import annotations

import math

import numpy as np

from lerobot.processor import RobotAction, RobotActionProcessorStep

SDK_EE_OFFSET_XYZRPY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

SINGLE_EE_RPY_NAMES = (
    "ee.x",
    "ee.y",
    "ee.z",
    "ee.roll",
    "ee.pitch",
    "ee.yaw",
    "gripper.pos",
)

BIMANUAL_EE_RPY_NAMES = (
    "left_ee.x",
    "left_ee.y",
    "left_ee.z",
    "left_ee.roll",
    "left_ee.pitch",
    "left_ee.yaw",
    "left_gripper.pos",
    "right_ee.x",
    "right_ee.y",
    "right_ee.z",
    "right_ee.roll",
    "right_ee.pitch",
    "right_ee.yaw",
    "right_gripper.pos",
)

SINGLE_EE_RXRYRZ_NAMES = (
    "ee.x",
    "ee.y",
    "ee.z",
    "ee.rx",
    "ee.ry",
    "ee.rz",
    "gripper.pos",
)

BIMANUAL_EE_RXRYRZ_NAMES = (
    "left_ee.x",
    "left_ee.y",
    "left_ee.z",
    "left_ee.rx",
    "left_ee.ry",
    "left_ee.rz",
    "left_gripper.pos",
    "right_ee.x",
    "right_ee.y",
    "right_ee.z",
    "right_ee.rx",
    "right_ee.ry",
    "right_ee.rz",
    "right_gripper.pos",
)

SINGLE_EE_QUAT_NAMES = (
    "ee.x",
    "ee.y",
    "ee.z",
    "ee.qx",
    "ee.qy",
    "ee.qz",
    "ee.qw",
    "gripper.pos",
)

BIMANUAL_EE_QUAT_NAMES = (
    "left_ee.x",
    "left_ee.y",
    "left_ee.z",
    "left_ee.qx",
    "left_ee.qy",
    "left_ee.qz",
    "left_ee.qw",
    "left_gripper.pos",
    "right_ee.x",
    "right_ee.y",
    "right_ee.z",
    "right_ee.qx",
    "right_ee.qy",
    "right_ee.qz",
    "right_ee.qw",
    "right_gripper.pos",
)

BIMANUAL_EE_SCHEMAS = (
    BIMANUAL_EE_RPY_NAMES,
    BIMANUAL_EE_RXRYRZ_NAMES,
    BIMANUAL_EE_QUAT_NAMES,
)

ACTION_SCHEMA_NAMES = {
    "single_ee_rpy": SINGLE_EE_RPY_NAMES,
    "bimanual_ee_rpy": BIMANUAL_EE_RPY_NAMES,
    "single_ee_rxryrz": SINGLE_EE_RXRYRZ_NAMES,
    "bimanual_ee_rxryrz": BIMANUAL_EE_RXRYRZ_NAMES,
    "single_ee_quat": SINGLE_EE_QUAT_NAMES,
    "bimanual_ee_quat": BIMANUAL_EE_QUAT_NAMES,
}


def detect_action_schema_from_names(
    action_names: list[str] | tuple[str, ...], joint_action_names: list[str] | tuple[str, ...]
) -> str:
    names = tuple(action_names)
    if names == tuple(joint_action_names):
        return "joint"
    for schema, schema_names in ACTION_SCHEMA_NAMES.items():
        if names == schema_names:
            return schema
    return "unknown"


def bimanual_ee_quat_to_rpy_values(values: RobotAction) -> RobotAction:
    """Return bimanual EE RPY values from bimanual EE quaternion values."""
    return {
        **_quat_arm_to_rpy_values(values, prefix="left_"),
        **_quat_arm_to_rpy_values(values, prefix="right_"),
    }


def detect_bimanual_ee_schema(action: RobotAction) -> tuple[str, ...] | None:
    """Return the matching bimanual EE action schema for a dict-like action."""
    keys = set(action)
    for schema in BIMANUAL_EE_SCHEMAS:
        if all(name in keys for name in schema):
            return schema
    return None


def infer_bimanual_ee_arm_enabled(
    action: RobotAction,
    previous_action: RobotAction | None,
    schema_names: tuple[str, ...],
    stationary_arm_delta_threshold: float,
) -> tuple[bool, bool]:
    """Infer per-arm enabled flags from left/right action deltas."""
    if previous_action is None:
        return True, True

    arm_width = len(schema_names) // 2
    left_delta = _max_abs_delta(action, previous_action, schema_names[:arm_width])
    right_delta = _max_abs_delta(action, previous_action, schema_names[arm_width:])
    return left_delta > stationary_arm_delta_threshold, right_delta > stationary_arm_delta_threshold


def with_bimanual_ee_enabled_flags(
    action: RobotAction,
    previous_action: RobotAction | None,
    stationary_arm_delta_threshold: float,
    *,
    schema_names: tuple[str, ...] | None = None,
) -> RobotAction:
    """Return a copy with `left_enabled/right_enabled` when action is a bimanual EE pose."""
    schema_names = schema_names or detect_bimanual_ee_schema(action)
    if schema_names is None:
        return action

    left_enabled, right_enabled = infer_bimanual_ee_arm_enabled(
        action, previous_action, schema_names, stationary_arm_delta_threshold
    )
    return {**action, "left_enabled": left_enabled, "right_enabled": right_enabled}


def _max_abs_delta(action: RobotAction, previous_action: RobotAction, names: tuple[str, ...]) -> float:
    return max(abs(float(action[name]) - float(previous_action[name])) for name in names)


class MapPolicyRPYActionToIKTargetsStep(RobotActionProcessorStep):
    """Map policy EE RPY fields into the absolute target keys consumed by the IK processor."""

    def __init__(self, *, bimanual: bool = False):
        self.bimanual = bimanual

    def action(self, action: RobotAction) -> RobotAction:
        if self.bimanual:
            return {
                **self._map_arm(action, input_prefix="left_", output_prefix="left_"),
                **self._map_arm(action, input_prefix="right_", output_prefix="right_"),
            }
        return self._map_arm(action, input_prefix="", output_prefix="")

    @staticmethod
    def _map_arm(action: RobotAction, *, input_prefix: str, output_prefix: str) -> RobotAction:
        return {
            f"{output_prefix}enabled": bool(action.get(f"{output_prefix}enabled", True)),
            f"{output_prefix}reset": bool(action.get(f"{output_prefix}reset", False)),
            f"{output_prefix}ee.target_x": float(action[f"{input_prefix}ee.x"]),
            f"{output_prefix}ee.target_y": float(action[f"{input_prefix}ee.y"]),
            f"{output_prefix}ee.target_z": float(action[f"{input_prefix}ee.z"]),
            f"{output_prefix}ee.target_rx": float(action[f"{input_prefix}ee.roll"]),
            f"{output_prefix}ee.target_ry": float(action[f"{input_prefix}ee.pitch"]),
            f"{output_prefix}ee.target_rz": float(action[f"{input_prefix}ee.yaw"]),
            f"{output_prefix}gripper.pos": float(action[f"{input_prefix}gripper.pos"]),
        }

    def transform_features(self, features):
        return features


class MapPolicyRXRYRZActionToRPYStep(RobotActionProcessorStep):
    """Convert XYZ Euler policy fields to RPY fields."""

    def __init__(self, *, bimanual: bool = False):
        self.bimanual = bimanual

    def action(self, action: RobotAction) -> RobotAction:
        if self.bimanual:
            return {
                **self._map_arm(action, prefix="left_"),
                **self._map_arm(action, prefix="right_"),
            }
        return self._map_arm(action, prefix="")

    @staticmethod
    def _map_arm(action: RobotAction, *, prefix: str) -> RobotAction:
        roll, pitch, yaw = _rxryrz_to_rpy(
            float(action[f"{prefix}ee.rx"]),
            float(action[f"{prefix}ee.ry"]),
            float(action[f"{prefix}ee.rz"]),
        )
        return {
            f"{prefix}ee.x": float(action[f"{prefix}ee.x"]),
            f"{prefix}ee.y": float(action[f"{prefix}ee.y"]),
            f"{prefix}ee.z": float(action[f"{prefix}ee.z"]),
            f"{prefix}ee.roll": roll,
            f"{prefix}ee.pitch": pitch,
            f"{prefix}ee.yaw": yaw,
            f"{prefix}gripper.pos": float(action[f"{prefix}gripper.pos"]),
        }

    def transform_features(self, features):
        return features


class MapPolicyQuatActionToRPYStep(RobotActionProcessorStep):
    """Convert quaternion policy fields to RPY fields."""

    def __init__(self, *, bimanual: bool = False):
        self.bimanual = bimanual

    def action(self, action: RobotAction) -> RobotAction:
        if self.bimanual:
            return {
                **self._map_arm(action, prefix="left_"),
                **self._map_arm(action, prefix="right_"),
            }
        return self._map_arm(action, prefix="")

    @staticmethod
    def _map_arm(action: RobotAction, *, prefix: str) -> RobotAction:
        return _quat_arm_to_rpy_values(action, prefix=prefix)

    def transform_features(self, features):
        return features


def _rxryrz_to_rpy(rx: float, ry: float, rz: float) -> tuple[float, float, float]:
    ca, sa = math.cos(rx), math.sin(rx)
    cb, sb = math.cos(ry), math.sin(ry)
    cc, sc = math.cos(rz), math.sin(rz)
    matrix = np.array(
        [
            [cb * cc, -cb * sc, sb],
            [sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb],
            [-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb],
        ],
        dtype=np.float64,
    )
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return roll, pitch, yaw


def _quat_arm_to_rpy_values(action: RobotAction, *, prefix: str) -> RobotAction:
    roll, pitch, yaw = _quat_xyzw_to_rpy(
        float(action[f"{prefix}ee.qx"]),
        float(action[f"{prefix}ee.qy"]),
        float(action[f"{prefix}ee.qz"]),
        float(action[f"{prefix}ee.qw"]),
    )
    return {
        f"{prefix}ee.x": float(action[f"{prefix}ee.x"]),
        f"{prefix}ee.y": float(action[f"{prefix}ee.y"]),
        f"{prefix}ee.z": float(action[f"{prefix}ee.z"]),
        f"{prefix}ee.roll": roll,
        f"{prefix}ee.pitch": pitch,
        f"{prefix}ee.yaw": yaw,
        f"{prefix}gripper.pos": float(action[f"{prefix}gripper.pos"]),
    }


def _quat_xyzw_to_rpy(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Invalid EE quaternion target: expected finite non-zero [qx, qy, qz, qw].")
    qx, qy, qz, qw = (quat / norm).tolist()

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw
