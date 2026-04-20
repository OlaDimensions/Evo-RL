#!/usr/bin/env python

"""Helpers for bimanual end-effector pose actions."""

from __future__ import annotations

from lerobot.processor import RobotAction

SDK_EE_OFFSET_XYZRPY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

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
