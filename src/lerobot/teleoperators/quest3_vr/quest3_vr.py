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

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pinocchio as pin

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_quest3_vr import Quest3VRTeleopConfig

logger = logging.getLogger(__name__)


@dataclass
class ArmRuntime:
    smooth_pose: np.ndarray | None = None
    base_pose: np.ndarray | None = None
    enable_prev: bool = False
    reset_prev: bool = False
    trig_prev: bool = False
    gripper_open: bool = True
    arm_T: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    last_T: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))


class Quest3VRTeleop(Teleoperator):
    """Quest3 VR single-arm teleoperator."""

    config_class = Quest3VRTeleopConfig
    name = "quest3_vr"

    def __init__(self, config: Quest3VRTeleopConfig):
        super().__init__(config)
        self.config = config
        self._reader: Any | None = None
        self._is_connected = False
        self._right = ArmRuntime()
        self._last_input_t = 0.0
        self._last_health_log_t = 0.0
        self._last_input_log_t = 0.0
        self._last_action_log_t = 0.0
        self._hz_window_start_t = 0.0
        self._frame_count = 0

    @property
    def action_features(self) -> dict:
        return {
            "enabled": bool,
            "reset": bool,
            "ee.delta_x": float,
            "ee.delta_y": float,
            "ee.delta_z": float,
            "ee.delta_rx": float,
            "ee.delta_ry": float,
            "ee.delta_rz": float,
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        from .oculus_reader import OculusReader

        if self.config.use_wifi and self.config.ip_address:
            self._reader = OculusReader(ip_address=self.config.ip_address)
            logger.info("[VR_HEALTH] connected OculusReader in WiFi mode ip=%s", self.config.ip_address)
        else:
            self._reader = OculusReader()
            logger.info("[VR_HEALTH] connected OculusReader in USB mode")

        now = time.monotonic()
        self._last_input_t = now
        self._hz_window_start_t = now
        self._is_connected = True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        now = time.monotonic()
        transforms, buttons = self._reader.get_transformations_and_buttons()

        right_pose = self._extract_pose(transforms, "r")
        if right_pose is None:
            age_s = now - self._last_input_t
            self._log_health(now, f"missing controller pose; age={age_s:.3f}s")
            return self._hold_action(gripper_open=self._right.gripper_open, enabled=False, reset=False)
        self._last_input_t = now

        action = self._arm_action(
            pose=right_pose,
            buttons=buttons,
            state=self._right,
            enable_button=self.config.enable_button,
            reset_button=self.config.reset_button,
            gripper_button=self.config.gripper_button,
        )

        self._log_input(now, buttons)
        self._log_action(now, action)
        self._update_hz(now)
        return action

    @check_if_not_connected
    def disconnect(self) -> None:
        self._reader = None
        self._is_connected = False
        logger.info("[VR_HEALTH] quest3_vr disconnected")

    def send_feedback(self, feedback: dict[str, Any]) -> None:  # noqa: ARG002
        return

    def _arm_action(
        self,
        *,
        pose: np.ndarray | None,
        buttons: dict[str, Any],
        state: ArmRuntime,
        enable_button: str,
        reset_button: str,
        gripper_button: str,
    ) -> RobotAction:
        if pose is None:
            return self._hold_action(gripper_open=state.gripper_open, enabled=False, reset=False)

        pose = np.asarray(pose, dtype=np.float64)
        state.smooth_pose = self._ema(pose, state.smooth_pose, self.config.smooth_alpha)
        pose = state.smooth_pose

        enable_now = bool(buttons.get(enable_button, False))
        reset_now = bool(buttons.get(reset_button, False))
        reset_edge = reset_now and not state.reset_prev
        trig_now = self._extract_trigger(buttons, gripper_button) > self.config.trigger_threshold

        if reset_edge:
            state.base_pose = pose.copy()
            state.arm_T = np.eye(4, dtype=np.float64)
            state.last_T = np.eye(4, dtype=np.float64)
        state.reset_prev = reset_now

        if enable_now and not state.enable_prev:
            state.base_pose = pose.copy()
            state.arm_T = state.last_T.copy()
        elif (not enable_now) and state.enable_prev:
            state.arm_T = state.last_T.copy()
        state.enable_prev = enable_now

        if trig_now and not state.trig_prev:
            state.gripper_open = not state.gripper_open
        state.trig_prev = trig_now

        if not enable_now:
            return self._hold_action(gripper_open=state.gripper_open, enabled=False, reset=reset_edge)

        if state.base_pose is None:
            state.base_pose = pose.copy()

        raw_dp = pose[:3] - state.base_pose[:3]
        raw_dr = pose[3:] - state.base_pose[3:]
        in_dead_zone = float(np.max(np.abs(raw_dp))) < self.config.pos_dead and float(np.max(np.abs(raw_dr))) < self.config.rot_dead
        if in_dead_zone:
            return self._hold_action(gripper_open=state.gripper_open, enabled=True, reset=reset_edge)

        current_T = self._xyzrpy_to_matrix(*pose.tolist())
        base_T = self._xyzrpy_to_matrix(*state.base_pose.tolist())
        delta_pos_raw = current_T[:3, 3] - base_T[:3, 3]
        dead_band = np.array([0.0, 0.018, 0.0], dtype=np.float64)
        delta_pos = np.sign(delta_pos_raw) * np.maximum(np.abs(delta_pos_raw) - dead_band, 0.0) * self.config.pos_scale
        delta_rot = self._rotation_vector_from_matrix(base_T[:3, :3].T @ current_T[:3, :3]) * self.config.rot_scale

        target_T = state.arm_T.copy()
        target_T[:3, 3] = target_T[:3, 3] + delta_pos
        target_T[:3, :3] = target_T[:3, :3] @ pin.exp3(delta_rot)
        step_pos = target_T[:3, 3] - state.last_T[:3, 3]
        step_rot = self._rotation_vector_from_matrix(state.last_T[:3, :3].T @ target_T[:3, :3])
        state.last_T = target_T.copy()

        return {
            "enabled": True,
            "reset": reset_edge,
            "ee.delta_x": float(step_pos[0]),
            "ee.delta_y": float(step_pos[1]),
            "ee.delta_z": float(step_pos[2]),
            "ee.delta_rx": float(step_rot[0]),
            "ee.delta_ry": float(step_rot[1]),
            "ee.delta_rz": float(step_rot[2]),
            "gripper.pos": float(self.config.gripper_open_value if state.gripper_open else self.config.gripper_close_value),
        }

    def _hold_action(self, *, gripper_open: bool, enabled: bool, reset: bool) -> RobotAction:
        return {
            "enabled": enabled,
            "reset": reset,
            "ee.delta_x": 0.0,
            "ee.delta_y": 0.0,
            "ee.delta_z": 0.0,
            "ee.delta_rx": 0.0,
            "ee.delta_ry": 0.0,
            "ee.delta_rz": 0.0,
            "gripper.pos": float(self.config.gripper_open_value if gripper_open else self.config.gripper_close_value),
        }

    @staticmethod
    def _extract_trigger(buttons: dict[str, Any], key: str) -> float:
        value = buttons.get(key, [0.0])
        if isinstance(value, (list, tuple)) and value:
            return float(value[0])
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def _extract_pose(self, transforms: dict[str, np.ndarray], key: str) -> np.ndarray | None:
        T = transforms.get(key)
        if T is None:
            return None
        arr = np.asarray(T, dtype=np.float64)
        if arr.shape != (4, 4) or np.isnan(arr).any():
            return None
        arr = self._adjust_frame(arr)
        x, y, z = arr[0, 3], arr[1, 3], arr[2, 3]
        roll = math.atan2(arr[2, 1], arr[2, 2])
        pitch = math.asin(np.clip(-arr[2, 0], -1.0, 1.0))
        yaw = math.atan2(arr[1, 0], arr[0, 0])
        return np.array([float(x), float(y), float(z), float(roll), float(pitch), float(yaw)], dtype=np.float64)

    def _adjust_frame(self, T: np.ndarray) -> np.ndarray:
        adj = np.array(
            [[0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        r_adj = self._xyzrpy_to_matrix(0.0, 0.0, 0.0, -math.pi, 0.0, -math.pi / 2.0)
        return adj @ T @ r_adj

    @staticmethod
    def _ema(raw: np.ndarray, prev: np.ndarray | None, alpha: float) -> np.ndarray:
        if prev is None:
            return raw.copy()
        return alpha * raw + (1.0 - alpha) * prev

    def _log_input(self, now: float, buttons: dict[str, Any]) -> None:
        if now - self._last_input_log_t < self.config.log_input_interval_s:
            return
        self._last_input_log_t = now
        logger.info(
            "[VR_INPUT] B=%s A=%s trig=%.2f",
            bool(buttons.get(self.config.enable_button, False)),
            bool(buttons.get(self.config.reset_button, False)),
            self._extract_trigger(buttons, self.config.gripper_button),
        )

    def _log_action(self, now: float, action: RobotAction) -> None:
        if now - self._last_action_log_t < self.config.log_action_interval_s:
            return
        self._last_action_log_t = now
        logger.info("[VR_ACTION] keys=%s", sorted(action.keys()))

    def _log_health(self, now: float, message: str) -> None:
        if now - self._last_health_log_t < self.config.log_health_interval_s:
            return
        self._last_health_log_t = now
        logger.warning("[VR_HEALTH] %s", message)

    def _update_hz(self, now: float) -> None:
        self._frame_count += 1
        span = now - self._hz_window_start_t
        if span >= 1.0:
            logger.info("[VR_HEALTH] output_hz=%.1f", self._frame_count / span)
            self._frame_count = 0
            self._hz_window_start_t = now

    @staticmethod
    def _xyzrpy_to_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        ca, sa = math.cos(yaw), math.sin(yaw)
        cb, sb = math.cos(pitch), math.sin(pitch)
        cc, sc = math.cos(roll), math.sin(roll)
        T[0] = [ca * cb, ca * sb * sc - sa * cc, sa * sc + ca * sb * cc, x]
        T[1] = [sa * cb, ca * cc + sa * sb * sc, sa * sb * cc - ca * sc, y]
        T[2] = [-sb, cb * sc, cb * cc, z]
        return T

    @staticmethod
    def _rotation_vector_from_matrix(R: np.ndarray) -> np.ndarray:
        cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        theta = math.acos(cos_theta)
        if theta < 1e-9:
            return np.zeros(3, dtype=np.float64)
        sin_theta = math.sin(theta)
        if abs(sin_theta) < 1e-12:
            return np.zeros(3, dtype=np.float64)
        rx = (R[2, 1] - R[1, 2]) / (2.0 * sin_theta)
        ry = (R[0, 2] - R[2, 0]) / (2.0 * sin_theta)
        rz = (R[1, 0] - R[0, 1]) / (2.0 * sin_theta)
        return theta * np.array([rx, ry, rz], dtype=np.float64)
