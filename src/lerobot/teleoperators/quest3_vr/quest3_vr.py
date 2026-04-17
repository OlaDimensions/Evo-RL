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
    raw_T: np.ndarray | None = None
    smooth_T: np.ndarray | None = None
    base_T: np.ndarray | None = None
    enable_prev: bool = False
    reset_prev: bool = False
    trig_prev: bool = False
    gripper_open: bool = True
    gripper_pos: float | None = None
    arm_T: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    last_T: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    last_motion_diag_log_t: float = 0.0
    pending_episode_reset: bool = False


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
        self._arm_init_T = self._xyzrpy_to_matrix(*self.config.arm_init_xyzrpy)
        self._right.arm_T = self._arm_init_T.copy()
        self._right.last_T = self._arm_init_T.copy()
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
            "ee.target_x": float,
            "ee.target_y": float,
            "ee.target_z": float,
            "ee.target_rx": float,
            "ee.target_ry": float,
            "ee.target_rz": float,
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

        right_T = self._extract_transform(transforms, "r")
        if right_T is None:
            if self._right.pending_episode_reset:
                return self._arm_action(
                    controller_T=None,
                    buttons=buttons,
                    state=self._right,
                    enable_button=self.config.enable_button,
                    reset_button=self.config.reset_button,
                    gripper_button=self.config.gripper_button,
                )
            age_s = now - self._last_input_t
            self._log_health(now, f"missing controller pose; age={age_s:.3f}s")
            return self._hold_action(
                gripper_open=self._right.gripper_open,
                enabled=False,
                reset=False,
                gripper_pos=self._right.gripper_pos,
            )
        self._last_input_t = now

        action = self._arm_action(
            controller_T=right_T,
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
        if self._reader is not None:
            try:
                self._reader.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[VR_HEALTH] failed to stop OculusReader cleanly: %s", exc)
        self._reader = None
        self._is_connected = False
        logger.info("[VR_HEALTH] quest3_vr disconnected")

    def send_feedback(self, feedback: dict[str, Any]) -> None:  # noqa: ARG002
        return

    def prepare_episode_reset(self) -> None:
        """Make the next action trigger a safe IK reset to the configured initial pose."""
        self._prepare_arm_episode_reset(self._right)

    def prepare_episode_start(self) -> None:
        """Drop retained VR pose anchors before a new recorded episode starts."""
        self._clear_arm_episode_state(self._right)

    def _arm_action(
        self,
        *,
        controller_T: np.ndarray | None,
        buttons: dict[str, Any],
        state: ArmRuntime,
        enable_button: str,
        reset_button: str,
        gripper_button: str,
    ) -> RobotAction:
        if state.pending_episode_reset:
            state.pending_episode_reset = False
            return self._episode_reset_action(state)

        if controller_T is None:
            return self._hold_action(
                gripper_open=state.gripper_open,
                enabled=False,
                reset=False,
                gripper_pos=state.gripper_pos,
            )

        raw_T = np.asarray(controller_T, dtype=np.float64)
        raw_step_pos, raw_step_rot = self._pose_step(state.raw_T, raw_T)
        state.raw_T = raw_T.copy()

        prev_smooth_T = state.smooth_T
        state.smooth_T = self._smooth_transform(raw_T, state.smooth_T, self.config.smooth_alpha)
        controller_T = state.smooth_T
        smooth_step_pos, smooth_step_rot = self._pose_step(prev_smooth_T, controller_T)

        enable_now = bool(buttons.get(enable_button, False))
        reset_now = bool(buttons.get(reset_button, False))
        reset_edge = reset_now and not state.reset_prev
        trig_now = self._extract_trigger(buttons, gripper_button) > self.config.trigger_threshold

        if reset_edge:
            state.base_T = controller_T.copy()
            state.arm_T = self._arm_init_T.copy()
            state.last_T = self._arm_init_T.copy()
            state.gripper_pos = float(self.config.gripper_reset_value)
        state.reset_prev = reset_now

        if enable_now and not state.enable_prev:
            state.base_T = controller_T.copy()
            state.arm_T = state.last_T.copy()
            # If trigger is already held when entering teleop, treat it as a fresh
            # press so B+trigger "simultaneous" operation still toggles gripper.
            state.trig_prev = False
        elif (not enable_now) and state.enable_prev:
            state.arm_T = state.last_T.copy()
        state.enable_prev = enable_now

        # Toggle gripper only during active teleoperation (B/enable held).
        if enable_now and trig_now and not state.trig_prev:
            state.gripper_open = not state.gripper_open
            state.gripper_pos = self._gripper_value(state.gripper_open)
        state.trig_prev = trig_now

        if not enable_now:
            return self._hold_action(
                gripper_open=state.gripper_open,
                enabled=False,
                reset=reset_edge,
                gripper_pos=state.gripper_pos,
            )

        if state.base_T is None:
            state.base_T = controller_T.copy()

        raw_dp = controller_T[:3, 3] - state.base_T[:3, 3]
        raw_dr = self._rotation_vector_from_matrix(state.base_T[:3, :3].T @ controller_T[:3, :3])
        in_dead_zone = float(np.max(np.abs(raw_dp))) < self.config.pos_dead and float(np.max(np.abs(raw_dr))) < self.config.rot_dead
        if in_dead_zone:
            self._log_motion_diagnostics(
                state=state,
                enable_now=enable_now,
                reset_edge=reset_edge,
                in_dead_zone=in_dead_zone,
                raw_step_pos=raw_step_pos,
                raw_step_rot=raw_step_rot,
                smooth_step_pos=smooth_step_pos,
                smooth_step_rot=smooth_step_rot,
                raw_dp=raw_dp,
                raw_dr=raw_dr,
                delta_pos=np.zeros(3, dtype=np.float64),
                delta_rot=np.zeros(3, dtype=np.float64),
                target_xyzrpy=None,
            )
            return self._hold_action(
                gripper_open=state.gripper_open,
                enabled=True,
                reset=reset_edge,
                gripper_pos=state.gripper_pos,
            )

        delta_pos = raw_dp * self.config.pos_scale
        delta_rot = raw_dr * self.config.rot_scale

        target_T = state.arm_T.copy()
        target_T[:3, 3] = target_T[:3, 3] + delta_pos
        target_T[:3, :3] = target_T[:3, :3] @ pin.exp3(delta_rot)
        state.last_T = target_T.copy()
        target_xyzrpy = self._matrix_to_xyzrpy(target_T)
        self._log_motion_diagnostics(
            state=state,
            enable_now=enable_now,
            reset_edge=reset_edge,
            in_dead_zone=in_dead_zone,
            raw_step_pos=raw_step_pos,
            raw_step_rot=raw_step_rot,
            smooth_step_pos=smooth_step_pos,
            smooth_step_rot=smooth_step_rot,
            raw_dp=raw_dp,
            raw_dr=raw_dr,
            delta_pos=delta_pos,
            delta_rot=delta_rot,
            target_xyzrpy=target_xyzrpy,
        )

        return {
            "enabled": True,
            "reset": reset_edge,
            "ee.target_x": float(target_xyzrpy[0]),
            "ee.target_y": float(target_xyzrpy[1]),
            "ee.target_z": float(target_xyzrpy[2]),
            "ee.target_rx": float(target_xyzrpy[3]),
            "ee.target_ry": float(target_xyzrpy[4]),
            "ee.target_rz": float(target_xyzrpy[5]),
            "ee.delta_x": 0.0,
            "ee.delta_y": 0.0,
            "ee.delta_z": 0.0,
            "ee.delta_rx": 0.0,
            "ee.delta_ry": 0.0,
            "ee.delta_rz": 0.0,
            "gripper.pos": self._current_gripper_value(state),
        }

    def _prepare_arm_episode_reset(self, state: ArmRuntime) -> None:
        self._clear_arm_episode_state(state)
        state.arm_T = self._arm_init_T.copy()
        state.last_T = self._arm_init_T.copy()
        state.gripper_pos = float(self.config.gripper_reset_value)
        state.pending_episode_reset = True

    def _clear_arm_episode_state(self, state: ArmRuntime) -> None:
        state.raw_T = None
        state.smooth_T = None
        state.base_T = None
        state.enable_prev = False
        state.reset_prev = False
        state.trig_prev = False
        state.arm_T = self._arm_init_T.copy()
        state.last_T = self._arm_init_T.copy()
        state.pending_episode_reset = False

    def _episode_reset_action(self, state: ArmRuntime) -> RobotAction:
        state.arm_T = self._arm_init_T.copy()
        state.last_T = self._arm_init_T.copy()
        state.base_T = None
        state.enable_prev = False
        state.reset_prev = False
        state.trig_prev = False
        state.gripper_pos = float(self.config.gripper_reset_value)
        return self._hold_action(
            gripper_open=state.gripper_open,
            enabled=False,
            reset=True,
            gripper_pos=state.gripper_pos,
        )

    def _hold_action(
        self,
        *,
        gripper_open: bool,
        enabled: bool,
        reset: bool,
        gripper_pos: float | None = None,
    ) -> RobotAction:
        gripper_pos = self._gripper_value(gripper_open) if gripper_pos is None else float(gripper_pos)
        return {
            "enabled": enabled,
            "reset": reset,
            "ee.delta_x": 0.0,
            "ee.delta_y": 0.0,
            "ee.delta_z": 0.0,
            "ee.delta_rx": 0.0,
            "ee.delta_ry": 0.0,
            "ee.delta_rz": 0.0,
            "gripper.pos": gripper_pos,
        }

    def _current_gripper_value(self, state: ArmRuntime) -> float:
        if state.gripper_pos is not None:
            return float(state.gripper_pos)
        return self._gripper_value(state.gripper_open)

    def _gripper_value(self, gripper_open: bool) -> float:
        return float(self.config.gripper_open_value if gripper_open else self.config.gripper_close_value)

    @staticmethod
    def _extract_trigger(buttons: dict[str, Any], key: str) -> float:
        value = buttons.get(key)
        if value is None:
            fallback_keys = {
                "rightTrig": "RTr",
                "leftTrig": "LTr",
                "rightGrip": "RG",
                "leftGrip": "LG",
            }
            value = buttons.get(fallback_keys.get(key, ""), [0.0])
        if isinstance(value, (list, tuple)) and value:
            return float(value[0])
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def _extract_transform(self, transforms: dict[str, np.ndarray], key: str) -> np.ndarray | None:
        T = transforms.get(key)
        if T is None:
            return None
        arr = np.asarray(T, dtype=np.float64)
        if arr.shape != (4, 4) or np.isnan(arr).any():
            return None
        return self._adjust_frame(arr)

    def _adjust_frame(self, T: np.ndarray) -> np.ndarray:
        adj = np.array(
            [[0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        r_adj = self._xyzrpy_to_matrix(0.0, 0.0, 0.0, -math.pi, 0.0, -math.pi / 2.0)
        return adj @ T @ r_adj

    @staticmethod
    def _smooth_transform(raw: np.ndarray, prev: np.ndarray | None, alpha: float) -> np.ndarray:
        if prev is None:
            return raw.copy()
        out = raw.copy()
        out[:3, 3] = alpha * raw[:3, 3] + (1.0 - alpha) * prev[:3, 3]
        rot_delta = pin.log3(prev[:3, :3].T @ raw[:3, :3])
        out[:3, :3] = prev[:3, :3] @ pin.exp3(alpha * rot_delta)
        return out

    def _log_input(self, now: float, buttons: dict[str, Any]) -> None:
        if now - self._last_input_log_t < self.config.log_input_interval_s:
            return
        self._last_input_log_t = now
        logger.info(
            "[VR_INPUT] B=%s A=%s trig=%.2f gripper=%s",
            bool(buttons.get(self.config.enable_button, False)),
            bool(buttons.get(self.config.reset_button, False)),
            self._extract_trigger(buttons, self.config.gripper_button),
            "open" if self._right.gripper_open else "closed",
        )

    def _log_action(self, now: float, action: RobotAction) -> None:
        if now - self._last_action_log_t < self.config.log_action_interval_s:
            return
        if self.config.log_only_on_enable and not bool(action.get("enabled", False)):
            return
        self._last_action_log_t = now
        target_keys = [
            "ee.target_x",
            "ee.target_y",
            "ee.target_z",
            "ee.target_rx",
            "ee.target_ry",
            "ee.target_rz",
        ]
        if all(key in action for key in target_keys):
            target_xyzrpy = np.array([float(action[key]) for key in target_keys], dtype=np.float64)
            target_T = self._xyzrpy_to_matrix(*target_xyzrpy.tolist())
            logger.info(
                "[VR_TRACE] enabled=%s reset=%s gripper=%.3f target_xyzrpy=%s target_T=%s",
                bool(action.get("enabled", False)),
                bool(action.get("reset", False)),
                float(action.get("gripper.pos", 0.0)),
                np.array2string(target_xyzrpy, precision=5, suppress_small=True),
                np.array2string(target_T, precision=5, suppress_small=True),
            )
            return

        delta_keys = [
            "ee.delta_x",
            "ee.delta_y",
            "ee.delta_z",
            "ee.delta_rx",
            "ee.delta_ry",
            "ee.delta_rz",
        ]
        delta = np.array([float(action.get(key, 0.0)) for key in delta_keys], dtype=np.float64)
        logger.info(
            "[VR_TRACE] enabled=%s reset=%s gripper=%.3f delta=%s keys=%s",
            bool(action.get("enabled", False)),
            bool(action.get("reset", False)),
            float(action.get("gripper.pos", 0.0)),
            np.array2string(delta, precision=5, suppress_small=True),
            sorted(action.keys()),
        )

    def _log_motion_diagnostics(
        self,
        *,
        state: ArmRuntime,
        enable_now: bool,
        reset_edge: bool,
        in_dead_zone: bool,
        raw_step_pos: float,
        raw_step_rot: float,
        smooth_step_pos: float,
        smooth_step_rot: float,
        raw_dp: np.ndarray,
        raw_dr: np.ndarray,
        delta_pos: np.ndarray,
        delta_rot: np.ndarray,
        target_xyzrpy: np.ndarray | None,
    ) -> None:
        now = time.monotonic()
        if now - state.last_motion_diag_log_t < self.config.log_diagnostics_interval_s:
            return
        state.last_motion_diag_log_t = now
        logger.info(
            "[VR_DIAG] enabled=%s reset_edge=%s dead_zone=%s raw_step_pos=%.6f raw_step_rot=%.6f "
            "smooth_step_pos=%.6f smooth_step_rot=%.6f base_dp=%s base_dr=%s delta_pos=%s delta_rot=%s target_xyzrpy=%s",
            enable_now,
            reset_edge,
            in_dead_zone,
            raw_step_pos,
            raw_step_rot,
            smooth_step_pos,
            smooth_step_rot,
            np.array2string(raw_dp, precision=6, suppress_small=True),
            np.array2string(raw_dr, precision=6, suppress_small=True),
            np.array2string(delta_pos, precision=6, suppress_small=True),
            np.array2string(delta_rot, precision=6, suppress_small=True),
            "None" if target_xyzrpy is None else np.array2string(target_xyzrpy, precision=6, suppress_small=True),
        )

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

    @classmethod
    def _pose_step(cls, prev: np.ndarray | None, cur: np.ndarray) -> tuple[float, float]:
        if prev is None:
            return 0.0, 0.0
        pos = float(np.linalg.norm(cur[:3, 3] - prev[:3, 3]))
        rot = float(np.linalg.norm(cls._rotation_vector_from_matrix(prev[:3, :3].T @ cur[:3, :3])))
        return pos, rot

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
    def _matrix_to_xyzrpy(T: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(T[0, 3]),
                float(T[1, 3]),
                float(T[2, 3]),
                float(math.atan2(T[2, 1], T[2, 2])),
                float(math.asin(np.clip(-T[2, 0], -1.0, 1.0))),
                float(math.atan2(T[1, 0], T[0, 0])),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_vector_from_matrix(R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3) or not np.isfinite(R).all():
            logger.warning("[VR_HEALTH] invalid rotation matrix for log3; returning zero rotation")
            return np.zeros(3, dtype=np.float64)
        try:
            return np.asarray(pin.log3(R), dtype=np.float64)
        except Exception as exc:  # pragma: no cover - defensive against malformed rotations
            logger.warning("[VR_HEALTH] failed to compute rotation log; returning zero rotation: %s", exc)
            return np.zeros(3, dtype=np.float64)
