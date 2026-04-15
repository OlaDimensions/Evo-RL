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

import copy
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np
import pinocchio as pin

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import RobotAction
from lerobot.utils.constants import ACTION
from lerobot.utils.piper_sdk import PIPER_JOINT_ACTION_KEYS

from ...processor.core import TransitionKey
from ...processor.pipeline import ProcessorStepRegistry, RobotActionProcessorStep
from .ik_types import IKBackend, IKSolveResult

logger = logging.getLogger(__name__)


@dataclass
class EETargetState:
    """Internal end-effector state tracked by the processor."""

    target_T: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    last_q: np.ndarray | None = None
    armed: bool = False
    reset_plan: list[np.ndarray] = field(default_factory=list)
    ik_inflight: bool = False
    async_action_ready: RobotAction | None = None
    solve_generation: int = 0
    last_seed_source: str = "none"
    last_seed_log_t: float = 0.0
    last_target_log_t: float = 0.0


@ProcessorStepRegistry.register("ee_to_joint_ik")
@dataclass
class EEToJointIKProcessorStep(RobotActionProcessorStep):
    """Convert Quest3 end-effector actions into joint actions via IK."""

    ik_backend: IKBackend | None = None
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    dead_zone_pos: float = 0.0
    dead_zone_rot: float = 0.0
    arm_init_T: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64), repr=False)
    gripper_key: str = "gripper.pos"
    enable_key: str = "enabled"
    reset_key: str = "reset"
    input_prefix: str = ""
    output_prefix: str = ""
    absolute_joint_command_key: str = "__absolute_joint_targets__"
    async_solve: bool = True
    reset_interp_steps: int = 25
    _state: EETargetState = field(default_factory=EETargetState, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        if self.ik_backend is None:
            raise RuntimeError("EEToJointIKProcessorStep requires an ik_backend instance.")
        gripper = float(action.get(self._in(self.gripper_key), 0.0))

        # Preserve quest3VR_ws semantics: reset motion keeps running for several ticks.
        if self._state.reset_plan:
            return self._emit_reset_step(action)

        enabled = bool(action.get(self._in(self.enable_key), True))
        reset = bool(action.get(self._in(self.reset_key), False))
        if reset:
            with self._lock:
                self._state.solve_generation += 1
                self._state.async_action_ready = None
            self._build_reset_plan(action)
            return self._emit_reset_step(action)

        if not enabled:
            with self._lock:
                self._state.solve_generation += 1
                self._state.async_action_ready = None
            return self._idle_action(gripper)

        # Deliver async result first if available.
        async_ready = self._pop_async_ready()
        if async_ready is not None:
            return self._with_current_gripper(async_ready, gripper)

        abs_target_T = self._action_to_absolute_target(action)
        if abs_target_T is None:
            delta_pos, delta_rot = self._action_to_delta_components(action)
            if self._is_in_dead_zone(delta_pos, delta_rot):
                return self._idle_action(gripper)
        else:
            delta_pos = np.zeros(3, dtype=np.float64)
            delta_rot = np.zeros(3, dtype=np.float64)

        with self._lock:
            if not self._state.armed:
                self._state.target_T = self.arm_init_T.copy()
                self._state.armed = True

            if abs_target_T is not None:
                self._state.target_T = abs_target_T.copy()
            else:
                self._state.target_T = self._state.target_T.copy()
                self._state.target_T[:3, 3] = self._state.target_T[:3, 3] + delta_pos
                self._state.target_T[:3, :3] = self._state.target_T[:3, :3] @ pin.exp3(delta_rot)

            q_seed, seed_source = self._pick_seed()
            self._log_seed_source(seed_source)
            target_T = self._state.target_T.copy()
            self._log_target(target_T, q_seed, seed_source, gripper, abs_target_T is not None)

            if self.async_solve:
                if not self._state.ik_inflight:
                    self._state.solve_generation += 1
                    generation = self._state.solve_generation
                    self._state.ik_inflight = True
                    thread = threading.Thread(
                        target=self._solve_async_worker,
                        args=(
                            target_T,
                            q_seed,
                            gripper,
                            seed_source,
                            generation,
                        ),
                        daemon=True,
                    )
                    thread.start()
                return self._idle_action(gripper)

        # Synchronous fallback.
        result = self.ik_backend.solve(target_T, q_seed=q_seed)
        return self._result_to_action(result, gripper, seed_source)

    def _build_reset_plan(self, action: RobotAction) -> None:
        self._state.target_T = self.arm_init_T.copy()
        self._state.armed = True

        q_start, _ = self._pick_seed()
        result = self.ik_backend.solve(self.arm_init_T, q_seed=q_start)
        if result.success and result.q is not None:
            q_goal = np.asarray(result.q, dtype=np.float64)
        else:
            logger.warning("[IK] failed to solve arm_init_T during reset; falling back to home_q (reason=%s)", result.reason)
            q_goal = self.ik_backend.home_q()
        steps = max(1, int(self.reset_interp_steps))
        if np.linalg.norm(q_goal - q_start) < 1e-8:
            self._state.reset_plan = [q_goal.copy()]
            return
        # Exclude the start point so each emitted command advances motion.
        alphas = np.linspace(1.0 / steps, 1.0, steps)
        self._state.reset_plan = [(1.0 - a) * q_start + a * q_goal for a in alphas]

    def _emit_reset_step(self, action: RobotAction) -> RobotAction:
        if not self._state.reset_plan:
            return self._idle_action()
        q = np.asarray(self._state.reset_plan.pop(0), dtype=np.float64)
        self._state.last_q = q
        gripper = float(action.get(self._in(self.gripper_key), 0.0))
        return self._compose_joint_action(q, gripper)

    def _solve_async_worker(
        self,
        target_T: np.ndarray,
        q_seed: np.ndarray,
        gripper: float,
        seed_source: str,
        generation: int,
    ) -> None:
        try:
            result = self.ik_backend.solve(target_T, q_seed=q_seed)
            action = self._result_to_action(result, gripper, seed_source)
            with self._lock:
                if generation == self._state.solve_generation:
                    self._state.async_action_ready = action
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[IK] async solve failed: %s", exc)
        finally:
            with self._lock:
                self._state.ik_inflight = False

    def _pop_async_ready(self) -> RobotAction | None:
        with self._lock:
            ready = self._state.async_action_ready
            self._state.async_action_ready = None
            return ready

    def _pick_seed(self) -> tuple[np.ndarray, str]:
        q_seed_from_obs = self._seed_from_observation()
        if q_seed_from_obs is not None:
            return q_seed_from_obs
        if self._state.last_q is not None:
            return self._state.last_q.copy(), "last_q"
        return self.ik_backend.home_q(), "home_q"

    def _seed_from_observation(self) -> tuple[np.ndarray, str] | None:
        transition = getattr(self, "_current_transition", None)
        if not isinstance(transition, dict):
            return None
        observation = transition.get(TransitionKey.OBSERVATION, {})
        if not isinstance(observation, dict):
            return None
        vals: list[float] = []
        for key in PIPER_JOINT_ACTION_KEYS:
            obs_key = f"{self.output_prefix}{key}"
            value = observation.get(obs_key)
            if value is None:
                return None
            vals.append(float(value))
        q_seed = np.asarray(vals, dtype=np.float64)

        # Piper follower observations are in degrees. Normalize to radians for IK.
        max_abs = float(np.max(np.abs(q_seed))) if q_seed.size else 0.0
        if max_abs > math.pi * 1.2:
            q_seed = np.deg2rad(q_seed)
            source = "observation_deg"
        else:
            source = "observation_rad"

        max_abs_after_norm = float(np.max(np.abs(q_seed))) if q_seed.size else 0.0
        if not np.isfinite(q_seed).all() or max_abs_after_norm > math.pi * 1.2:
            logger.warning("[IK] invalid observation seed (max_abs=%.3f); fallback to non-observation seed", max_abs_after_norm)
            return None
        return q_seed, source

    def _action_to_delta_components(self, action: RobotAction) -> tuple[np.ndarray, np.ndarray]:
        dx = float(action.get(self._in("ee.delta_x"), 0.0)) * self.position_scale
        dy = float(action.get(self._in("ee.delta_y"), 0.0)) * self.position_scale
        dz = float(action.get(self._in("ee.delta_z"), 0.0)) * self.position_scale
        drx = float(action.get(self._in("ee.delta_rx"), 0.0)) * self.rotation_scale
        dry = float(action.get(self._in("ee.delta_ry"), 0.0)) * self.rotation_scale
        drz = float(action.get(self._in("ee.delta_rz"), 0.0)) * self.rotation_scale
        return np.array([dx, dy, dz], dtype=np.float64), np.array([drx, dry, drz], dtype=np.float64)

    def _action_to_absolute_target(self, action: RobotAction) -> np.ndarray | None:
        keys = [
            "ee.target_x",
            "ee.target_y",
            "ee.target_z",
            "ee.target_rx",
            "ee.target_ry",
            "ee.target_rz",
        ]
        values: list[float] = []
        for key in keys:
            value = action.get(self._in(key))
            if value is None:
                return None
            values.append(float(value))
        return self._xyzrpy_to_matrix(*values)

    def _is_in_dead_zone(self, delta_pos: np.ndarray, delta_rot: np.ndarray) -> bool:
        return np.linalg.norm(delta_pos) < self.dead_zone_pos and np.linalg.norm(delta_rot) < self.dead_zone_rot

    def _result_to_action(self, result: IKSolveResult, gripper: float, seed_source: str) -> RobotAction:
        if result.q is None or not result.success:
            logger.warning("[IK] solve failed; skipping command this tick (reason=%s)", result.reason)
            return self._idle_action(gripper)
        self._state.last_q = np.asarray(result.q, dtype=np.float64)
        q_deg = np.rad2deg(self._state.last_q[: len(PIPER_JOINT_ACTION_KEYS)])
        logger.info(
            "[IK_TRACE] solve_ms=%.2f collision_free=%s seed=%s q_rad=%s q_deg=%s gripper=%.3f",
            result.solve_ms,
            result.collision_free,
            seed_source,
            np.array2string(self._state.last_q, precision=5, suppress_small=True),
            np.array2string(q_deg, precision=3, suppress_small=True),
            gripper,
        )
        return self._compose_joint_action(self._state.last_q, gripper)

    def _compose_joint_action(self, q: np.ndarray, gripper: float) -> RobotAction:
        # Pinocchio/CasADi IK solutions are in radians, while Piper SDK joint
        # commands are interpreted in degrees (then converted to milli-units).
        q_deg = np.rad2deg(q[: len(PIPER_JOINT_ACTION_KEYS)])
        action: RobotAction = {
            f"{self.output_prefix}{key}": float(val)
            for key, val in zip(PIPER_JOINT_ACTION_KEYS, q_deg, strict=True)
        }
        action[f"{self.output_prefix}{self.gripper_key}"] = gripper
        action[f"{self.output_prefix}{self.absolute_joint_command_key}"] = True
        return action

    def _idle_action(self, gripper: float | None = None) -> RobotAction:
        if gripper is None:
            return {}
        return {
            f"{self.output_prefix}{self.gripper_key}": float(gripper),
            f"{self.output_prefix}{self.absolute_joint_command_key}": True,
        }

    def _with_current_gripper(self, action: RobotAction, gripper: float) -> RobotAction:
        action = copy.copy(action)
        action[f"{self.output_prefix}{self.gripper_key}"] = float(gripper)
        return action

    def _log_target(
        self,
        target_T: np.ndarray,
        q_seed: np.ndarray,
        seed_source: str,
        gripper: float,
        absolute_target: bool,
    ) -> None:
        now = time.monotonic()
        if now - self._state.last_target_log_t < 0.5:
            return
        self._state.last_target_log_t = now
        logger.info(
            "[IK_TRACE] absolute_target=%s gripper=%.3f seed=%s q_seed=%s target_T=%s",
            absolute_target,
            gripper,
            seed_source,
            np.array2string(q_seed, precision=5, suppress_small=True),
            np.array2string(target_T, precision=5, suppress_small=True),
        )

    def _log_seed_source(self, seed_source: str) -> None:
        now = time.monotonic()
        if (
            seed_source != self._state.last_seed_source
            or now - self._state.last_seed_log_t >= 2.0
        ):
            logger.info("[IK] seed_source=%s", seed_source)
            self._state.last_seed_source = seed_source
            self._state.last_seed_log_t = now

    def _in(self, key: str) -> str:
        return f"{self.input_prefix}{key}"

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        output = {k: dict(v) for k, v in features.items()}
        action_features = dict(output.get(ACTION, {}))

        for key in [
            self.enable_key,
            self.reset_key,
            "ee.target_x",
            "ee.target_y",
            "ee.target_z",
            "ee.target_rx",
            "ee.target_ry",
            "ee.target_rz",
            "ee.delta_x",
            "ee.delta_y",
            "ee.delta_z",
            "ee.delta_rx",
            "ee.delta_ry",
            "ee.delta_rz",
        ]:
            action_features.pop(self._in(key), None)

        for key in PIPER_JOINT_ACTION_KEYS:
            out_key = f"{self.output_prefix}{key}"
            action_features[out_key] = action_features.get(out_key, PolicyFeature(type="float32"))
        action_features[f"{self.output_prefix}{self.gripper_key}"] = action_features.get(
            f"{self.output_prefix}{self.gripper_key}", PolicyFeature(type="float32")
        )
        action_features[f"{self.output_prefix}{self.absolute_joint_command_key}"] = action_features.get(
            f"{self.output_prefix}{self.absolute_joint_command_key}", PolicyFeature(type="bool")
        )
        output[ACTION] = action_features
        return output

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


@ProcessorStepRegistry.register("dual_ee_to_joint_ik")
@dataclass
class DualArmEEToJointIKProcessorStep(RobotActionProcessorStep):
    """Apply two EEToJointIKProcessorStep instances and merge their outputs."""

    left_step: EEToJointIKProcessorStep
    right_step: EEToJointIKProcessorStep

    def action(self, action: RobotAction) -> RobotAction:
        transition = getattr(self, "_current_transition", None)
        if not isinstance(transition, dict):
            transition = {TransitionKey.OBSERVATION: {}, TransitionKey.ACTION: action}

        left_transition: dict[Any, Any] = copy.copy(transition)
        left_transition[TransitionKey.ACTION] = action.copy()
        right_transition: dict[Any, Any] = copy.copy(transition)
        right_transition[TransitionKey.ACTION] = action.copy()

        left_action = self.left_step(left_transition)[TransitionKey.ACTION]
        right_action = self.right_step(right_transition)[TransitionKey.ACTION]
        return {**left_action, **right_action}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        out = self.left_step.transform_features(features)
        out = self.right_step.transform_features(out)
        return out
