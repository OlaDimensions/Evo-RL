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

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
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
    last_command_q: np.ndarray | None = None
    initial_q: np.ndarray | None = None
    armed: bool = False
    reset_plan: list[np.ndarray] = field(default_factory=list)
    reset_active: bool = False
    reset_goal_q: np.ndarray | None = None
    reset_settle_count: int = 0
    reset_started_t: float = 0.0
    absolute_input_anchor_T: np.ndarray | None = None
    absolute_output_anchor_T: np.ndarray | None = None
    ik_inflight: bool = False
    async_action_ready: RobotAction | None = None
    solve_generation: int = 0
    last_seed_source: str = "none"
    last_seed_log_t: float = 0.0
    last_target_log_t: float = 0.0
    last_idle_log_t: float = 0.0
    last_async_log_t: float = 0.0
    last_async_event: str = ""


@dataclass
class TargetSolveResult:
    result: IKSolveResult
    accepted_target_T: np.ndarray | None = None
    target_source: str = "failed"
    alpha: float = 0.0


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
    reset_target: str = "home_q"
    reset_joint_target_degrees: tuple[float, ...] = ()
    reset_joint_tolerance_rad: float = math.radians(6.0)
    reset_settle_ticks: int = 5
    reset_timeout_s: float = 10.0
    joint_smooth_alpha: float = 1.0
    max_joint_step_rad: float = 0.0
    target_retry_segment_counts: tuple[int, ...] = (2, 4, 8)
    enable_branch_stable_ik: bool = True
    branch_target_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    commit_accepted_target_only: bool = True
    diagnostics_interval_s: float = 0.5
    _state: EETargetState = field(default_factory=EETargetState, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def reset(self) -> None:
        with self._lock:
            generation = self._state.solve_generation + 1
            self._state = EETargetState(
                target_T=self.arm_init_T.copy(),
                solve_generation=generation,
            )
        self._clear_backend_previous_q()

    def action(self, action: RobotAction) -> RobotAction:
        if self.ik_backend is None:
            raise RuntimeError("EEToJointIKProcessorStep requires an ik_backend instance.")
        gripper = float(action.get(self._in(self.gripper_key), 0.0))

        enabled = bool(action.get(self._in(self.enable_key), True))
        reset = bool(action.get(self._in(self.reset_key), False))
        if not reset and not self._state.reset_active:
            self._capture_initial_q_if_available()
        if reset:
            with self._lock:
                self._state.solve_generation += 1
                self._state.async_action_ready = None
                generation = self._state.solve_generation
            self._log_async_event("reset_cancel", generation=generation, seed_source="n/a")
            self._build_reset_plan(action)
            return self._continue_reset(action)

        if self._state.reset_active:
            return self._continue_reset(action)

        if not enabled:
            with self._lock:
                self._state.solve_generation += 1
                self._state.async_action_ready = None
                generation = self._state.solve_generation
            self._log_async_event("disabled_cancel", generation=generation, seed_source="n/a")
            return self._idle_action(gripper, reason="disabled")

        # Deliver async result first if available.
        async_ready = self._pop_async_ready()
        if async_ready is not None:
            self._log_async_event("deliver_ready", generation=self._state.solve_generation, seed_source="n/a")
            return self._with_current_gripper(async_ready, gripper)

        abs_target_T = self._action_to_absolute_target(action)
        if abs_target_T is None:
            delta_pos, delta_rot = self._action_to_delta_components(action)
            if self._is_in_dead_zone(delta_pos, delta_rot):
                return self._idle_action(gripper, reason="delta_dead_zone")
        else:
            delta_pos = np.zeros(3, dtype=np.float64)
            delta_rot = np.zeros(3, dtype=np.float64)

        with self._lock:
            if not self._state.armed:
                self._state.target_T = self.arm_init_T.copy()
                self._state.armed = True

            if abs_target_T is not None:
                target_T = self._map_absolute_target(abs_target_T)
            else:
                target_T = self._state.target_T.copy()
                target_T[:3, 3] = target_T[:3, 3] + delta_pos
                target_T[:3, :3] = target_T[:3, :3] @ pin.exp3(delta_rot)
            if not self.commit_accepted_target_only:
                self._state.target_T = target_T.copy()

            q_seed, seed_source = self._pick_seed()
            self._log_seed_source(seed_source)
            self._log_target(target_T, q_seed, seed_source, gripper, abs_target_T is not None)

            if self.async_solve:
                if not self._state.ik_inflight:
                    self._state.solve_generation += 1
                    generation = self._state.solve_generation
                    self._state.ik_inflight = True
                    self._log_async_event("start", generation=generation, seed_source=seed_source)
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
                    return self._idle_action(gripper, reason="async_started")
                self._log_async_event(
                    "already_inflight",
                    generation=self._state.solve_generation,
                    seed_source=seed_source,
                )
                return self._idle_action(gripper, reason="async_inflight")

        # Synchronous fallback.
        solve_result = self._solve_target_with_retries(target_T, q_seed=q_seed)
        self._commit_accepted_target(solve_result)
        return self._result_to_action(solve_result.result, gripper, seed_source, q_seed=q_seed)

    def _build_reset_plan(self, action: RobotAction) -> None:
        self._state.target_T = self.arm_init_T.copy()
        self._state.absolute_input_anchor_T = None
        self._state.absolute_output_anchor_T = None
        self._state.armed = True
        self._state.reset_active = True
        self._state.reset_settle_count = 0
        self._state.reset_started_t = time.monotonic()

        q_start, _ = self._pick_seed()
        q_goal, goal_source = self._resolve_reset_goal(q_start)
        self._state.reset_goal_q = q_goal.copy()
        logger.info(
            "[IK_DIAG] reset_plan target=arm_init_T goal_source=%s q_start_deg=%s q_goal_deg=%s steps=%d tolerance_deg=%.3f settle_ticks=%d timeout_s=%.3f",
            goal_source,
            np.array2string(np.rad2deg(q_start[: len(PIPER_JOINT_ACTION_KEYS)]), precision=3, suppress_small=True),
            np.array2string(np.rad2deg(q_goal[: len(PIPER_JOINT_ACTION_KEYS)]), precision=3, suppress_small=True),
            max(1, int(self.reset_interp_steps)),
            math.degrees(self.reset_joint_tolerance_rad),
            max(1, int(self.reset_settle_ticks)),
            self.reset_timeout_s,
        )
        steps = max(1, int(self.reset_interp_steps))
        if np.linalg.norm(q_goal - q_start) < 1e-8:
            self._state.reset_plan = [q_goal.copy()]
            return
        # Exclude the start point so each emitted command advances motion.
        alphas = np.linspace(1.0 / steps, 1.0, steps)
        self._state.reset_plan = [(1.0 - a) * q_start + a * q_goal for a in alphas]

    def _resolve_reset_goal(self, q_start: np.ndarray) -> tuple[np.ndarray, str]:
        if self.reset_target == "joint_degrees":
            return np.deg2rad(np.asarray(self.reset_joint_target_degrees, dtype=np.float64)), "joint_degrees"
        if self.reset_target == "initial_observation" and self._state.initial_q is not None:
            return self._state.initial_q.copy(), "initial_observation"
        if self.reset_target == "initial_observation":
            logger.warning("[IK] reset_target=initial_observation but no initial observation was captured; falling back to home_q")
            return self.ik_backend.home_q(), "home_q_fallback"
        if self.reset_target == "arm_init_ik":
            result = self.ik_backend.solve(self.arm_init_T, q_seed=q_start)
            if result.success and result.q is not None:
                return np.asarray(result.q, dtype=np.float64), "arm_init_ik"
            logger.warning("[IK] failed to solve arm_init_T during reset; falling back to home_q (reason=%s)", result.reason)
            return self.ik_backend.home_q(), "home_q_fallback"
        return self.ik_backend.home_q(), "home_q"

    def _continue_reset(self, action: RobotAction) -> RobotAction:
        if self._state.reset_plan:
            q = np.asarray(self._state.reset_plan.pop(0), dtype=np.float64)
            phase = "interpolate"
        elif self._state.reset_goal_q is not None:
            q = np.asarray(self._state.reset_goal_q, dtype=np.float64)
            phase = "hold_goal"
        else:
            self._finish_reset(reason="missing_goal")
            return self._idle_action(float(action.get(self._in(self.gripper_key), 0.0)), reason="reset_missing_goal")

        self._state.last_q = q
        self._state.last_command_q = q.copy()
        gripper = float(action.get(self._in(self.gripper_key), 0.0))
        self._update_reset_completion(phase)
        return self._compose_joint_action(q, gripper)

    def _update_reset_completion(self, phase: str) -> None:
        goal_q = self._state.reset_goal_q
        if goal_q is None:
            self._finish_reset(reason="missing_goal")
            return
        if phase != "hold_goal":
            self._state.reset_settle_count = 0
            self._log_reset_status(phase, error_rad=None, settled=False, obs_source="not_checked")
            return

        obs_q, obs_source = self._observation_q_rad()
        if obs_q is None:
            self._state.reset_settle_count = 0
            if self._reset_timed_out():
                self._finish_reset(reason="timeout_no_observation")
            else:
                self._log_reset_status(phase, error_rad=None, settled=False, obs_source="none")
            return

        error_rad = float(np.max(np.abs(obs_q - goal_q[: obs_q.size])))
        settled = error_rad <= self.reset_joint_tolerance_rad
        if settled:
            self._state.reset_settle_count += 1
        else:
            self._state.reset_settle_count = 0

        enough_settle_ticks = self._state.reset_settle_count >= max(1, int(self.reset_settle_ticks))
        if enough_settle_ticks:
            self._finish_reset(reason="settled", error_rad=error_rad, obs_source=obs_source)
            return
        if self._reset_timed_out():
            self._finish_reset(reason="timeout", error_rad=error_rad, obs_source=obs_source)
            return
        self._log_reset_status(phase, error_rad=error_rad, settled=settled, obs_source=obs_source)

    def _reset_timed_out(self) -> bool:
        return self.reset_timeout_s > 0 and time.monotonic() - self._state.reset_started_t >= self.reset_timeout_s

    def _finish_reset(self, reason: str, error_rad: float | None = None, obs_source: str = "none") -> None:
        self._state.reset_active = False
        self._state.reset_plan = []
        self._state.reset_settle_count = 0
        if self._state.reset_goal_q is not None:
            self._state.last_q = self._state.reset_goal_q.copy()
            self._state.last_command_q = self._state.reset_goal_q.copy()
            if reason == "settled":
                self._set_backend_previous_q(self._state.reset_goal_q)
            else:
                self._clear_backend_previous_q()
        logger.info(
            "[IK_DIAG] reset_done reason=%s error_deg=%s obs=%s",
            reason,
            "None" if error_rad is None else f"{math.degrees(error_rad):.3f}",
            obs_source,
        )

    def _set_backend_previous_q(self, q: np.ndarray) -> None:
        setter = getattr(self.ik_backend, "set_previous_q", None)
        if callable(setter):
            setter(q)

    def _clear_backend_previous_q(self) -> None:
        clearer = getattr(self.ik_backend, "clear_previous_q", None)
        if callable(clearer):
            clearer()

    def _log_reset_status(self, phase: str, error_rad: float | None, settled: bool, obs_source: str) -> None:
        now = time.monotonic()
        if self.diagnostics_interval_s > 0 and now - self._state.last_idle_log_t < self.diagnostics_interval_s:
            return
        self._state.last_idle_log_t = now
        logger.info(
            "[IK_DIAG] reset_active phase=%s remaining_steps=%d error_deg=%s settled=%s settle_count=%d obs=%s",
            phase,
            len(self._state.reset_plan),
            "None" if error_rad is None else f"{math.degrees(error_rad):.3f}",
            settled,
            self._state.reset_settle_count,
            obs_source,
        )

    def _solve_async_worker(
        self,
        target_T: np.ndarray,
        q_seed: np.ndarray,
        gripper: float,
        seed_source: str,
        generation: int,
    ) -> None:
        try:
            solve_result = self._solve_target_with_retries(target_T, q_seed=q_seed)
            action = self._result_to_action(solve_result.result, gripper, seed_source, q_seed=q_seed)
            with self._lock:
                if generation == self._state.solve_generation:
                    self._commit_accepted_target_locked(solve_result)
                    self._state.async_action_ready = action
                    self._log_async_event("worker_ready", generation=generation, seed_source=seed_source)
                else:
                    self._log_async_event(
                        "worker_stale",
                        generation=generation,
                        seed_source=seed_source,
                        active_generation=self._state.solve_generation,
                    )
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

    def _solve_target_with_retries(self, target_T: np.ndarray, q_seed: np.ndarray) -> TargetSolveResult:
        if self.enable_branch_stable_ik:
            return self._solve_target_with_adaptive_alphas(target_T, q_seed=q_seed)

        direct_result = self.ik_backend.solve(target_T, q_seed=q_seed)
        if direct_result.success and direct_result.q is not None:
            return TargetSolveResult(direct_result, accepted_target_T=target_T.copy(), target_source="full", alpha=1.0)

        segment_counts = tuple(sorted({int(count) for count in self.target_retry_segment_counts if int(count) >= 2}))
        if not segment_counts:
            return TargetSolveResult(direct_result)

        try:
            start_T = self.ik_backend.fk(q_seed)
        except Exception as exc:  # pragma: no cover - defensive against backend-specific FK failures
            logger.warning("[IK] segmented retry skipped because FK failed: %s", exc)
            return TargetSolveResult(direct_result)

        for segment_count in segment_counts:
            alpha = 1.0 / float(segment_count)
            segment_T = self._interpolate_pose(start_T, target_T, alpha)
            result = self.ik_backend.solve(segment_T, q_seed=q_seed)
            logger.info(
                "[IK_DIAG] segmented_retry segments=%d alpha=%.3f success=%s reason=%s",
                segment_count,
                alpha,
                result.success,
                result.reason,
            )
            if result.success and result.q is not None:
                return TargetSolveResult(
                    result,
                    accepted_target_T=segment_T.copy(),
                    target_source="segment",
                    alpha=alpha,
                )

        return TargetSolveResult(direct_result)

    def _solve_target_with_adaptive_alphas(self, target_T: np.ndarray, q_seed: np.ndarray) -> TargetSolveResult:
        alphas = self._normalized_branch_target_alphas()
        result = self.ik_backend.solve(target_T, q_seed=q_seed)
        logger.info(
            "[IK_DIAG] branch_target alpha=1.000 success=%s reason=%s",
            result.success,
            result.reason,
        )
        if result.success and result.q is not None:
            return TargetSolveResult(result, accepted_target_T=target_T.copy(), target_source="full", alpha=1.0)

        last_result: IKSolveResult = result
        segment_alphas = tuple(alpha for alpha in alphas if alpha < 1.0)
        if not segment_alphas:
            return TargetSolveResult(last_result)

        try:
            start_T = self.ik_backend.fk(q_seed)
        except Exception as exc:  # pragma: no cover - defensive against backend-specific FK failures
            logger.warning("[IK] branch-stable retry skipped because FK failed: %s", exc)
            return TargetSolveResult(last_result)

        for alpha in segment_alphas:
            candidate_T = self._interpolate_pose(start_T, target_T, alpha)
            result = self.ik_backend.solve(candidate_T, q_seed=q_seed)
            last_result = result
            logger.info(
                "[IK_DIAG] branch_target alpha=%.3f success=%s reason=%s",
                alpha,
                result.success,
                result.reason,
            )
            if result.success and result.q is not None:
                return TargetSolveResult(
                    result,
                    accepted_target_T=np.asarray(candidate_T, dtype=np.float64).copy(),
                    target_source="segment",
                    alpha=alpha,
                )

        return TargetSolveResult(last_result)

    def _normalized_branch_target_alphas(self) -> tuple[float, ...]:
        alphas = sorted({float(alpha) for alpha in self.branch_target_alphas if 0.0 < float(alpha) <= 1.0}, reverse=True)
        if 1.0 not in alphas:
            alphas.insert(0, 1.0)
        return tuple(alphas)

    def _commit_accepted_target(self, solve_result: TargetSolveResult) -> None:
        with self._lock:
            self._commit_accepted_target_locked(solve_result)

    def _commit_accepted_target_locked(self, solve_result: TargetSolveResult) -> None:
        if not self.commit_accepted_target_only:
            return
        if solve_result.accepted_target_T is None:
            logger.info(
                "[IK_DIAG] committed_target=unchanged source=%s reason=%s",
                solve_result.target_source,
                solve_result.result.reason,
            )
            return
        self._state.target_T = solve_result.accepted_target_T.copy()
        logger.info(
            "[IK_DIAG] committed_target=%s alpha=%.3f",
            solve_result.target_source,
            solve_result.alpha,
        )

    def _pick_seed(self) -> tuple[np.ndarray, str]:
        q_seed_from_obs = self._seed_from_observation()
        if q_seed_from_obs is not None:
            return q_seed_from_obs
        if self._state.last_q is not None:
            return self._state.last_q.copy(), "last_q"
        return self.ik_backend.home_q(), "home_q"

    def _capture_initial_q_if_available(self) -> None:
        if self._state.initial_q is not None:
            return
        q_initial, source = self._observation_q_rad()
        if q_initial is None:
            return
        self._state.initial_q = q_initial.copy()
        logger.info(
            "[IK_DIAG] captured_initial_q source=%s q_deg=%s",
            source,
            np.array2string(np.rad2deg(q_initial[: len(PIPER_JOINT_ACTION_KEYS)]), precision=3, suppress_small=True),
        )

    def _seed_from_observation(self) -> tuple[np.ndarray, str] | None:
        q_seed, source = self._observation_q_rad()
        if q_seed is None:
            return None
        return q_seed, source

    def _observation_q_rad(self) -> tuple[np.ndarray | None, str]:
        transition = getattr(self, "_current_transition", None)
        if not isinstance(transition, dict):
            return None, "none"
        observation = transition.get(TransitionKey.OBSERVATION, {})
        if not isinstance(observation, dict):
            return None, "none"
        vals: list[float] = []
        for key in PIPER_JOINT_ACTION_KEYS:
            obs_key = f"{self.output_prefix}{key}"
            value = observation.get(obs_key)
            if value is None:
                return None, "none"
            vals.append(float(value))
        # Piper follower observations use the same unit as joint actions: degrees.
        q_seed = np.deg2rad(np.asarray(vals, dtype=np.float64))
        source = "observation_deg"

        max_abs_after_norm = float(np.max(np.abs(q_seed))) if q_seed.size else 0.0
        if not np.isfinite(q_seed).all() or max_abs_after_norm > math.pi * 1.2:
            logger.warning("[IK] invalid observation seed (max_abs=%.3f); fallback to non-observation seed", max_abs_after_norm)
            return None, "invalid"
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
        target_rpy_keys = [
            "ee.target_x",
            "ee.target_y",
            "ee.target_z",
            "ee.target_rx",
            "ee.target_ry",
            "ee.target_rz",
        ]
        values: list[float] = []
        for key in target_rpy_keys:
            value = action.get(self._in(key))
            if value is None:
                break
            values.append(float(value))
        else:
            return self._xyzrpy_to_matrix(*values)

        target_quat_keys = [
            "ee.x",
            "ee.y",
            "ee.z",
            "ee.qx",
            "ee.qy",
            "ee.qz",
            "ee.qw",
        ]
        values = []
        for key in target_quat_keys:
            value = action.get(self._in(key))
            if value is None:
                return None
            values.append(float(value))
        return self._xyzquat_to_matrix(*values)

    def _map_absolute_target(self, abs_target_T: np.ndarray) -> np.ndarray:
        input_anchor_T = self._state.absolute_input_anchor_T
        output_anchor_T = self._state.absolute_output_anchor_T
        if input_anchor_T is None or output_anchor_T is None:
            return abs_target_T.copy()

        target_T = output_anchor_T.copy()
        target_T[:3, 3] = output_anchor_T[:3, 3] + (abs_target_T[:3, 3] - input_anchor_T[:3, 3])
        target_T[:3, :3] = output_anchor_T[:3, :3] @ input_anchor_T[:3, :3].T @ abs_target_T[:3, :3]
        return target_T

    def _is_in_dead_zone(self, delta_pos: np.ndarray, delta_rot: np.ndarray) -> bool:
        return np.linalg.norm(delta_pos) < self.dead_zone_pos and np.linalg.norm(delta_rot) < self.dead_zone_rot

    @staticmethod
    def _interpolate_pose(start_T: np.ndarray, target_T: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        out = np.asarray(start_T, dtype=np.float64).copy()
        target_T = np.asarray(target_T, dtype=np.float64)
        out[:3, 3] = out[:3, 3] + alpha * (target_T[:3, 3] - out[:3, 3])
        rot_delta = pin.log3(out[:3, :3].T @ target_T[:3, :3])
        out[:3, :3] = out[:3, :3] @ pin.exp3(alpha * rot_delta)
        return out

    def _result_to_action(
        self,
        result: IKSolveResult,
        gripper: float,
        seed_source: str,
        *,
        q_seed: np.ndarray | None = None,
    ) -> RobotAction:
        if result.q is None or not result.success:
            logger.warning("[IK] solve failed; skipping command this tick (reason=%s)", result.reason)
            return self._idle_action(gripper)
        raw_q = np.asarray(result.q, dtype=np.float64)
        prev_command_q = self._state.last_command_q
        command_q, limit_baseline = self._smooth_joint_solution(raw_q, seed_q=q_seed)
        self._state.last_q = command_q.copy()
        self._state.last_command_q = command_q.copy()
        raw_q_deg = np.rad2deg(raw_q[: len(PIPER_JOINT_ACTION_KEYS)])
        command_q_deg = np.rad2deg(command_q[: len(PIPER_JOINT_ACTION_KEYS)])
        if limit_baseline is not None and limit_baseline.shape == command_q.shape:
            max_step_rad = float(np.max(np.abs(command_q - limit_baseline)))
            limit_baseline_name = "last_command_q" if prev_command_q is not None else "seed_q"
        else:
            max_step_rad = 0.0
            limit_baseline_name = "none"
        logger.info(
            "[IK_TRACE] solve_ms=%.2f collision_free=%s seed=%s raw_q_deg=%s command_q_deg=%s "
            "joint_smooth_alpha=%.3f max_joint_step_deg=%.3f limit_baseline=%s gripper=%.3f",
            result.solve_ms,
            result.collision_free,
            seed_source,
            np.array2string(raw_q_deg, precision=3, suppress_small=True),
            np.array2string(command_q_deg, precision=3, suppress_small=True),
            self.joint_smooth_alpha,
            math.degrees(max_step_rad),
            limit_baseline_name,
            gripper,
        )
        return self._compose_joint_action(command_q, gripper)

    def _smooth_joint_solution(self, raw_q: np.ndarray, seed_q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        raw_q = np.asarray(raw_q, dtype=np.float64)
        prev_q = self._state.last_command_q
        if prev_q is None or prev_q.shape != raw_q.shape:
            if seed_q is None:
                logger.warning("[IK] no previous command or seed available; first IK solution is not joint-step limited")
                return raw_q.copy(), None
            prev_q = np.asarray(seed_q, dtype=np.float64)
            if prev_q.shape != raw_q.shape or not np.isfinite(prev_q).all():
                logger.warning("[IK] invalid seed for first IK solution; first IK solution is not joint-step limited")
                return raw_q.copy(), None

        alpha = float(np.clip(self.joint_smooth_alpha, 0.0, 1.0))
        if alpha < 1.0:
            command_q = (1.0 - alpha) * prev_q + alpha * raw_q
        else:
            command_q = raw_q.copy()

        max_step = float(self.max_joint_step_rad)
        if max_step > 0.0:
            command_q = prev_q + np.clip(command_q - prev_q, -max_step, max_step)
        return command_q, prev_q.copy()

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

    def _idle_action(self, gripper: float | None = None, *, reason: str = "idle") -> RobotAction:
        if gripper is None:
            return {}
        self._log_idle_action(reason, gripper)
        return {f"{self.output_prefix}{self.gripper_key}": float(gripper)}

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

    def _log_idle_action(self, reason: str, gripper: float) -> None:
        now = time.monotonic()
        if self.diagnostics_interval_s > 0 and now - self._state.last_idle_log_t < self.diagnostics_interval_s:
            return
        self._state.last_idle_log_t = now
        logger.info(
            "[IK_DIAG] idle reason=%s generation=%d inflight=%s ready=%s gripper=%.3f",
            reason,
            self._state.solve_generation,
            self._state.ik_inflight,
            self._state.async_action_ready is not None,
            gripper,
        )

    def _log_async_event(
        self,
        event: str,
        *,
        generation: int,
        seed_source: str,
        active_generation: int | None = None,
    ) -> None:
        now = time.monotonic()
        if (
            event == self._state.last_async_event
            and self.diagnostics_interval_s > 0
            and now - self._state.last_async_log_t < self.diagnostics_interval_s
        ):
            return
        self._state.last_async_log_t = now
        self._state.last_async_event = event
        logger.info(
            "[IK_DIAG] async event=%s generation=%d active_generation=%s inflight=%s ready=%s seed=%s",
            event,
            generation,
            "None" if active_generation is None else active_generation,
            self._state.ik_inflight,
            self._state.async_action_ready is not None,
            seed_source,
        )

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
            action_features[out_key] = action_features.get(out_key, PolicyFeature(type=FeatureType.ACTION, shape=(1,)))
        action_features[f"{self.output_prefix}{self.gripper_key}"] = action_features.get(
            f"{self.output_prefix}{self.gripper_key}", PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        )
        action_features.pop(f"{self.output_prefix}{self.absolute_joint_command_key}", None)
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

    @staticmethod
    def _xyzquat_to_matrix(
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> np.ndarray:
        quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Invalid EE quaternion target: expected finite non-zero [qx, qy, qz, qw].")
        qx, qy, qz, qw = (quat / norm).tolist()

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [x, y, z]
        T[:3, :3] = np.array(
            [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                ],
                [
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                ],
                [
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float64,
        )
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

    def reset(self) -> None:
        self.left_step.reset()
        self.right_step.reset()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        out = self.left_step.transform_features(features)
        out = self.right_step.transform_features(out)
        return out
