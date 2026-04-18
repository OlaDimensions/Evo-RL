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
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from .ik_types import IKBackend, IKSolveResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PiperIKConfig:
    """Configuration for Piper IK backend."""

    urdf_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "assets/piper_description/urdf/piper_no_gripper_description.urdf"
    )
    package_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "assets/piper_description")
    ee_link_name: str = "ee"
    locked_joint_names: tuple[str, ...] = ()
    ee_link_joint_name: str = "joint6"
    ee_offset_xyzrpy: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ik_max_iter: int = 50
    ik_tol: float = 1e-4
    pose_cost_scale: float = 20.0
    pos_weight: float = 1.0
    ori_weight: float = 0.2
    regularization_weight: float = 0.05
    enable_branch_stable_ik: bool = True
    branch_trust_region_rad: tuple[float, ...] = (
        math.radians(25.0),
        math.radians(25.0),
        math.radians(25.0),
        math.radians(35.0),
        math.radians(30.0),
        math.radians(35.0),
    )
    collision_pair_joint_range: tuple[int, int] = (4, 9)
    collision_pair_link_range: tuple[int, int] = (0, 3)
    jump_threshold_rad: float = math.radians(50.0)
    pose_error_mode: str = "log_only"
    max_position_error_m: float = 0.08
    max_orientation_error_rad: float = math.radians(60.0)
    pose_error_log_interval_s: float = 0.5

    def __post_init__(self) -> None:
        if self.pose_error_mode not in {"off", "log_only", "reject"}:
            raise ValueError("pose_error_mode must be one of: off, log_only, reject.")
        if self.max_position_error_m <= 0:
            raise ValueError("max_position_error_m must be > 0.")
        if self.max_orientation_error_rad <= 0:
            raise ValueError("max_orientation_error_rad must be > 0.")
        if self.pose_error_log_interval_s < 0:
            raise ValueError("pose_error_log_interval_s must be >= 0.")
        if not isinstance(self.enable_branch_stable_ik, bool):
            raise ValueError("enable_branch_stable_ik must be true or false.")
        if not self.branch_trust_region_rad or any(value <= 0 for value in self.branch_trust_region_rad):
            raise ValueError("branch_trust_region_rad values must be > 0.")


class PiperPinocchioIKBackend(IKBackend):
    """CasADi + IPOPT based IK backend for Piper.

    This backend intentionally mirrors the original quest3VR_ws IK strategy:
    - Pinocchio for kinematic model and collision checking
    - CasADi Opti for nonlinear pose matching
    - IPOPT solve_limited for low-latency teleoperation
    - warm-start from measured joints or previous solution
    """

    def __init__(self, config: PiperIKConfig):
        self.config = config
        self._lock = Lock()
        self._q_prev = np.zeros(0)
        self._last_pose_error_log_t = 0.0
        self._build_models()
        self._build_solver()

    def _build_models(self) -> None:
        urdf_path = self.config.urdf_path.resolve()
        package_dir = self.config.package_dir.resolve()
        patched_urdf = self._patch_package_uris(urdf_path, package_dir)
        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(patched_urdf),
            package_dirs=[str(package_dir)],
        )
        logger.info("[IK] using urdf_path=%s package_dir=%s patched_urdf=%s", urdf_path, package_dir, patched_urdf)
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=list(self.config.locked_joint_names),
            reference_configuration=np.zeros(self.robot.model.nq),
        )
        self._full_q_ref = np.zeros(self.robot.model.nq, dtype=np.float64)
        self._full_joint_slices = self._joint_slices_by_name(self.robot.model)
        self._reduced_joint_slices = self._joint_slices_by_name(self.reduced_robot.model)

        ee_base_T = self._xyzrpy_to_matrix(*self.config.ee_offset_xyzrpy)
        q = self._quaternion_from_matrix(ee_base_T)
        self.ee_link_joint_id = self.reduced_robot.model.getJointId(self.config.ee_link_joint_name)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                self.config.ee_link_name,
                self.ee_link_joint_id,
                pin.SE3(
                    pin.Quaternion(q[3], q[0], q[1], q[2]),
                    np.array([ee_base_T[0, 3], ee_base_T[1, 3], ee_base_T[2, 3]]),
                ),
                pin.FrameType.OP_FRAME,
            )
        )
        self.reduced_robot.data = self.reduced_robot.model.createData()

        self.geom_model = pin.buildGeomFromUrdf(
            self.robot.model,
            str(patched_urdf),
            pin.GeometryType.COLLISION,
        )
        geom_count = len(self.geom_model.geometryObjects)
        logger.info("[IK] geom_count=%d", geom_count)
        j0, j1 = self.config.collision_pair_joint_range
        l0, l1 = self.config.collision_pair_link_range
        j_start = max(0, min(j0, geom_count))
        j_end = max(0, min(j1, geom_count))
        l_start = max(0, min(l0, geom_count))
        l_end = max(0, min(l1, geom_count))
        for i in range(j_start, j_end):
            for j in range(l_start, l_end):
                if i < geom_count and j < geom_count and i != j:
                    self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        self.ee_id = self.reduced_robot.model.getFrameId(self.config.ee_link_name)
        self.error_fn = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [casadi.vertcat(cpin.log6(self.cdata.oMf[self.ee_id].inverse() * cpin.SE3(self.cTf)).vector)],
        )

    def _build_solver(self) -> None:
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        self.param_qwarm = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_q_lower = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_q_upper = self.opti.parameter(self.reduced_robot.model.nq)

        error_vec = self.error_fn(self.var_q, self.param_tf)
        pos_error = error_vec[:3]
        ori_error = error_vec[3:]

        pose_cost = self.config.pos_weight * casadi.sumsqr(pos_error) + self.config.ori_weight * casadi.sumsqr(ori_error)
        regularization_cost = self.config.regularization_weight * casadi.sumsqr(self.var_q - self.param_qwarm)
        total_cost = self.config.pose_cost_scale * pose_cost + regularization_cost
        self.opti.minimize(total_cost)
        self.opti.subject_to(
            self.opti.bounded(
                self.param_q_lower,
                self.var_q,
                self.param_q_upper,
            )
        )
        opts = {
            "ipopt": {
                "print_level": 0,
                "max_iter": self.config.ik_max_iter,
                "tol": self.config.ik_tol,
            },
            "print_time": False,
        }
        self.opti.solver("ipopt", opts)

    def solve(self, target_T: np.ndarray, q_seed: np.ndarray | None = None) -> IKSolveResult:
        with self._lock:
            warm = self._warm_start(q_seed)
            q_lower, q_upper = self._q_bounds_for_warm_start(warm)
            if self._q_prev.size == 0:
                self._q_prev = warm.copy()
            self.opti.set_initial(self.var_q, warm)
            self.opti.set_value(self.param_tf, target_T)
            self.opti.set_value(self.param_qwarm, warm)
            self.opti.set_value(self.param_q_lower, q_lower)
            self.opti.set_value(self.param_q_upper, q_upper)

            start_total = time.perf_counter()
            try:
                start_solve = time.perf_counter()
                self.opti.solve_limited()
                q = np.asarray(self.opti.value(self.var_q), dtype=np.float64)
                solve_ms = (time.perf_counter() - start_solve) * 1000.0
                start_collision = time.perf_counter()
                collision_free = not self._has_collision(q)
                collision_ms = (time.perf_counter() - start_collision) * 1000.0
                total_ms = (time.perf_counter() - start_total) * 1000.0
                stats = self.opti.stats()
                iter_count = stats.get("iter_count", -1)
                return_status = stats.get("return_status", "unknown")
                position_error_m = 0.0
                orientation_error_rad = 0.0
                pose_error_exceeded = False
                pose_error_check_reason = ""
                if self.config.pose_error_mode != "off" and self._is_valid_q(q):
                    try:
                        position_error_m, orientation_error_rad = self._compute_pose_error(q, target_T)
                        pose_error_exceeded = self._pose_error_exceeded(
                            position_error_m, orientation_error_rad
                        )
                        self._maybe_log_pose_error(
                            position_error_m, orientation_error_rad, pose_error_exceeded
                        )
                    except Exception as exc:  # pragma: no cover - defensive against backend FK failures
                        pose_error_check_reason = f"pose_error_check_failed:{exc}"
                        logger.warning("[IK] failed to compute pose error: %s", exc)

                rejection_reason = self._solution_rejection_reason(warm, q, collision_free)
                if not rejection_reason:
                    rejection_reason = self._pose_error_rejection_reason(
                        position_error_m,
                        orientation_error_rad,
                        pose_error_exceeded,
                        pose_error_check_reason,
                    )
                logger.info(
                    "[IK] solve_ms=%.2f collision_ms=%.2f total_ms=%.2f iter=%s status=%s "
                    "collision_free=%s rejected=%s pose_error_mode=%s pos_error_m=%.4f "
                    "ori_error_deg=%.2f pose_error_exceeded=%s",
                    solve_ms,
                    collision_ms,
                    total_ms,
                    iter_count,
                    return_status,
                    collision_free,
                    rejection_reason or "none",
                    self.config.pose_error_mode,
                    position_error_m,
                    math.degrees(orientation_error_rad),
                    pose_error_exceeded,
                )
                if rejection_reason:
                    logger.warning("[IK] rejecting Piper solution: %s", rejection_reason)
                    return IKSolveResult(
                        q=None,
                        success=False,
                        collision_free=collision_free,
                        solve_ms=solve_ms,
                        reason=rejection_reason,
                        position_error_m=position_error_m,
                        orientation_error_rad=orientation_error_rad,
                        pose_error_exceeded=pose_error_exceeded,
                    )
                self._q_prev = q.copy()
                return IKSolveResult(
                    q=q,
                    success=True,
                    collision_free=collision_free,
                    solve_ms=solve_ms,
                    position_error_m=position_error_m,
                    orientation_error_rad=orientation_error_rad,
                    pose_error_exceeded=pose_error_exceeded,
                )
            except Exception as exc:
                total_ms = (time.perf_counter() - start_total) * 1000.0
                logger.warning("[IK] solve failed after %.2f ms: %s", total_ms, exc)
                return IKSolveResult(
                    q=None, success=False, collision_free=False, solve_ms=total_ms, reason=str(exc)
                )

    def _compute_pose_error(self, q: np.ndarray, target_T: np.ndarray) -> tuple[float, float]:
        actual_T = self.fk(q)
        return self._pose_error(actual_T, target_T)

    def _pose_error_exceeded(self, position_error_m: float, orientation_error_rad: float) -> bool:
        return (
            position_error_m > self.config.max_position_error_m
            or orientation_error_rad > self.config.max_orientation_error_rad
        )

    def _pose_error_rejection_reason(
        self,
        position_error_m: float,
        orientation_error_rad: float,
        pose_error_exceeded: bool,
        pose_error_check_reason: str = "",
    ) -> str:
        if self.config.pose_error_mode != "reject":
            return ""
        if pose_error_check_reason:
            return pose_error_check_reason
        if not pose_error_exceeded:
            return ""
        return (
            f"pose_error_exceeded:pos={position_error_m:.4f}m,"
            f"ori={math.degrees(orientation_error_rad):.2f}deg"
        )

    def _maybe_log_pose_error(
        self,
        position_error_m: float,
        orientation_error_rad: float,
        pose_error_exceeded: bool,
    ) -> None:
        interval_s = self.config.pose_error_log_interval_s
        now = time.monotonic()
        if interval_s > 0 and now - self._last_pose_error_log_t < interval_s:
            return
        self._last_pose_error_log_t = now
        logger.info(
            "[IK_DIAG] pose_error mode=%s pos_error_m=%.4f ori_error_deg=%.2f exceeded=%s "
            "max_pos_m=%.4f max_ori_deg=%.2f",
            self.config.pose_error_mode,
            position_error_m,
            math.degrees(orientation_error_rad),
            pose_error_exceeded,
            self.config.max_position_error_m,
            math.degrees(self.config.max_orientation_error_rad),
        )

    @staticmethod
    def _pose_error(actual_T: np.ndarray, target_T: np.ndarray) -> tuple[float, float]:
        actual_T = np.asarray(actual_T, dtype=np.float64)
        target_T = np.asarray(target_T, dtype=np.float64)
        position_error_m = float(np.linalg.norm(actual_T[:3, 3] - target_T[:3, 3]))
        orientation_error_rad = float(np.linalg.norm(pin.log3(actual_T[:3, :3].T @ target_T[:3, :3])))
        return position_error_m, orientation_error_rad

    def fk(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        return self.reduced_robot.data.oMf[self.ee_id].homogeneous

    def home_q(self) -> np.ndarray:
        return np.zeros(self.reduced_robot.model.nq, dtype=np.float64)

    def set_previous_q(self, q: np.ndarray | None) -> None:
        with self._lock:
            if q is None:
                self._q_prev = np.zeros(0, dtype=np.float64)
                return
            q_arr = np.asarray(q, dtype=np.float64)[: self.reduced_robot.model.nq]
            if self._is_valid_q(q_arr):
                self._q_prev = q_arr.copy()
            else:
                logger.warning("[IK] ignoring invalid previous_q update")
                self._q_prev = np.zeros(0, dtype=np.float64)

    def clear_previous_q(self) -> None:
        self.set_previous_q(None)

    def _warm_start(self, q_seed: np.ndarray | None) -> np.ndarray:
        if q_seed is not None:
            q = np.asarray(q_seed, dtype=np.float64)[: self.reduced_robot.model.nq]
            if self._is_valid_q(q):
                return q
            logger.warning("[IK] invalid q_seed; falling back to previous/home seed")
        if self._is_valid_q(self._q_prev):
            return self._q_prev.copy()
        return self.home_q()

    def _q_bounds_for_warm_start(self, warm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(self.reduced_robot.model.lowerPositionLimit, dtype=np.float64).copy()
        upper = np.asarray(self.reduced_robot.model.upperPositionLimit, dtype=np.float64).copy()
        if not self.config.enable_branch_stable_ik:
            return lower, upper

        trust = self._branch_trust_region()
        warm = np.asarray(warm, dtype=np.float64)[: self.reduced_robot.model.nq]
        return np.maximum(lower, warm - trust), np.minimum(upper, warm + trust)

    def _branch_trust_region(self) -> np.ndarray:
        trust = np.asarray(self.config.branch_trust_region_rad, dtype=np.float64)
        nq = self.reduced_robot.model.nq
        if trust.size == 1:
            return np.full(nq, float(trust[0]), dtype=np.float64)
        if trust.size != nq:
            raise ValueError(
                f"branch_trust_region_rad must contain 1 or {nq} values for this IK model, got {trust.size}."
            )
        return trust.copy()

    def _is_valid_q(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.reduced_robot.model.nq,) or not np.isfinite(q).all():
            return False
        return bool(
            np.all(q >= self.reduced_robot.model.lowerPositionLimit)
            and np.all(q <= self.reduced_robot.model.upperPositionLimit)
        )

    def _solution_rejection_reason(self, warm: np.ndarray, q: np.ndarray, collision_free: bool) -> str:
        warm = np.asarray(warm, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if not self._is_valid_q(q):
            return "invalid_solution"
        if warm.shape != q.shape or not np.isfinite(warm).all():
            return "invalid_warm_start"
        jump_rad = float(np.max(np.abs(warm - q))) if q.size else 0.0
        if jump_rad > self.config.jump_threshold_rad:
            return f"joint_jump_exceeded:{math.degrees(jump_rad):.2f}deg"
        if not collision_free:
            return "self_collision"
        return ""

    def _has_collision(self, q: np.ndarray) -> bool:
        q_full = self._to_full_configuration(np.asarray(q, dtype=np.float64))
        pin.forwardKinematics(self.robot.model, self.robot.data, q_full)
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        return bool(pin.computeCollisions(self.geom_model, self.geometry_data, False))

    def _to_full_configuration(self, q_reduced: np.ndarray) -> np.ndarray:
        """Expand reduced configuration to full model configuration for collision FK."""
        if q_reduced.shape[0] == self.robot.model.nq:
            return q_reduced
        if q_reduced.shape[0] != self.reduced_robot.model.nq:
            raise ValueError(
                f"wrong reduced configuration size: expected {self.reduced_robot.model.nq}, got {q_reduced.shape[0]}"
            )

        q_full = self._full_q_ref.copy()
        for joint_name, (rq0, rq1) in self._reduced_joint_slices.items():
            full_slice = self._full_joint_slices.get(joint_name)
            if full_slice is None:
                continue
            fq0, fq1 = full_slice
            if (fq1 - fq0) != (rq1 - rq0):
                raise ValueError(
                    f"joint slice size mismatch for {joint_name}: full={fq1 - fq0}, reduced={rq1 - rq0}"
                )
            q_full[fq0:fq1] = q_reduced[rq0:rq1]
        return q_full

    @staticmethod
    def _joint_slices_by_name(model: pin.Model) -> dict[str, tuple[int, int]]:
        slices: dict[str, tuple[int, int]] = {}
        for joint_id in range(1, model.njoints):
            joint_name = model.names[joint_id]
            joint = model.joints[joint_id]
            if joint.nq <= 0:
                continue
            q0 = joint.idx_q
            q1 = q0 + joint.nq
            slices[joint_name] = (q0, q1)
        return slices

    @staticmethod
    def _patch_package_uris(urdf_path: Path, package_dir: Path) -> Path:
        text = urdf_path.read_text()
        replacements = {
            "package://piper_description/meshes/": f"file://{(package_dir / 'meshes').resolve()}/",
            "package://piper_description/urdf/": f"file://{(package_dir / 'urdf').resolve()}/",
            "package://meshes/": f"file://{(package_dir / 'meshes').resolve()}/",
            "package://urdf/": f"file://{(package_dir / 'urdf').resolve()}/",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        tmp = NamedTemporaryFile("w", suffix=".urdf", delete=False)
        tmp.write(text)
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    @staticmethod
    def _xyzrpy_to_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        T = np.eye(4)
        ca, sa = np.cos(yaw), np.sin(yaw)
        cb, sb = np.cos(pitch), np.sin(pitch)
        cc, sc = np.cos(roll), np.sin(roll)
        T[0] = [ca * cb, ca * sb * sc - sa * cc, sa * sc + ca * sb * cc, x]
        T[1] = [sa * cb, ca * cc + sa * sb * sc, sa * sb * cc - ca * sc, y]
        T[2] = [-sb, cb * sc, cb * cc, z]
        return T

    @staticmethod
    def _quaternion_from_matrix(T: np.ndarray) -> tuple[float, float, float, float]:
        q = pin.Quaternion(T[:3, :3])
        return (q.x, q.y, q.z, q.w)
