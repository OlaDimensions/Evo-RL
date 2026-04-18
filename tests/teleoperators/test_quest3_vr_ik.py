import ast
import math
import numpy as np
import pytest
import time
from pathlib import Path
from types import SimpleNamespace
from threading import Lock

from lerobot.processor.converters import create_transition
from lerobot.scripts.recording_loop import _complete_action_values_for_dataset
from lerobot.teleoperators.bi_quest3_vr.bi_quest3_vr import BiQuest3VRTeleop
from lerobot.teleoperators.bi_quest3_vr.config_bi_quest3_vr import BiQuest3VRTeleopConfig
from lerobot.teleoperators.quest3_vr.config_quest3_vr import Quest3VRTeleopConfig
from lerobot.teleoperators.quest3_vr.ee_to_joint_ik import DualArmEEToJointIKProcessorStep, EEToJointIKProcessorStep
from lerobot.teleoperators.quest3_vr.quest3_vr import Quest3VRTeleop
from lerobot.utils.piper_sdk import PIPER_JOINT_ACTION_KEYS


QUEST3VR_WS_ROOT = Path("/home/ola/code/quest3VR_ws/src/oculus_reader/oculus_reader")


def test_complete_action_values_for_dataset_fills_partial_idle_action_without_mutating_command():
    features = {
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["joint_1.pos", "joint_2.pos", "gripper.pos"],
        }
    }
    command_action = {"gripper.pos": 0.08}
    observation = {"joint_1.pos": 10.0}
    previous = {"joint_1.pos": 9.0, "joint_2.pos": 20.0, "gripper.pos": 0.0}

    completed = _complete_action_values_for_dataset(features, command_action, observation, previous)

    assert command_action == {"gripper.pos": 0.08}
    assert completed == {"joint_1.pos": 10.0, "joint_2.pos": 20.0, "gripper.pos": 0.08}


def _piper_backend_classes_or_skip():
    try:
        from lerobot.teleoperators.quest3_vr.piper_pinocchio import (
            PiperIKConfig,
            PiperPinocchioIKBackend,
        )
    except ImportError as exc:
        pytest.skip(f"Piper Pinocchio IK backend dependencies are unavailable: {exc}")
    return PiperIKConfig, PiperPinocchioIKBackend

def _load_ws_functions(source_path: Path, names: set[str], extra_globals: dict | None = None) -> dict:
    if not source_path.exists():
        pytest.skip(f"quest3VR_ws source file not found: {source_path}")
    tree = ast.parse(source_path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    found = {node.name for node in nodes}
    missing = names - found
    if missing:
        pytest.skip(f"quest3VR_ws source file {source_path} is missing functions: {sorted(missing)}")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    if extra_globals:
        namespace.update(extra_globals)
    exec(compile(module, str(source_path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _raw_quest_T(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
    return Quest3VRTeleop._xyzrpy_to_matrix(x, y, z, roll, pitch, yaw)


def _action_target_T(action):
    keys = ["ee.target_x", "ee.target_y", "ee.target_z", "ee.target_rx", "ee.target_ry", "ee.target_rz"]
    assert all(key in action for key in keys), f"action missing target keys: {action}"
    return Quest3VRTeleop._xyzrpy_to_matrix(*[float(action[key]) for key in keys])


@pytest.mark.slow
def test_piper_pinocchio_ik_solve_returns_valid_solution():
    PiperIKConfig, PiperPinocchioIKBackend = _piper_backend_classes_or_skip()

    backend = PiperPinocchioIKBackend(PiperIKConfig())
    q_seed = backend.home_q()
    target_T = backend.fk(q_seed)

    result = backend.solve(target_T, q_seed=q_seed)

    assert result.success is True
    assert result.q is not None
    assert result.collision_free is True
    assert result.q.shape == (backend.reduced_robot.model.nq,)
    assert np.all(np.isfinite(result.q))

    fk_T = backend.fk(result.q)
    pos_err = np.linalg.norm(fk_T[:3, 3] - target_T[:3, 3])
    rot_err = np.linalg.norm(fk_T[:3, :3] - target_T[:3, :3])

    assert pos_err < 1e-2
    assert rot_err < 5e-2


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.previous_q = None

    def home_q(self):
        return np.zeros(6, dtype=np.float64)

    def set_previous_q(self, q):
        self.previous_q = None if q is None else np.asarray(q, dtype=np.float64).copy()

    def clear_previous_q(self):
        self.previous_q = None

    def solve(self, target_T, q_seed=None):
        self.calls.append((target_T.copy(), None if q_seed is None else q_seed.copy()))
        return type(
            "Result",
            (),
            {
                "q": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64),
                "success": True,
                "collision_free": True,
                "solve_ms": 12.34,
                "reason": None,
            },
        )()


class SegmentedRetryBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.fk_calls = []

    def fk(self, q):
        self.fk_calls.append(np.asarray(q, dtype=np.float64).copy())
        return np.eye(4, dtype=np.float64)

    def solve(self, target_T, q_seed=None):
        self.calls.append((target_T.copy(), None if q_seed is None else q_seed.copy()))
        success = float(target_T[0, 3]) <= 0.51
        return type(
            "Result",
            (),
            {
                "q": np.full(6, 0.2, dtype=np.float64) if success else None,
                "success": success,
                "collision_free": success,
                "solve_ms": 1.0,
                "reason": "" if success else "target_too_far",
            },
        )()


def _piper_backend_without_solver(nq=6):
    _, PiperPinocchioIKBackend = _piper_backend_classes_or_skip()

    backend = PiperPinocchioIKBackend.__new__(PiperPinocchioIKBackend)
    backend.config = SimpleNamespace(
        jump_threshold_rad=math.radians(30.0),
        enable_branch_stable_ik=True,
        branch_trust_region_rad=(math.radians(10.0),) * nq,
        pose_error_mode="log_only",
        max_position_error_m=0.08,
        max_orientation_error_rad=math.radians(60.0),
        pose_error_log_interval_s=0.5,
    )
    backend._lock = Lock()
    backend._q_prev = np.zeros(0, dtype=np.float64)
    backend.reduced_robot = SimpleNamespace(
        model=SimpleNamespace(
            nq=nq,
            lowerPositionLimit=-np.ones(nq, dtype=np.float64),
            upperPositionLimit=np.ones(nq, dtype=np.float64),
        )
    )
    return backend


def test_piper_backend_computes_pose_error_components():
    _, PiperPinocchioIKBackend = _piper_backend_classes_or_skip()

    actual_T = np.eye(4, dtype=np.float64)
    target_T = np.eye(4, dtype=np.float64)
    target_T[0, 3] = 0.03
    target_T[:3, :3] = actual_T[:3, :3] @ np.array(
        [
            [math.cos(math.radians(20.0)), -math.sin(math.radians(20.0)), 0.0],
            [math.sin(math.radians(20.0)), math.cos(math.radians(20.0)), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    pos_error, ori_error = PiperPinocchioIKBackend._pose_error(actual_T, target_T)

    assert pos_error == pytest.approx(0.03, abs=1e-12)
    assert math.degrees(ori_error) == pytest.approx(20.0, abs=1e-9)


def test_piper_backend_pose_error_log_only_does_not_reject():
    backend = _piper_backend_without_solver()
    backend.config.pose_error_mode = "log_only"

    reason = backend._pose_error_rejection_reason(0.20, math.radians(90.0), pose_error_exceeded=True)

    assert reason == ""


def test_piper_backend_pose_error_reject_mode_rejects_exceeded_error():
    backend = _piper_backend_without_solver()
    backend.config.pose_error_mode = "reject"

    reason = backend._pose_error_rejection_reason(0.20, math.radians(90.0), pose_error_exceeded=True)

    assert reason.startswith("pose_error_exceeded:pos=0.2000m,ori=90.00deg")


def test_piper_backend_pose_error_reject_mode_preserves_existing_reject_priority():
    backend = _piper_backend_without_solver()
    backend.config.pose_error_mode = "reject"
    warm = np.zeros(6, dtype=np.float64)
    q = np.zeros(6, dtype=np.float64)
    q[2] = math.radians(45.0)

    existing_reason = backend._solution_rejection_reason(warm, q, collision_free=True)
    pose_reason = backend._pose_error_rejection_reason(0.20, math.radians(90.0), pose_error_exceeded=True)
    final_reason = existing_reason or pose_reason

    assert final_reason.startswith("joint_jump_exceeded:")


def test_quest3_vr_config_validates_pose_error_mode():
    with pytest.raises(ValueError, match="ik_pose_error_mode"):
        Quest3VRTeleopConfig(ik_pose_error_mode="invalid")

def test_piper_backend_warm_start_falls_back_to_valid_previous_q_when_seed_missing():
    backend = _piper_backend_without_solver()
    previous_q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float64)

    backend.set_previous_q(previous_q)

    np.testing.assert_allclose(backend._warm_start(None), previous_q, atol=1e-12)


def test_piper_backend_warm_start_rejects_invalid_previous_q():
    backend = _piper_backend_without_solver()
    backend._q_prev = np.array([0.1, np.nan, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)

    np.testing.assert_allclose(backend._warm_start(None), np.zeros(6, dtype=np.float64), atol=1e-12)


def test_piper_backend_rejects_large_joint_jump_solution():
    backend = _piper_backend_without_solver()
    warm = np.zeros(6, dtype=np.float64)
    q = np.zeros(6, dtype=np.float64)
    q[2] = math.radians(45.0)

    reason = backend._solution_rejection_reason(warm, q, collision_free=True)

    assert reason.startswith("joint_jump_exceeded:")


def test_piper_backend_rejects_self_collision_solution():
    backend = _piper_backend_without_solver()
    warm = np.zeros(6, dtype=np.float64)
    q = np.zeros(6, dtype=np.float64)

    reason = backend._solution_rejection_reason(warm, q, collision_free=False)

    assert reason == "self_collision"


def test_piper_backend_accepts_valid_collision_free_solution():
    backend = _piper_backend_without_solver()
    warm = np.zeros(6, dtype=np.float64)
    q = np.full(6, math.radians(3.0), dtype=np.float64)

    reason = backend._solution_rejection_reason(warm, q, collision_free=True)

    assert reason == ""


class SlowFakeBackend(FakeBackend):
    def solve(self, target_T, q_seed=None):
        time.sleep(0.02)
        return super().solve(target_T, q_seed=q_seed)


class AdaptiveAlphaBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.fk_calls = []

    def fk(self, q):
        self.fk_calls.append(np.asarray(q, dtype=np.float64).copy())
        return np.eye(4, dtype=np.float64)

    def solve(self, target_T, q_seed=None):
        self.calls.append((target_T.copy(), None if q_seed is None else q_seed.copy()))
        success = float(target_T[0, 3]) <= 0.51
        return type(
            "Result",
            (),
            {
                "q": np.full(6, 0.2, dtype=np.float64) if success else None,
                "success": success,
                "collision_free": success,
                "solve_ms": 1.0,
                "reason": "" if success else "target_too_far",
            },
        )()


class AlwaysFailBackend(AdaptiveAlphaBackend):
    def solve(self, target_T, q_seed=None):
        self.calls.append((target_T.copy(), None if q_seed is None else q_seed.copy()))
        return type(
            "Result",
            (),
            {
                "q": None,
                "success": False,
                "collision_free": False,
                "solve_ms": 1.0,
                "reason": "always_fail",
            },
        )()


class _FakeReader:
    def __init__(self, samples):
        self.samples = list(samples)
        self.idx = 0

    def get_transformations_and_buttons(self):
        if self.idx >= len(self.samples):
            return self.samples[-1]
        out = self.samples[self.idx]
        self.idx += 1
        return out


def _T(x=0.0, y=0.0, z=0.0):
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


@pytest.mark.parametrize("angle_deg", [179.99, 180.0, 180.01])
def test_quest3_vr_rotation_vector_preserves_pi_rotation(angle_deg):
    R = _raw_quest_T(roll=math.radians(angle_deg))[:3, :3]

    rot_vec = Quest3VRTeleop._rotation_vector_from_matrix(R)

    assert math.degrees(float(np.linalg.norm(rot_vec))) == pytest.approx(180.0, abs=0.02)


def test_quest3_vr_rotation_vector_rejects_invalid_matrix():
    rot_vec = Quest3VRTeleop._rotation_vector_from_matrix(np.full((3, 3), np.nan, dtype=np.float64))

    np.testing.assert_allclose(rot_vec, np.zeros(3, dtype=np.float64), atol=0.0)


@pytest.mark.parametrize(
    "raw_T",
    [
        _raw_quest_T(),
        _raw_quest_T(0.10, 0.0, 0.0),
        _raw_quest_T(0.0, 0.10, 0.0),
        _raw_quest_T(0.0, 0.0, 0.10),
        _raw_quest_T(0.03, -0.04, 0.05, 0.10, -0.20, 0.30),
        _raw_quest_T(-0.02, 0.06, -0.04, -0.20, 0.15, -0.35),
    ],
)
def test_quest3_vr_adjust_frame_matches_quest3vr_ws(raw_T):
    ws = _load_ws_functions(QUEST3VR_WS_ROOT / "pub_pose_node.py", {"_xyzrpy_to_mat", "_adjust_frame"})
    teleop = Quest3VRTeleop(Quest3VRTeleopConfig())

    ws_adjusted = ws["_adjust_frame"](raw_T)
    evo_adjusted = teleop._adjust_frame(raw_T)

    np.testing.assert_allclose(evo_adjusted, ws_adjusted, atol=1e-12)


@pytest.mark.parametrize(
    "raw_current_T",
    [
        _raw_quest_T(0.10, 0.0, 0.0),
        _raw_quest_T(0.0, 0.10, 0.0),
        _raw_quest_T(0.0, 0.0, 0.10),
        _raw_quest_T(0.04, -0.02, 0.08, 0.0, 0.0, math.radians(12.0)),
        _raw_quest_T(-0.05, 0.03, -0.02, math.radians(-8.0), math.radians(6.0), 0.0),
    ],
)
def test_quest3_vr_single_arm_target_T_uses_matrix_delta_without_y_deadband(raw_current_T):
    pin = pytest.importorskip("pinocchio")

    cfg = Quest3VRTeleopConfig(smooth_alpha=1.0)
    teleop = Quest3VRTeleop(cfg)
    state = teleop._right
    buttons = {"B": True, "A": False, "rightTrig": (0.0,)}
    raw_base_T = _raw_quest_T()

    base_T = teleop._extract_transform({"r": raw_base_T}, "r")
    current_T = teleop._extract_transform({"r": raw_current_T}, "r")
    assert base_T is not None and current_T is not None

    _ = teleop._arm_action(
        controller_T=base_T,
        buttons=buttons,
        state=state,
        enable_button=cfg.enable_button,
        reset_button=cfg.reset_button,
        gripper_button=cfg.gripper_button,
    )
    evo_action = teleop._arm_action(
        controller_T=current_T,
        buttons=buttons,
        state=state,
        enable_button=cfg.enable_button,
        reset_button=cfg.reset_button,
        gripper_button=cfg.gripper_button,
    )
    evo_target_T = _action_target_T(evo_action)

    expected_T = teleop._arm_init_T.copy()
    expected_T[:3, 3] += (current_T[:3, 3] - base_T[:3, 3]) * cfg.pos_scale
    expected_T[:3, :3] = expected_T[:3, :3] @ pin.exp3(
        pin.log3(base_T[:3, :3].T @ current_T[:3, :3]) * cfg.rot_scale
    )

    np.testing.assert_allclose(evo_target_T, expected_T, atol=1e-10)


def test_ee_to_joint_ik_processor_math_flow():
    backend = FakeBackend()
    step = EEToJointIKProcessorStep(ik_backend=backend, async_solve=False)

    action = {
        "enabled": True,
        "reset": False,
        "ee.delta_x": 0.10,
        "ee.delta_y": 0.00,
        "ee.delta_z": 0.00,
        "ee.delta_rx": 0.00,
        "ee.delta_ry": 0.00,
        "ee.delta_rz": 0.00,
        "gripper.pos": 0.08,
    }

    out = step.action(action)

    assert backend.calls, "IK backend was not called"
    target_T, q_seed = backend.calls[-1]
    assert q_seed is not None
    np.testing.assert_allclose(target_T[:3, 3], np.array([0.10, 0.0, 0.0]), atol=1e-8)
    np.testing.assert_allclose(target_T[:3, :3], np.eye(3), atol=1e-8)

    for idx, key in enumerate(PIPER_JOINT_ACTION_KEYS):
        expected_deg = np.degrees(0.1 * (idx + 1))
        assert out[key] == pytest.approx(expected_deg, abs=1e-9)
    assert out["gripper.pos"] == pytest.approx(0.08, abs=1e-9)
    assert out["__absolute_joint_targets__"] is True


def test_ee_to_joint_ik_smooths_joint_solution_before_commanding():
    backend = FakeBackend()
    max_step_rad = math.radians(3.0)
    step = EEToJointIKProcessorStep(
        ik_backend=backend,
        async_solve=False,
        joint_smooth_alpha=0.5,
        max_joint_step_rad=max_step_rad,
    )
    step._state.last_command_q = np.zeros(6, dtype=np.float64)

    out = step.action(
        {
            "enabled": True,
            "reset": False,
            "ee.delta_x": 0.10,
            "ee.delta_y": 0.00,
            "ee.delta_z": 0.00,
            "ee.delta_rx": 0.00,
            "ee.delta_ry": 0.00,
            "ee.delta_rz": 0.00,
            "gripper.pos": 0.08,
        }
    )

    raw_q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    expected_q = np.clip(0.5 * raw_q, -max_step_rad, max_step_rad)
    for idx, key in enumerate(PIPER_JOINT_ACTION_KEYS):
        assert out[key] == pytest.approx(math.degrees(expected_q[idx]), abs=1e-9)
    np.testing.assert_allclose(step._state.last_command_q, expected_q, atol=1e-12)


def test_ee_to_joint_ik_limits_first_solution_from_seed_before_commanding():
    backend = FakeBackend()
    max_step_rad = math.radians(3.0)
    step = EEToJointIKProcessorStep(
        ik_backend=backend,
        async_solve=False,
        joint_smooth_alpha=1.0,
        max_joint_step_rad=max_step_rad,
    )

    out = step.action(
        {
            "enabled": True,
            "reset": False,
            "ee.delta_x": 0.10,
            "ee.delta_y": 0.00,
            "ee.delta_z": 0.00,
            "ee.delta_rx": 0.00,
            "ee.delta_ry": 0.00,
            "ee.delta_rz": 0.00,
            "gripper.pos": 0.08,
        }
    )

    expected_q = np.full(6, max_step_rad, dtype=np.float64)
    for key in PIPER_JOINT_ACTION_KEYS:
        assert out[key] == pytest.approx(math.degrees(max_step_rad), abs=1e-9)
    np.testing.assert_allclose(step._state.last_command_q, expected_q, atol=1e-12)


def test_ee_to_joint_ik_retries_failed_target_with_segmented_step():
    backend = SegmentedRetryBackend()
    step = EEToJointIKProcessorStep(
        ik_backend=backend,
        async_solve=False,
        enable_branch_stable_ik=False,
        target_retry_segment_counts=(2,),
    )

    out = step.action(
        {
            "enabled": True,
            "reset": False,
            "ee.delta_x": 1.0,
            "ee.delta_y": 0.0,
            "ee.delta_z": 0.0,
            "ee.delta_rx": 0.0,
            "ee.delta_ry": 0.0,
            "ee.delta_rz": 0.0,
            "gripper.pos": 0.08,
        }
    )

    assert len(backend.calls) == 2
    np.testing.assert_allclose(backend.calls[0][0][:3, 3], np.array([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(backend.calls[1][0][:3, 3], np.array([0.5, 0.0, 0.0]), atol=1e-12)
    assert backend.fk_calls
    for key in PIPER_JOINT_ACTION_KEYS:
        assert out[key] == pytest.approx(math.degrees(0.2), abs=1e-9)
    assert out["__absolute_joint_targets__"] is True


def test_ee_to_joint_ik_adaptive_alpha_accepts_largest_reachable_target():
    backend = AdaptiveAlphaBackend()
    step = EEToJointIKProcessorStep(
        ik_backend=backend,
        async_solve=False,
        enable_branch_stable_ik=True,
        branch_target_alphas=(1.0, 0.5, 0.25),
    )

    out = step.action(
        {
            "enabled": True,
            "reset": False,
            "ee.delta_x": 1.0,
            "ee.delta_y": 0.0,
            "ee.delta_z": 0.0,
            "ee.delta_rx": 0.0,
            "ee.delta_ry": 0.0,
            "ee.delta_rz": 0.0,
            "gripper.pos": 0.08,
        }
    )

    assert len(backend.calls) == 2
    np.testing.assert_allclose(backend.calls[0][0][:3, 3], np.array([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(backend.calls[1][0][:3, 3], np.array([0.5, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(step._state.target_T[:3, 3], np.array([0.5, 0.0, 0.0]), atol=1e-12)
    for key in PIPER_JOINT_ACTION_KEYS:
        assert out[key] == pytest.approx(math.degrees(0.2), abs=1e-9)
    assert out["__absolute_joint_targets__"] is True


def test_ee_to_joint_ik_does_not_commit_failed_target():
    backend = AlwaysFailBackend()
    step = EEToJointIKProcessorStep(
        ik_backend=backend,
        async_solve=False,
        enable_branch_stable_ik=True,
        branch_target_alphas=(1.0, 0.5),
    )

    out = step.action(
        {
            "enabled": True,
            "reset": False,
            "ee.delta_x": 1.0,
            "ee.delta_y": 0.0,
            "ee.delta_z": 0.0,
            "ee.delta_rx": 0.0,
            "ee.delta_ry": 0.0,
            "ee.delta_rz": 0.0,
            "gripper.pos": 0.08,
        }
    )

    np.testing.assert_allclose(step._state.target_T, np.eye(4, dtype=np.float64), atol=1e-12)
    assert out["gripper.pos"] == pytest.approx(0.08, abs=1e-9)
    assert "__absolute_joint_targets__" not in out
    for key in PIPER_JOINT_ACTION_KEYS:
        assert key not in out


def test_piper_backend_branch_trust_region_intersects_joint_limits():
    backend = _piper_backend_without_solver()
    warm = np.array([0.0, 0.9, -0.9, 0.2, -0.2, 0.0], dtype=np.float64)

    lower, upper = backend._q_bounds_for_warm_start(warm)

    trust = math.radians(10.0)
    np.testing.assert_allclose(lower, np.maximum(-np.ones(6), warm - trust), atol=1e-12)
    np.testing.assert_allclose(upper, np.minimum(np.ones(6), warm + trust), atol=1e-12)


def test_ee_to_joint_ik_disabled_action_outputs_empty():
    step = EEToJointIKProcessorStep(ik_backend=FakeBackend(), async_solve=False)
    out = step.action({"enabled": False, "reset": False, "gripper.pos": 0.08})
    assert out["gripper.pos"] == pytest.approx(0.08, abs=1e-9)
    assert "__absolute_joint_targets__" not in out
    for key in PIPER_JOINT_ACTION_KEYS:
        assert key not in out


def test_ee_to_joint_ik_reset_outputs_interpolated_joint_commands():
    backend = FakeBackend()
    step = EEToJointIKProcessorStep(
        ik_backend=backend,
        async_solve=False,
        reset_interp_steps=4,
        reset_target="arm_init_ik",
    )

    _ = step.action(
        {
            "enabled": True,
            "reset": False,
            "ee.delta_x": 0.10,
            "ee.delta_y": 0.0,
            "ee.delta_z": 0.0,
            "ee.delta_rx": 0.0,
            "ee.delta_ry": 0.0,
            "ee.delta_rz": 0.0,
            "gripper.pos": 0.08,
        }
    )
    step._state.last_q = np.zeros(6, dtype=np.float64)
    out1 = step.action({"enabled": True, "reset": True, "gripper.pos": 0.08})
    out2 = step.action({"enabled": True, "reset": False, "gripper.pos": 0.08})
    out3 = step.action({"enabled": True, "reset": False, "gripper.pos": 0.08})
    out4 = step.action({"enabled": True, "reset": False, "gripper.pos": 0.08})

    for out in [out1, out2, out3, out4]:
        for key in PIPER_JOINT_ACTION_KEYS:
            assert key in out
        assert out["__absolute_joint_targets__"] is True
    np.testing.assert_allclose(backend.calls[-1][0], np.eye(4), atol=1e-12)
    assert out4[PIPER_JOINT_ACTION_KEYS[0]] == pytest.approx(np.degrees(0.1), abs=1e-9)


def test_ee_to_joint_ik_reset_holds_goal_until_observation_settles():
    backend = FakeBackend()
    step = EEToJointIKProcessorStep(
        ik_backend=backend,
        async_solve=False,
        reset_interp_steps=2,
        reset_target="arm_init_ik",
        reset_joint_tolerance_rad=math.radians(1.0),
        reset_settle_ticks=2,
    )
    reset_action = {"enabled": True, "reset": True, "gripper.pos": 0.08}
    disabled_action = {"enabled": False, "reset": False, "gripper.pos": 0.08}
    goal_deg = {key: np.degrees(0.1 * (idx + 1)) for idx, key in enumerate(PIPER_JOINT_ACTION_KEYS)}
    partial_obs = {key: 0.0 for key in PIPER_JOINT_ACTION_KEYS}

    step._state.last_q = np.zeros(6, dtype=np.float64)
    _ = step.action(reset_action)
    _ = step.action(disabled_action)
    assert step._state.reset_plan == []
    assert step._state.reset_active is True

    hold = step(create_transition(observation=partial_obs, action=disabled_action))["action"]

    for key, expected in goal_deg.items():
        assert hold[key] == pytest.approx(expected, abs=1e-9)
    assert step._state.reset_active is True

    _ = step(create_transition(observation=goal_deg, action=disabled_action))["action"]
    assert step._state.reset_active is True
    _ = step(create_transition(observation=goal_deg, action=disabled_action))["action"]
    assert step._state.reset_active is False
    np.testing.assert_allclose(backend.previous_q, np.deg2rad(np.fromiter(goal_deg.values(), dtype=np.float64)), atol=1e-9)


def test_ee_to_joint_ik_reset_prefers_captured_initial_observation():
    backend = FakeBackend()
    step = EEToJointIKProcessorStep(
        ik_backend=backend,
        async_solve=False,
        reset_interp_steps=1,
        reset_target="initial_observation",
    )
    idle_action = {"enabled": False, "reset": False, "gripper.pos": 0.08}
    reset_action = {"enabled": True, "reset": True, "gripper.pos": 0.08}
    initial_obs = {key: 12.0 for key in PIPER_JOINT_ACTION_KEYS}
    current_obs = {key: 30.0 for key in PIPER_JOINT_ACTION_KEYS}

    _ = step(create_transition(observation=initial_obs, action=idle_action))["action"]
    out = step(create_transition(observation=current_obs, action=reset_action))["action"]

    assert backend.calls == []
    for key in PIPER_JOINT_ACTION_KEYS:
        assert out[key] == pytest.approx(12.0, abs=1e-9)


def test_ee_to_joint_ik_reset_defaults_to_home_q():
    backend = FakeBackend()
    step = EEToJointIKProcessorStep(ik_backend=backend, async_solve=False, reset_interp_steps=1)
    step._state.last_q = np.ones(6, dtype=np.float64)

    out = step.action({"enabled": True, "reset": True, "gripper.pos": 0.08})

    assert backend.calls == []
    for key in PIPER_JOINT_ACTION_KEYS:
        assert out[key] == pytest.approx(0.0, abs=1e-9)


def test_ee_to_joint_ik_reset_supports_configured_joint_degree_target():
    backend = FakeBackend()
    step = EEToJointIKProcessorStep(
        ik_backend=backend,
        async_solve=False,
        reset_interp_steps=1,
        reset_target="joint_degrees",
        reset_joint_target_degrees=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    )
    step._state.last_q = np.zeros(6, dtype=np.float64)

    out = step.action({"enabled": True, "reset": True, "gripper.pos": 0.08})

    for idx, key in enumerate(PIPER_JOINT_ACTION_KEYS):
        assert out[key] == pytest.approx(float(idx + 1), abs=1e-9)


def test_ee_to_joint_ik_small_degree_observation_is_not_treated_as_radians():
    backend = FakeBackend()
    step = EEToJointIKProcessorStep(ik_backend=backend, async_solve=False)
    action = {
        "enabled": True,
        "reset": False,
        "ee.delta_x": 0.01,
        "ee.delta_y": 0.0,
        "ee.delta_z": 0.0,
        "ee.delta_rx": 0.0,
        "ee.delta_ry": 0.0,
        "ee.delta_rz": 0.0,
        "gripper.pos": 0.08,
    }
    obs_deg = {key: 2.0 for key in PIPER_JOINT_ACTION_KEYS}

    _ = step(create_transition(observation=obs_deg, action=action))["action"]

    assert backend.calls
    _, q_seed = backend.calls[-1]
    assert q_seed is not None
    np.testing.assert_allclose(q_seed, np.deg2rad(np.full(6, 2.0)), atol=1e-9)


def test_ee_to_joint_ik_reset_clears_retained_targets_and_async_state():
    arm_init_T = _raw_quest_T(0.19, 0.0, 0.2)
    step = EEToJointIKProcessorStep(ik_backend=FakeBackend(), arm_init_T=arm_init_T, async_solve=False)
    step._state.target_T = _raw_quest_T(0.5, 0.0, 0.0)
    step._state.last_q = np.ones(6, dtype=np.float64)
    step._state.initial_q = np.ones(6, dtype=np.float64)
    step._state.armed = True
    step._state.reset_plan = [np.ones(6, dtype=np.float64)]
    step._state.reset_active = True
    step._state.reset_goal_q = np.ones(6, dtype=np.float64)
    step._state.reset_settle_count = 1
    step._state.ik_inflight = True
    step._state.async_action_ready = {"joint_1.pos": 1.0}
    backend = step.ik_backend
    backend.set_previous_q(np.ones(6, dtype=np.float64))
    generation = step._state.solve_generation

    step.reset()

    np.testing.assert_allclose(step._state.target_T, arm_init_T, atol=1e-12)
    assert step._state.last_q is None
    assert step._state.initial_q is None
    assert step._state.armed is False
    assert step._state.reset_plan == []
    assert step._state.reset_active is False
    assert step._state.reset_goal_q is None
    assert step._state.reset_settle_count == 0
    assert step._state.ik_inflight is False
    assert step._state.async_action_ready is None
    assert step._state.solve_generation == generation + 1
    assert backend.previous_q is None


def test_ee_to_joint_ik_async_returns_result_on_following_tick():
    step = EEToJointIKProcessorStep(ik_backend=SlowFakeBackend(), async_solve=True)
    action = {
        "enabled": True,
        "reset": False,
        "ee.delta_x": 0.05,
        "ee.delta_y": 0.0,
        "ee.delta_z": 0.0,
        "ee.delta_rx": 0.0,
        "ee.delta_ry": 0.0,
        "ee.delta_rz": 0.0,
        "gripper.pos": 0.08,
    }
    first = step.action(action)
    assert first["gripper.pos"] == pytest.approx(0.08, abs=1e-9)
    assert "__absolute_joint_targets__" not in first
    for key in PIPER_JOINT_ACTION_KEYS:
        assert key not in first
    time.sleep(0.04)
    second = step.action(action)
    assert "__absolute_joint_targets__" in second
    assert second["gripper.pos"] == pytest.approx(0.08, abs=1e-9)


def test_ee_to_joint_ik_async_ready_uses_current_gripper_value():
    step = EEToJointIKProcessorStep(ik_backend=SlowFakeBackend(), async_solve=True)
    open_action = {
        "enabled": True,
        "reset": False,
        "ee.delta_x": 0.05,
        "ee.delta_y": 0.0,
        "ee.delta_z": 0.0,
        "ee.delta_rx": 0.0,
        "ee.delta_ry": 0.0,
        "ee.delta_rz": 0.0,
        "gripper.pos": 0.08,
    }
    closed_action = {**open_action, "gripper.pos": 0.0}

    first = step.action(open_action)
    assert first["gripper.pos"] == pytest.approx(0.08, abs=1e-9)
    assert "__absolute_joint_targets__" not in first
    for key in PIPER_JOINT_ACTION_KEYS:
        assert key not in first
    time.sleep(0.04)
    second = step.action(closed_action)

    assert "__absolute_joint_targets__" in second
    assert second["gripper.pos"] == pytest.approx(0.0, abs=1e-9)


def test_ee_to_joint_ik_seed_from_observation_converts_deg_to_rad():
    backend = FakeBackend()
    step = EEToJointIKProcessorStep(ik_backend=backend, async_solve=False)
    action = {
        "enabled": True,
        "reset": False,
        "ee.delta_x": 0.01,
        "ee.delta_y": 0.0,
        "ee.delta_z": 0.0,
        "ee.delta_rx": 0.0,
        "ee.delta_ry": 0.0,
        "ee.delta_rz": 0.0,
        "gripper.pos": 0.08,
    }
    obs_deg = {key: 30.0 for key in PIPER_JOINT_ACTION_KEYS}

    _ = step(create_transition(observation=obs_deg, action=action))["action"]

    assert backend.calls
    _, q_seed = backend.calls[-1]
    assert q_seed is not None
    np.testing.assert_allclose(q_seed, np.deg2rad(np.full(6, 30.0)), atol=1e-9)


def test_ee_to_joint_ik_invalid_observation_seed_falls_back_to_home():
    backend = FakeBackend()
    step = EEToJointIKProcessorStep(ik_backend=backend, async_solve=False)
    action = {
        "enabled": True,
        "reset": False,
        "ee.delta_x": 0.01,
        "ee.delta_y": 0.0,
        "ee.delta_z": 0.0,
        "ee.delta_rx": 0.0,
        "ee.delta_ry": 0.0,
        "ee.delta_rz": 0.0,
        "gripper.pos": 0.08,
    }
    obs_bad = {key: 10000.0 for key in PIPER_JOINT_ACTION_KEYS}

    _ = step(create_transition(observation=obs_bad, action=action))["action"]

    assert backend.calls
    _, q_seed = backend.calls[-1]
    assert q_seed is not None
    np.testing.assert_allclose(q_seed, np.zeros(6, dtype=np.float64), atol=1e-9)


def test_dual_ik_step_preserves_left_right_output_prefixes():
    left = EEToJointIKProcessorStep(ik_backend=FakeBackend(), async_solve=False, input_prefix="left_", output_prefix="left_")
    right = EEToJointIKProcessorStep(
        ik_backend=FakeBackend(), async_solve=False, input_prefix="right_", output_prefix="right_"
    )
    dual = DualArmEEToJointIKProcessorStep(left_step=left, right_step=right)
    action = {
        "left_enabled": True,
        "left_reset": False,
        "left_ee.delta_x": 0.05,
        "left_ee.delta_y": 0.0,
        "left_ee.delta_z": 0.0,
        "left_ee.delta_rx": 0.0,
        "left_ee.delta_ry": 0.0,
        "left_ee.delta_rz": 0.0,
        "left_gripper.pos": 0.08,
        "right_enabled": True,
        "right_reset": False,
        "right_ee.delta_x": 0.06,
        "right_ee.delta_y": 0.0,
        "right_ee.delta_z": 0.0,
        "right_ee.delta_rx": 0.0,
        "right_ee.delta_ry": 0.0,
        "right_ee.delta_rz": 0.0,
        "right_gripper.pos": 0.0,
    }
    out = dual(create_transition(observation={}, action=action))["action"]
    assert "left_joint_1.pos" in out and "right_joint_1.pos" in out
    assert "left___absolute_joint_targets__" in out and "right___absolute_joint_targets__" in out


def test_dual_ik_step_reset_clears_both_arms():
    left = EEToJointIKProcessorStep(ik_backend=FakeBackend(), async_solve=False, input_prefix="left_", output_prefix="left_")
    right = EEToJointIKProcessorStep(
        ik_backend=FakeBackend(), async_solve=False, input_prefix="right_", output_prefix="right_"
    )
    dual = DualArmEEToJointIKProcessorStep(left_step=left, right_step=right)
    left._state.last_q = np.ones(6, dtype=np.float64)
    right._state.last_q = np.ones(6, dtype=np.float64)

    dual.reset()

    assert left._state.last_q is None
    assert right._state.last_q is None


def test_quest3_vr_single_arm_contract_enable_move_release():
    cfg = Quest3VRTeleopConfig()
    teleop = Quest3VRTeleop(cfg)
    teleop._is_connected = True
    teleop._reader = _FakeReader(
        [
            ({"r": _T(0.0, 0.0, 0.0)}, {"B": False, "A": False, "rightTrig": (0.0,)}),
            ({"r": _T(0.0, 0.0, 0.0)}, {"B": True, "A": False, "rightTrig": (0.0,)}),
            ({"r": _T(0.2, 0.0, 0.0)}, {"B": True, "A": False, "rightTrig": (0.0,)}),
            ({"r": _T(0.2, 0.0, 0.0)}, {"B": False, "A": False, "rightTrig": (0.0,)}),
        ]
    )

    a0 = teleop.get_action()
    assert a0["enabled"] is False
    a1 = teleop.get_action()
    assert a1["enabled"] is True
    a2 = teleop.get_action()
    assert all(k in a2 for k in ["ee.target_x", "ee.target_y", "ee.target_z", "ee.target_rx", "ee.target_ry", "ee.target_rz"])
    a3 = teleop.get_action()
    assert a3["enabled"] is False


def test_quest3_vr_prepare_episode_reset_emits_reset_without_controller_pose():
    cfg = Quest3VRTeleopConfig()
    teleop = Quest3VRTeleop(cfg)
    teleop._is_connected = True
    teleop._reader = _FakeReader([({}, {"B": False, "A": False, "rightTrig": (0.0,)})])
    teleop._right.last_T = _raw_quest_T(0.5, 0.0, 0.0)
    teleop._right.arm_T = teleop._right.last_T.copy()

    teleop.prepare_episode_reset()
    action = teleop.get_action()

    assert action["enabled"] is False
    assert action["reset"] is True
    assert action["gripper.pos"] == pytest.approx(cfg.gripper_reset_value, abs=1e-9)
    assert teleop._right.pending_episode_reset is False
    np.testing.assert_allclose(teleop._right.last_T, teleop._arm_init_T, atol=1e-12)


def test_quest3_vr_prepare_episode_start_drops_retained_vr_state():
    cfg = Quest3VRTeleopConfig()
    teleop = Quest3VRTeleop(cfg)
    teleop._right.raw_T = _raw_quest_T(0.1, 0.0, 0.0)
    teleop._right.smooth_T = _raw_quest_T(0.2, 0.0, 0.0)
    teleop._right.base_T = _raw_quest_T(0.3, 0.0, 0.0)
    teleop._right.enable_prev = True
    teleop._right.reset_prev = True
    teleop._right.trig_prev = True
    teleop._right.pending_episode_reset = True

    teleop.prepare_episode_start()

    assert teleop._right.raw_T is None
    assert teleop._right.smooth_T is None
    assert teleop._right.base_T is None
    assert teleop._right.enable_prev is False
    assert teleop._right.reset_prev is False
    assert teleop._right.trig_prev is False
    assert teleop._right.pending_episode_reset is False
    np.testing.assert_allclose(teleop._right.last_T, teleop._arm_init_T, atol=1e-12)


def test_quest3_vr_single_arm_toggles_gripper_on_b_trigger_same_sample():
    cfg = Quest3VRTeleopConfig()
    teleop = Quest3VRTeleop(cfg)
    teleop._is_connected = True
    teleop._reader = _FakeReader(
        [
            ({"r": _T(0.0, 0.0, 0.0)}, {"B": True, "A": False, "rightTrig": (1.0,)}),
        ]
    )

    action = teleop.get_action()

    assert action["enabled"] is True
    assert action["gripper.pos"] == pytest.approx(cfg.gripper_close_value, abs=1e-9)


def test_quest3_vr_single_arm_trigger_falls_back_to_sdk_boolean_key():
    cfg = Quest3VRTeleopConfig()
    teleop = Quest3VRTeleop(cfg)
    teleop._is_connected = True
    teleop._reader = _FakeReader(
        [
            ({"r": _T(0.0, 0.0, 0.0)}, {"B": True, "A": False, "RTr": True}),
        ]
    )

    action = teleop.get_action()

    assert action["enabled"] is True
    assert action["gripper.pos"] == pytest.approx(cfg.gripper_close_value, abs=1e-9)


def test_quest3_vr_dual_arm_contract_outputs_prefixed_actions():
    cfg = BiQuest3VRTeleopConfig(calibration_dir=Path("/tmp/lerobot_test_calib_bi_quest3_vr"))
    teleop = BiQuest3VRTeleop(cfg)
    teleop._is_connected = True
    teleop._reader = _FakeReader(
        [
            (
                {"r": _T(0.0, 0.0, 0.0), "l": _T(0.0, 0.0, 0.0)},
                {"B": True, "A": False, "rightTrig": (0.0,), "Y": True, "X": False, "leftTrig": (0.0,)},
            ),
            (
                {"r": _T(0.03, 0.0, 0.0), "l": _T(0.0, 0.03, 0.0)},
                {"B": True, "A": False, "rightTrig": (0.0,), "Y": True, "X": False, "leftTrig": (0.0,)},
            ),
        ]
    )

    _ = teleop.get_action()
    out = teleop.get_action()
    assert "left_enabled" in out and "right_enabled" in out
    assert "left_ee.delta_x" in out and "right_ee.delta_x" in out
