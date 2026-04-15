import ast
import math
import numpy as np
import pytest
import time
from pathlib import Path

from lerobot.processor.converters import create_transition
from lerobot.teleoperators.bi_quest3_vr.bi_quest3_vr import BiQuest3VRTeleop
from lerobot.teleoperators.bi_quest3_vr.config_bi_quest3_vr import BiQuest3VRTeleopConfig
from lerobot.teleoperators.quest3_vr.config_quest3_vr import Quest3VRTeleopConfig
from lerobot.teleoperators.quest3_vr.ee_to_joint_ik import DualArmEEToJointIKProcessorStep, EEToJointIKProcessorStep
from lerobot.teleoperators.quest3_vr.quest3_vr import Quest3VRTeleop
from lerobot.utils.piper_sdk import PIPER_JOINT_ACTION_KEYS


QUEST3VR_WS_ROOT = Path("/home/ola/code/quest3VR_ws/src/oculus_reader/oculus_reader")


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
    from lerobot.teleoperators.quest3_vr.piper_pinocchio import PiperIKConfig, PiperPinocchioIKBackend

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

    def home_q(self):
        return np.zeros(6, dtype=np.float64)

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


class SlowFakeBackend(FakeBackend):
    def solve(self, target_T, q_seed=None):
        time.sleep(0.02)
        return super().solve(target_T, q_seed=q_seed)


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


def test_ee_to_joint_ik_disabled_action_outputs_empty():
    step = EEToJointIKProcessorStep(ik_backend=FakeBackend(), async_solve=False)
    out = step.action({"enabled": False, "reset": False, "gripper.pos": 0.08})
    assert out["gripper.pos"] == pytest.approx(0.08, abs=1e-9)
    assert out["__absolute_joint_targets__"] is True


def test_ee_to_joint_ik_reset_outputs_interpolated_joint_commands():
    backend = FakeBackend()
    step = EEToJointIKProcessorStep(ik_backend=backend, async_solve=False, reset_interp_steps=4)

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
    assert first["__absolute_joint_targets__"] is True
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
    assert first["__absolute_joint_targets__"] is True
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
