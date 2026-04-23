import math

import pytest

from lerobot.datasets.utils import build_dataset_frame
from lerobot.scripts.ee_pose_action_utils import BIMANUAL_EE_RPY_NAMES
from lerobot.scripts.recording_ee_pose import (
    PiperEEPoseStorage,
    replace_low_dim_features_with_piper_ee_pose,
)
from lerobot.scripts.recording_loop import _zero_values_for_feature
from lerobot.utils.constants import ACTION, OBS_STATE


class _EndPose:
    def __init__(
        self,
        x: int = 100_000,
        y: int = -200_000,
        z: int = 300_000,
        rx: int = 1_000,
        ry: int = -2_000,
        rz: int = 3_000,
    ):
        self.X_axis = x
        self.Y_axis = y
        self.Z_axis = z
        self.RX_axis = rx
        self.RY_axis = ry
        self.RZ_axis = rz


class _PoseMsg:
    def __init__(self, end_pose=None):
        self.end_pose = end_pose if end_pose is not None else _EndPose()


class _GripperState:
    grippers_angle = 42_000


class _GripperMsg:
    gripper_state = _GripperState()


class _PiperSdk:
    def __init__(self, poses=None):
        self._poses = list(poses) if poses is not None else [_EndPose()]
        self._idx = 0

    def GetArmEndPoseMsgs(self):
        pose = self._poses[min(self._idx, len(self._poses) - 1)]
        self._idx += 1
        return _PoseMsg(pose)

    def GetArmGripperMsgs(self):
        return _GripperMsg()


class _SinglePiper:
    def __init__(self, poses=None):
        self.arm = _PiperSdk(poses)


class _BiPiper:
    def __init__(self, left_poses=None, right_poses=None):
        self.left_arm = _SinglePiper(left_poses)
        self.right_arm = _SinglePiper(right_poses)


def _quat_norm(values, prefix="ee"):
    return math.sqrt(
        values[f"{prefix}.qx"] ** 2
        + values[f"{prefix}.qy"] ** 2
        + values[f"{prefix}.qz"] ** 2
        + values[f"{prefix}.qw"] ** 2
    )


def test_single_piper_sdk_pose_storage_reads_8d_pose():
    values = PiperEEPoseStorage(_SinglePiper()).read()

    assert list(values) == ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"]
    assert values["ee.x"] == pytest.approx(0.1)
    assert values["ee.y"] == pytest.approx(-0.2)
    assert values["ee.z"] == pytest.approx(0.3)
    assert values["ee.qx"] == pytest.approx(0.009179, abs=1e-6)
    assert values["ee.qy"] == pytest.approx(-0.017218, abs=1e-6)
    assert values["ee.qz"] == pytest.approx(0.026324, abs=1e-6)
    assert values["ee.qw"] == pytest.approx(0.999464, abs=1e-6)
    assert _quat_norm(values) == pytest.approx(1.0)
    assert values["gripper.pos"] == pytest.approx(42.0)


def test_bimanual_piper_sdk_pose_storage_reads_left_and_right_poses():
    values = PiperEEPoseStorage(_BiPiper()).read()

    assert len(values) == 16
    assert values["left_ee.x"] == pytest.approx(0.1)
    assert values["left_gripper.pos"] == pytest.approx(42.0)
    assert values["right_ee.x"] == pytest.approx(0.1)
    assert values["right_gripper.pos"] == pytest.approx(42.0)
    assert _quat_norm(values, "left_ee") == pytest.approx(1.0)
    assert _quat_norm(values, "right_ee") == pytest.approx(1.0)


def test_replace_low_dim_features_with_piper_ee_pose_updates_state_action_and_policy_action():
    features = {
        ACTION: {"dtype": "float32", "shape": (7,), "names": ["joint_1.pos"]},
        OBS_STATE: {"dtype": "float32", "shape": (7,), "names": ["joint_1.pos"]},
        "complementary_info.policy_action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_1.pos"],
        },
    }

    replace_low_dim_features_with_piper_ee_pose(features, _SinglePiper())

    assert features[ACTION]["shape"] == (8,)
    assert features[ACTION]["names"] == [
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.qx",
        "ee.qy",
        "ee.qz",
        "ee.qw",
        "gripper.pos",
    ]
    assert features[OBS_STATE] == features[ACTION]
    assert features["complementary_info.policy_action"] == features[ACTION]


def test_zero_policy_action_uses_rpy_policy_action_schema_names():
    features = {
        "complementary_info.policy_action": {
            "dtype": "float32",
            "shape": (len(BIMANUAL_EE_RPY_NAMES),),
            "names": list(BIMANUAL_EE_RPY_NAMES),
        }
    }

    values = _zero_values_for_feature(features, "complementary_info.policy_action")
    frame = build_dataset_frame(features, values, prefix="complementary_info.policy_action")

    assert set(values) == set(BIMANUAL_EE_RPY_NAMES)
    assert frame["complementary_info.policy_action"].shape == (len(BIMANUAL_EE_RPY_NAMES),)


def test_single_piper_quaternion_sign_is_continuous():
    robot = _SinglePiper(
        poses=[
            _EndPose(rx=0, ry=0, rz=181_000),
            _EndPose(rx=0, ry=0, rz=-179_000),
        ]
    )
    storage = PiperEEPoseStorage(robot)

    first = storage.read()
    second = storage.read()

    dot = (
        first["ee.qx"] * second["ee.qx"]
        + first["ee.qy"] * second["ee.qy"]
        + first["ee.qz"] * second["ee.qz"]
        + first["ee.qw"] * second["ee.qw"]
    )
    assert dot > 0.99


def test_bimanual_piper_quaternion_sign_is_continuous_per_arm():
    robot = _BiPiper(
        left_poses=[_EndPose(rx=0, ry=0, rz=181_000), _EndPose(rx=0, ry=0, rz=-179_000)],
        right_poses=[_EndPose(rx=181_000, ry=0, rz=0), _EndPose(rx=-179_000, ry=0, rz=0)],
    )
    storage = PiperEEPoseStorage(robot)

    first = storage.read()
    second = storage.read()

    left_dot = sum(
        first[key] * second[key]
        for key in ("left_ee.qx", "left_ee.qy", "left_ee.qz", "left_ee.qw")
    )
    right_dot = sum(
        first[key] * second[key]
        for key in ("right_ee.qx", "right_ee.qy", "right_ee.qz", "right_ee.qw")
    )
    assert left_dot > 0.99
    assert right_dot > 0.99
