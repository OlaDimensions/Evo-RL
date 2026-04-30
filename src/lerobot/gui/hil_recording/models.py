from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StatusLevel(str, Enum):
    UNKNOWN = "unknown"
    CHECKING = "checking"
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


class RecordingState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    RESETTING = "resetting"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    default: str = ""
    placeholder: str = ""
    choices: tuple[tuple[str, str], ...] = ()


@dataclass
class RecordingParameters:
    values: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        value = self.values.get(key, default)
        return value.strip() if isinstance(value, str) else default


@dataclass(frozen=True)
class HealthCheckResult:
    name: str
    level: StatusLevel
    detail: str = ""


@dataclass(frozen=True)
class HardwareHealthReport:
    camera: HealthCheckResult
    can: HealthCheckResult
    adb: HealthCheckResult


@dataclass(frozen=True)
class CameraSlotSpec:
    key: str
    label: str
    arm: str
    camera_name: str
    realsense_serial: str
    gopro_index_or_path: str


CAMERA_TYPE_CHOICES = (
    ("gopro", "GoPro"),
    ("realsense", "RealSense"),
    ("none", "No Camera"),
)

CAMERA_SLOT_SPECS = (
    CameraSlotSpec(
        key="left_wrist",
        label="Left Wrist Camera",
        arm="left",
        camera_name="wrist",
        realsense_serial="152122072280",
        gopro_index_or_path="/dev/video0",
    ),
    CameraSlotSpec(
        key="right_wrist",
        label="Right Wrist Camera",
        arm="right",
        camera_name="wrist",
        realsense_serial="008222070618",
        gopro_index_or_path="/dev/video2",
    ),
    CameraSlotSpec(
        key="right_front",
        label="Right Front Camera",
        arm="right",
        camera_name="front",
        realsense_serial="213622074413",
        gopro_index_or_path="/dev/video4",
    ),
)

CAMERA_SLOT_KEYS = tuple(slot.key for slot in CAMERA_SLOT_SPECS)
CAMERA_TYPE_KEYS = tuple(f"{slot.key}_camera_type" for slot in CAMERA_SLOT_SPECS)
CAMERA_ID_KEYS = tuple(f"{slot.key}_camera_id" for slot in CAMERA_SLOT_SPECS)
CAMERA_PARAMETER_KEYS = tuple(
    key for slot in CAMERA_SLOT_SPECS for key in (f"{slot.key}_camera_type", f"{slot.key}_camera_id")
)

DEFAULT_GOPRO_CAPTURE_WIDTH = 1920
DEFAULT_GOPRO_CAPTURE_HEIGHT = 1080
DEFAULT_GOPRO_WIDTH = 448
DEFAULT_GOPRO_HEIGHT = 448
DEFAULT_GOPRO_FPS = 30
DEFAULT_GOPRO_CROP_RATIO = 1.0
DEFAULT_GOPRO_FOURCC = "MJPG"
DEFAULT_CAMERA_WARMUP_S = 2
DEFAULT_REALSENSE_WIDTH = 640
DEFAULT_REALSENSE_HEIGHT = 480
DEFAULT_REALSENSE_FPS = 30


def _camera_parameter_specs() -> list[ParameterSpec]:
    specs: list[ParameterSpec] = []
    for slot in CAMERA_SLOT_SPECS:
        prefix = slot.key
        specs.extend(
            [
                ParameterSpec(
                    key=f"{prefix}_camera_type",
                    label=slot.label,
                    default="gopro",
                    choices=CAMERA_TYPE_CHOICES,
                ),
                ParameterSpec(
                    key=f"{prefix}_camera_id",
                    label=f"{slot.label} ID",
                    default=slot.gopro_index_or_path,
                    placeholder="/dev/video0 or RealSense serial",
                ),
            ]
        )
    return specs


DEFAULT_PARAMETER_SPECS = [
    ParameterSpec(
        key="dataset_name",
        label="Dataset Name",
        default="ruanafan/eval_evo-rl-data-pnp-vr-ee-pose-gui",
        placeholder="hf_user/dataset_name",
    ),
    ParameterSpec(
        key="dataset_single_task",
        label="Task",
        default=(
            "Locate and pull open the air fryer drawer, pick up the sweet potato and place it "
            "steadily into the basket, then push the drawer back."
        ),
        placeholder="Task instruction for this dataset",
    ),
    ParameterSpec(
        key="dataset_num_episodes",
        label="Num Episodes",
        default="10",
        placeholder="10",
    ),
    ParameterSpec(
        key="dataset_episode_time_s",
        label="Episode Time (s)",
        default="150",
        placeholder="150",
    ),
    ParameterSpec(
        key="dataset_reset_time_s",
        label="Reset Time (s)",
        default="20",
        placeholder="20",
    ),
    *_camera_parameter_specs(),
    ParameterSpec(
        key="policy_mode",
        label="Policy Mode",
        default="openpi_remote",
        choices=(
            ("teleop_only", "Teleop Only"),
            ("openpi_remote", "OpenPI Remote"),
            ("local_path", "Local Policy Path"),
        ),
    ),
    ParameterSpec(
        key="policy_path",
        label="Policy Path",
        default="",
        placeholder="outputs/train/.../pretrained_model",
    ),
    ParameterSpec(
        key="openpi_policy_dir",
        label="OpenPI Policy Dir",
        default="/home/ola/code/Evo-RL/outputs/train/openpi05-0421-all-rpy/29999",
        placeholder="/home/ola/code/Evo-RL/outputs/train/.../checkpoint",
    ),
    ParameterSpec(
        key="openpi_policy_config",
        label="OpenPI Policy Config",
        default="pi05_bipiper_absolute_lora",
        placeholder="pi05_bipiper_absolute_lora",
    ),
    ParameterSpec(
        key="openpi_server_root",
        label="OpenPI Server Root",
        default="/home/ola/code/openpi",
        placeholder="/home/ola/code/openpi",
    ),
    ParameterSpec(
        key="policy_host",
        label="Policy Host",
        default="127.0.0.1",
        placeholder="127.0.0.1",
    ),
    ParameterSpec(
        key="policy_port",
        label="Policy Port",
        default="8000",
        placeholder="8000",
    ),
]


PARAMETER_ENV_NAMES = {
    "dataset_name": "DATASET_REPO_ID",
    "dataset_single_task": "DATASET_SINGLE_TASK",
    "dataset_num_episodes": "DATASET_NUM_EPISODES",
    "dataset_episode_time_s": "DATASET_EPISODE_TIME_S",
    "dataset_reset_time_s": "DATASET_RESET_TIME_S",
    "resume": "RESUME",
    "policy_mode": "POLICY_MODE",
    "policy_path": "POLICY_PATH",
    "openpi_policy_dir": "OPENPI_POLICY_DIR",
    "openpi_policy_config": "OPENPI_POLICY_CONFIG",
    "openpi_server_root": "OPENPI_SERVER_ROOT",
    "policy_host": "POLICY_HOST",
    "policy_port": "POLICY_PORT",
}


def _legacy_camera_type(profile: str, slot: CameraSlotSpec) -> str:
    if profile == "realsense":
        return "realsense"
    if profile == "none":
        return "none"
    if profile == "gopro":
        return "gopro"
    return "gopro"


def migrate_legacy_camera_parameters(values: dict[str, str]) -> dict[str, str]:
    if any(key in values for key in (*CAMERA_TYPE_KEYS, *CAMERA_ID_KEYS)):
        return values

    profile = values.get("camera_profile", "")
    if not profile:
        return values

    migrated = dict(values)
    for slot in CAMERA_SLOT_SPECS:
        migrated[f"{slot.key}_camera_type"] = _legacy_camera_type(profile, slot)
        if profile == "realsense":
            migrated[f"{slot.key}_camera_id"] = slot.realsense_serial
        else:
            migrated[f"{slot.key}_camera_id"] = slot.gopro_index_or_path
    return migrated


def _camera_type_for_slot(params: RecordingParameters, slot: CameraSlotSpec) -> str:
    key = f"{slot.key}_camera_type"
    if key in params.values:
        return params.get(key, "gopro") or "gopro"
    return _legacy_camera_type(params.get("camera_profile", "gopro") or "gopro", slot)


def _default_parameter_value(key: str) -> str:
    for spec in DEFAULT_PARAMETER_SPECS:
        if spec.key == key:
            return spec.default
    return ""


def _required_str(params: RecordingParameters, key: str, label: str) -> str:
    value = params.get(key) or _default_parameter_value(key)
    if not value:
        raise ValueError(f"{label} is required.")
    return value


def _gopro_config(params: RecordingParameters, slot: CameraSlotSpec) -> dict[str, Any]:
    prefix = slot.key
    return {
        "type": "gopro",
        "index_or_path": _required_str(params, f"{prefix}_camera_id", f"{slot.label} ID"),
        "capture_width": DEFAULT_GOPRO_CAPTURE_WIDTH,
        "capture_height": DEFAULT_GOPRO_CAPTURE_HEIGHT,
        "width": DEFAULT_GOPRO_WIDTH,
        "height": DEFAULT_GOPRO_HEIGHT,
        "fps": DEFAULT_GOPRO_FPS,
        "crop_ratio": DEFAULT_GOPRO_CROP_RATIO,
        "fourcc": DEFAULT_GOPRO_FOURCC,
        "warmup_s": DEFAULT_CAMERA_WARMUP_S,
    }


def _realsense_config(params: RecordingParameters, slot: CameraSlotSpec) -> dict[str, Any]:
    prefix = slot.key
    return {
        "type": "intelrealsense",
        "serial_number_or_name": _required_str(params, f"{prefix}_camera_id", f"{slot.label} ID"),
        "width": DEFAULT_REALSENSE_WIDTH,
        "height": DEFAULT_REALSENSE_HEIGHT,
        "fps": DEFAULT_REALSENSE_FPS,
        "warmup_s": DEFAULT_CAMERA_WARMUP_S,
    }


def build_robot_camera_configs(params: RecordingParameters) -> tuple[dict[str, Any], dict[str, Any]]:
    left_cameras: dict[str, Any] = {}
    right_cameras: dict[str, Any] = {}
    for slot in CAMERA_SLOT_SPECS:
        camera_type = _camera_type_for_slot(params, slot)
        if camera_type == "none":
            continue
        if camera_type == "gopro":
            camera_config = _gopro_config(params, slot)
        elif camera_type == "realsense":
            camera_config = _realsense_config(params, slot)
        else:
            raise ValueError(f"Unknown camera type for {slot.label}: {camera_type}")

        target = left_cameras if slot.arm == "left" else right_cameras
        target[slot.camera_name] = camera_config
    return left_cameras, right_cameras


def camera_configs_to_env(left_cameras: dict[str, Any], right_cameras: dict[str, Any]) -> dict[str, str]:
    return {
        "CAMERA_PROFILE": "custom",
        "ROBOT_LEFT_CAMERAS": json.dumps(left_cameras, sort_keys=True),
        "ROBOT_RIGHT_CAMERAS": json.dumps(right_cameras, sort_keys=True),
    }


def robot_camera_env_from_params(params: RecordingParameters) -> dict[str, str]:
    return camera_configs_to_env(*build_robot_camera_configs(params))


def recording_parameters_to_env(params: RecordingParameters) -> dict[str, str]:
    env = {}
    for key, env_name in PARAMETER_ENV_NAMES.items():
        value = params.get(key)
        if value:
            env[env_name] = value
    env.update(robot_camera_env_from_params(params))
    return env
