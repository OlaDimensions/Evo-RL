from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
    ParameterSpec(
        key="resume",
        label="Resume Existing Dataset",
        default="false",
    ),
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
    "openpi_server_root": "OPENPI_SERVER_ROOT",
    "policy_host": "POLICY_HOST",
    "policy_port": "POLICY_PORT",
}


def recording_parameters_to_env(params: RecordingParameters) -> dict[str, str]:
    env = {}
    for key, env_name in PARAMETER_ENV_NAMES.items():
        value = params.get(key)
        if value:
            env[env_name] = value
    return env
