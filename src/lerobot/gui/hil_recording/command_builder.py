from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from lerobot.gui.hil_recording.models import RecordingParameters


DEFAULT_RECORD_SCRIPT = Path(__file__).resolve().with_name("run_teleop_with_vt3_ik.sh")
DEFAULT_OPENPI_SERVER_ROOT = Path("/home/ola/code/openpi")
OPENPI_SERVER_POLICY_CONFIG = "pi05_bipiper_absolute_lora"

PARAMETER_CLI_ARGS = {
    "dataset_name": "--dataset.repo_id",
    "dataset_single_task": "--dataset.single_task",
    "dataset_num_episodes": "--dataset.num_episodes",
    "dataset_episode_time_s": "--dataset.episode_time_s",
    "dataset_reset_time_s": "--dataset.reset_time_s",
}

POLICY_AUX_CLI_ARGS = {
    "--policy_tighten_closed_gripper": "true",
    "--policy_gripper_tighten_enter_threshold": "50",
    "--policy_gripper_tighten_release_threshold": "65",
    "--policy_sync_to_teleop": "true",
    "--policy_sync_parallel": "true",
    "--acp_inference.enable": "true",
}


def parameter_is_enabled(params: RecordingParameters, key: str) -> bool:
    return params.get(key).lower() in {"1", "true", "yes", "on"}


@dataclass
class RecordingCommandBuilder:
    """Build the external recording command from GUI parameters.

    Keep this class intentionally small: robot, teleop, and policy arguments are expected
    to evolve at the lab bench, and `get_command_args` is the extension point.
    """

    repo_root: Path

    def _dataset_cli_args(self, params: RecordingParameters) -> list[str]:
        args = []
        for key, cli_name in PARAMETER_CLI_ARGS.items():
            value = params.get(key)
            if value:
                args.append(f"{cli_name}={value}")
        return args

    def _policy_cli_args(self, params: RecordingParameters) -> list[str]:
        mode = params.get("policy_mode", "openpi_remote") or "openpi_remote"
        if mode == "teleop_only":
            return []

        args = [f"{key}={value}" for key, value in POLICY_AUX_CLI_ARGS.items()]
        if mode == "openpi_remote":
            policy_host = params.get("policy_host", "127.0.0.1") or "127.0.0.1"
            policy_port = params.get("policy_port", "8000") or "8000"
            return [
                *args,
                "--policy.type=openpi_remote",
                f"--policy.host={policy_host}",
                f"--policy.port={policy_port}",
            ]
        if mode == "local_path":
            policy_path = params.get("policy_path")
            if not policy_path:
                raise ValueError("Policy Path is required when Policy Mode is Local Policy Path.")
            return [*args, f"--policy.path={policy_path}"]
        raise ValueError(f"Unknown Policy Mode: {mode}")

    def _validate_policy_params(self, params: RecordingParameters) -> None:
        self._policy_cli_args(params)

    def _cli_args(self, params: RecordingParameters) -> list[str]:
        resume_args = ["--resume=true"] if parameter_is_enabled(params, "resume") else []
        return [
            *resume_args,
            *self._dataset_cli_args(params),
            "--record_ee_pose=true",
            "--policy_action_schema=bimanual_ee_rpy",
            *self._policy_cli_args(params),
        ]

    def get_command_args(self, params: RecordingParameters) -> list[str]:
        dataset_name = params.get("dataset_name")
        if not dataset_name:
            raise ValueError("Dataset Name is required.")
        self._validate_policy_params(params)

        override = os.environ.get("EVO_HIL_RECORD_GUI_CMD", "").strip()
        if override:
            return [*shlex.split(override), *self._cli_args(params)]

        if DEFAULT_RECORD_SCRIPT.is_file():
            return [str(DEFAULT_RECORD_SCRIPT)]

        return ["lerobot-human-inloop-record", *self._cli_args(params)]


@dataclass
class OpenPIServerCommandBuilder:
    repo_root: Path

    def get_working_dir(self, params: RecordingParameters) -> Path | None:
        mode = params.get("policy_mode", "openpi_remote") or "openpi_remote"
        if mode != "openpi_remote":
            return None
        return Path(params.get("openpi_server_root", str(DEFAULT_OPENPI_SERVER_ROOT)) or DEFAULT_OPENPI_SERVER_ROOT)

    def get_command_args(self, params: RecordingParameters) -> list[str] | None:
        mode = params.get("policy_mode", "openpi_remote") or "openpi_remote"
        if mode != "openpi_remote":
            return None

        policy_dir = params.get("openpi_policy_dir")
        if not policy_dir:
            raise ValueError("OpenPI Policy Dir is required when Policy Mode is OpenPI Remote.")
        server_root = self.get_working_dir(params)
        if server_root is None:
            return None
        if not server_root.is_dir():
            raise ValueError(f"OpenPI Server Root does not exist: {server_root}")
        if not (server_root / "scripts" / "serve_policy.py").is_file():
            raise ValueError(f"OpenPI serve script not found: {server_root / 'scripts' / 'serve_policy.py'}")

        policy_port = params.get("policy_port", "8000") or "8000"

        return [
            "uv",
            "run",
            "scripts/serve_policy.py",
            "policy:checkpoint",
            f"--policy.config={OPENPI_SERVER_POLICY_CONFIG}",
            f"--policy.dir={policy_dir}",
            f"--port={policy_port}",
        ]
