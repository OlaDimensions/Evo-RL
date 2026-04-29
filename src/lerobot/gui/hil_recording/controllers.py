from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from lerobot.gui.hil_recording.command_builder import OpenPIServerCommandBuilder, RecordingCommandBuilder
from lerobot.gui.hil_recording.models import (
    RecordingParameters,
    RecordingState,
    recording_parameters_to_env,
)
from lerobot.gui.hil_recording.views import MainWindow
from lerobot.gui.hil_recording.workers import HealthCheckWorker, ManagedProcessWorker, RecordingProcessWorker
from lerobot.utils.constants import HF_LEROBOT_HOME


CONTROL_KEY_PAYLOADS = {
    "success": b"s",
    "fail": b"f",
    "rerecord": b"\x1b[D",
    "intervention": b"i",
    "advance": b"\x1b[C",
    "stop": b"\x1b",
}


GUI_LOG_PATH = Path.home() / ".config" / "evo-rl" / "hil_recording_gui.log"

DATASET_CONFLICT_DELETE = "delete"
DATASET_CONFLICT_RESUME = "resume"
DATASET_CONFLICT_CANCEL = "cancel"

logger = logging.getLogger(__name__)


class RecordingStartCancelled(Exception):
    pass


def parameter_is_enabled(params: RecordingParameters, key: str) -> bool:
    return params.get(key).lower() in {"1", "true", "yes", "on"}


def dataset_path_for_params(params: RecordingParameters, base_root: Path | None = None) -> Path:
    repo_id = params.get("dataset_name")
    if not repo_id:
        raise ValueError("Dataset Name is required.")
    base_root = base_root or HF_LEROBOT_HOME
    return base_root / repo_id


def ensure_dataset_path_is_safe_to_delete(path: Path, base_root: Path | None = None) -> None:
    base_root = base_root or HF_LEROBOT_HOME
    resolved_path = path.expanduser().resolve()
    resolved_base = base_root.expanduser().resolve()
    if not resolved_path.is_relative_to(resolved_base):
        raise ValueError(f"Refusing to delete dataset outside HF_LEROBOT_HOME: {resolved_path}")


def recording_state_from_output(message: str) -> RecordingState | None:
    if "Recording episode" in message:
        return RecordingState.RUNNING
    if "Reset the environment" in message:
        return RecordingState.RESETTING
    return None


def append_gui_log_file(message: str) -> None:
    try:
        GUI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with GUI_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(message)
    except OSError:
        logger.exception("Failed to write GUI log file")


class RecordingController:
    def __init__(self, window: MainWindow, repo_root: Path):
        self.window = window
        self.repo_root = repo_root
        self.command_builder = RecordingCommandBuilder(repo_root=repo_root)
        self.openpi_server_command_builder = OpenPIServerCommandBuilder(repo_root=repo_root)
        self.recording_worker: RecordingProcessWorker | None = None
        self.openpi_server_worker: ManagedProcessWorker | None = None
        self.health_worker: HealthCheckWorker | None = None
        self._openpi_stop_requested = False
        self._recording_stop_requested = False

        self.window.start_recording_requested.connect(self.start_recording)
        self.window.success_requested.connect(lambda: self.send_control_action("success"))
        self.window.fail_requested.connect(lambda: self.send_control_action("fail"))
        self.window.rerecord_requested.connect(lambda: self.send_control_action("rerecord"))
        self.window.intervention_requested.connect(lambda: self.send_control_action("intervention"))
        self.window.advance_requested.connect(lambda: self.send_control_action("advance"))
        self.window.stop_recording_requested.connect(self.request_stop_recording)
        self.window.health_check_requested.connect(self.start_health_check)

    def get_command_args(self, params: RecordingParameters) -> list[str]:
        return self.command_builder.get_command_args(params)

    def get_openpi_server_command_args(self, params: RecordingParameters) -> list[str] | None:
        return self.openpi_server_command_builder.get_command_args(params)

    def get_openpi_server_working_dir(self, params: RecordingParameters) -> Path | None:
        return self.openpi_server_command_builder.get_working_dir(params)

    def start_recording(self) -> None:
        try:
            params = self.window.parameters()
            params = self.resolve_dataset_start_policy(params)
            command = self.get_command_args(params)
            openpi_command = self.get_openpi_server_command_args(params)
            openpi_cwd = self.get_openpi_server_working_dir(params)
        except RecordingStartCancelled:
            return
        except Exception as exc:
            QMessageBox.critical(self.window, "Invalid recording parameters", str(exc))
            return
        self.window.save_parameters()
        self._recording_stop_requested = False

        env = {
            "PYTHONPATH": f"{self.repo_root / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        }
        env.update(recording_parameters_to_env(params))
        if openpi_command is not None:
            assert openpi_cwd is not None
            self.start_openpi_server(openpi_command, openpi_cwd, env)
        self.recording_worker = RecordingProcessWorker(command=command, cwd=self.repo_root, env=env)
        self.recording_worker.process_started.connect(self.on_recording_started)
        self.recording_worker.output_received.connect(self.on_recording_output)
        self.recording_worker.process_finished.connect(self.on_recording_finished)
        self.recording_worker.process_failed.connect(self.on_recording_failed)
        self.recording_worker.finished.connect(
            lambda worker=self.recording_worker: self.on_recording_thread_finished(worker)
        )
        self.recording_worker.finished.connect(self.recording_worker.deleteLater)

        self.window.set_recording_state(RecordingState.STARTING)
        self.append_controller_log(f"$ {' '.join(command)}\n")
        self.recording_worker.start()

    def resolve_dataset_start_policy(self, params: RecordingParameters) -> RecordingParameters:
        dataset_path = dataset_path_for_params(params)
        resume = parameter_is_enabled(params, "resume")
        self.append_controller_log(
            f"[gui] checking dataset path={dataset_path} resume={str(resume).lower()} "
            f"exists={str(dataset_path.exists()).lower()}\n"
        )

        if resume:
            if dataset_path.exists():
                self.append_controller_log(f"[gui] resuming existing dataset: {dataset_path}\n")
                return params
            raise ValueError(f"Cannot resume because dataset directory does not exist:\n{dataset_path}")

        if not dataset_path.exists():
            return params

        choice = self.ask_dataset_conflict(dataset_path)
        if choice == DATASET_CONFLICT_DELETE:
            self.append_controller_log(f"[gui] dataset conflict choice=delete path={dataset_path}\n")
            self.delete_existing_dataset(dataset_path)
            return params
        if choice == DATASET_CONFLICT_RESUME:
            self.append_controller_log(f"[gui] dataset conflict choice=resume path={dataset_path}\n")
            values = dict(params.values)
            values["resume"] = "true"
            self.window.set_parameter("resume", "true")
            return RecordingParameters(values)

        self.append_controller_log(f"[gui] dataset conflict choice=cancel path={dataset_path}\n")
        self.append_controller_log(
            f"[gui] dataset already exists: {dataset_path}. Rename Dataset Name or enable resume.\n"
        )
        self.window.focus_parameter("dataset_name")
        raise RecordingStartCancelled()

    def ask_dataset_conflict(self, dataset_path: Path) -> str:
        dialog = QMessageBox(self.window)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Dataset already exists")
        dialog.setText("The local dataset directory already exists.")
        dialog.setInformativeText(
            f"{dataset_path}\n\n"
            "Choose Delete to remove it and record from scratch, Resume to append to it, "
            "or Cancel to rename the dataset."
        )
        delete_button = dialog.addButton("删除并重新录制", QMessageBox.ButtonRole.DestructiveRole)
        resume_button = dialog.addButton("添加 --resume", QMessageBox.ButtonRole.AcceptRole)
        cancel_button = dialog.addButton("取消并改名", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(resume_button)
        dialog.exec()

        clicked = dialog.clickedButton()
        if clicked == delete_button:
            return DATASET_CONFLICT_DELETE
        if clicked == resume_button:
            return DATASET_CONFLICT_RESUME
        if clicked == cancel_button:
            return DATASET_CONFLICT_CANCEL
        return DATASET_CONFLICT_CANCEL

    def delete_existing_dataset(self, dataset_path: Path) -> None:
        ensure_dataset_path_is_safe_to_delete(dataset_path)
        answer = QMessageBox.question(
            self.window,
            "Confirm dataset deletion",
            f"Delete this local dataset directory?\n\n{dataset_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.window.focus_parameter("dataset_name")
            raise RecordingStartCancelled()

        shutil.rmtree(dataset_path)
        self.append_controller_log(f"[gui] deleted existing dataset: {dataset_path}\n")

    def start_openpi_server(self, command: list[str], cwd: Path, env: dict[str, str]) -> None:
        if self.openpi_server_worker is not None:
            self.stop_openpi_server()

        self._openpi_stop_requested = False
        self.openpi_server_worker = ManagedProcessWorker(command=command, cwd=cwd, env=env)
        self.openpi_server_worker.process_started.connect(self.on_openpi_server_started)
        self.openpi_server_worker.output_received.connect(self.on_openpi_server_output)
        self.openpi_server_worker.process_finished.connect(self.on_openpi_server_finished)
        self.openpi_server_worker.process_failed.connect(self.on_openpi_server_failed)
        self.openpi_server_worker.finished.connect(
            lambda worker=self.openpi_server_worker: self.on_openpi_server_thread_finished(worker)
        )
        self.openpi_server_worker.finished.connect(self.openpi_server_worker.deleteLater)
        self.append_controller_log(f"[openpi] cwd={cwd}\n")
        self.append_controller_log(f"[openpi] $ {' '.join(command)}\n")
        self.openpi_server_worker.start()

    def stop_openpi_server(self, wait: bool = False) -> None:
        if self.openpi_server_worker is not None:
            self._openpi_stop_requested = True
            self.openpi_server_worker.terminate_process()
            if wait:
                self.wait_for_worker(self.openpi_server_worker, "openpi server")

    def request_stop_recording(self) -> None:
        self._recording_stop_requested = True
        self.append_controller_log("[gui] stop recording requested\n")
        self.stop_openpi_server()
        if self.recording_worker is None:
            return

        try:
            self.recording_worker.send_key(CONTROL_KEY_PAYLOADS["stop"])
        except Exception as exc:
            self.append_controller_log(f"[gui] graceful stop key failed: {exc}\n")
            self.force_stop_recording("stop key unavailable")
            return

        QTimer.singleShot(1500, lambda: self.force_stop_recording("stop timeout"))

    def force_stop_recording(self, reason: str) -> None:
        if self.recording_worker is not None and self.recording_worker.isRunning():
            self.append_controller_log(f"[gui] force stopping recording process: {reason}\n")
            self.recording_worker.terminate_process()

    def shutdown(self) -> None:
        if self.recording_worker is not None:
            self.recording_worker.terminate_process()
            self.wait_for_worker(self.recording_worker, "recording")
        self.stop_openpi_server(wait=True)

    def wait_for_worker(self, worker: RecordingProcessWorker | ManagedProcessWorker, name: str) -> None:
        if worker.isRunning() and not worker.wait(3000):
            self.append_controller_log(f"[gui] timed out waiting for {name} worker to stop\n")

    def append_controller_log(self, message: str) -> None:
        self.window.append_log(message)
        append_gui_log_file(message)

    def on_recording_started(self, pid: int) -> None:
        self.window.set_recording_state(RecordingState.RUNNING)
        self.append_controller_log(f"[gui] recording process started pid={pid}\n")

    def on_recording_output(self, message: str) -> None:
        self.window.append_log(message)
        state = recording_state_from_output(message)
        if state is not None:
            self.window.set_recording_state(state)

    def on_recording_finished(self, exit_code: int) -> None:
        self.window.set_recording_state(RecordingState.STOPPED)
        self.append_controller_log(f"[gui] recording process exited code={exit_code}\n")
        self.stop_openpi_server()

    def on_recording_failed(self, message: str) -> None:
        self.window.set_recording_state(RecordingState.ERROR)
        self.append_controller_log(f"[gui] recording failed: {message}\n")
        self.stop_openpi_server()

    def on_recording_thread_finished(self, worker: RecordingProcessWorker) -> None:
        if self.recording_worker is worker:
            self.recording_worker = None
            self._recording_stop_requested = False
            self.append_controller_log("[gui] recording worker cleaned up\n")

    def on_openpi_server_started(self, pid: int) -> None:
        self.append_controller_log(f"[openpi] server started pid={pid}\n")

    def on_openpi_server_output(self, message: str) -> None:
        self.window.append_log(f"[openpi] {message}")

    def on_openpi_server_finished(self, exit_code: int) -> None:
        self.append_controller_log(f"[openpi] server exited code={exit_code}\n")
        if exit_code != 0 and not getattr(self, "_openpi_stop_requested", False):
            self.window.set_recording_state(RecordingState.ERROR)
            self.append_controller_log("[gui] OpenPI server exited unexpectedly; stopping recording process.\n")
            self._recording_stop_requested = True
            self.force_stop_recording(f"openpi exited code={exit_code}")

    def on_openpi_server_failed(self, message: str) -> None:
        self.append_controller_log(f"[openpi] server failed: {message}\n")
        if not getattr(self, "_openpi_stop_requested", False):
            self.window.set_recording_state(RecordingState.ERROR)
            self.append_controller_log("[gui] OpenPI server failed; stopping recording process.\n")
            self._recording_stop_requested = True
            self.force_stop_recording("openpi server failed")

    def on_openpi_server_thread_finished(self, worker: ManagedProcessWorker) -> None:
        if self.openpi_server_worker is worker:
            self.openpi_server_worker = None
            self._openpi_stop_requested = False
            self.append_controller_log("[openpi] server worker cleaned up\n")

    def send_control_key(self, payload: bytes) -> None:
        if self.recording_worker is None:
            return
        try:
            self.recording_worker.send_key(payload)
        except Exception as exc:
            QMessageBox.warning(self.window, "Control unavailable", str(exc))

    def send_control_action(self, action: str) -> None:
        if action == "stop":
            self.request_stop_recording()
            return
        self.send_control_key(CONTROL_KEY_PAYLOADS[action])

    def start_health_check(self) -> None:
        if self.health_worker is not None:
            return
        self.window.set_health_checking()
        env = recording_parameters_to_env(self.window.parameters())
        self.health_worker = HealthCheckWorker(env=env)
        self.health_worker.health_finished.connect(self.on_health_finished)
        self.health_worker.health_failed.connect(self.on_health_failed)
        self.health_worker.finished.connect(
            lambda worker=self.health_worker: self.on_health_worker_finished(worker)
        )
        self.health_worker.finished.connect(self.health_worker.deleteLater)
        self.health_worker.start()

    def on_health_finished(self, report: object) -> None:
        self.window.set_health_report(report)

    def on_health_failed(self, message: str) -> None:
        self.append_controller_log(f"[gui] health check failed: {message}\n")

    def on_health_worker_finished(self, worker: HealthCheckWorker) -> None:
        if self.health_worker is worker:
            self.health_worker = None
