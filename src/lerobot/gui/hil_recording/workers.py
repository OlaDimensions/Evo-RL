from __future__ import annotations

import os
import pty
import select
import signal
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from lerobot.gui.hil_recording.health import run_health_checks
from lerobot.gui.hil_recording.models import HardwareHealthReport


class RecordingProcessWorker(QThread):
    process_started = Signal(int)
    output_received = Signal(str)
    process_finished = Signal(int)
    process_failed = Signal(str)

    def __init__(self, command: list[str], cwd: Path, env: dict[str, str] | None = None):
        super().__init__()
        self.command = command
        self.cwd = cwd
        self.env = env
        self._master_fd: int | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def run(self) -> None:
        master_fd = slave_fd = None
        try:
            master_fd, slave_fd = pty.openpty()
            self._master_fd = master_fd
            proc_env = os.environ.copy()
            if self.env:
                proc_env.update(self.env)
            proc_env["LEROBOT_FORCE_TTY_KEYBOARD"] = "1"
            proc_env.setdefault("PYTHONUNBUFFERED", "1")

            self._process = subprocess.Popen(
                self.command,
                cwd=str(self.cwd),
                env=proc_env,
                stdin=slave_fd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            os.close(slave_fd)
            slave_fd = None
            self.process_started.emit(self._process.pid)

            assert self._process.stdout is not None
            while True:
                ready, _, _ = select.select([self._process.stdout], [], [], 0.1)
                if ready:
                    chunk = self._process.stdout.readline()
                    if chunk:
                        self.output_received.emit(chunk.decode(errors="replace"))

                exit_code = self._process.poll()
                if exit_code is not None:
                    remaining = self._process.stdout.read()
                    if remaining:
                        self.output_received.emit(remaining.decode(errors="replace"))
                    self.process_finished.emit(exit_code)
                    return
        except Exception as exc:
            self.process_failed.emit(str(exc))
        finally:
            if slave_fd is not None:
                os.close(slave_fd)
            if master_fd is not None:
                os.close(master_fd)
            self._master_fd = None

    def send_key(self, payload: bytes) -> None:
        if self._master_fd is None:
            raise RuntimeError("Recording process is not ready for keyboard input.")
        os.write(self._master_fd, payload)

    def terminate_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()


class ManagedProcessWorker(QThread):
    process_started = Signal(int)
    output_received = Signal(str)
    process_finished = Signal(int)
    process_failed = Signal(str)

    def __init__(self, command: list[str], cwd: Path, env: dict[str, str] | None = None):
        super().__init__()
        self.command = command
        self.cwd = cwd
        self.env = env
        self._process: subprocess.Popen[bytes] | None = None

    def run(self) -> None:
        try:
            proc_env = os.environ.copy()
            if self.env:
                proc_env.update(self.env)
            proc_env.setdefault("PYTHONUNBUFFERED", "1")

            self._process = subprocess.Popen(
                self.command,
                cwd=str(self.cwd),
                env=proc_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
            self.process_started.emit(self._process.pid)

            assert self._process.stdout is not None
            for chunk in iter(self._process.stdout.readline, b""):
                if chunk:
                    self.output_received.emit(chunk.decode(errors="replace"))

            exit_code = self._process.wait()
            self.process_finished.emit(exit_code)
        except Exception as exc:
            self.process_failed.emit(str(exc))

    def terminate_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except Exception:
                self._process.terminate()


class HealthCheckWorker(QThread):
    health_finished = Signal(object)
    health_failed = Signal(str)

    def __init__(self, env: dict[str, str] | None = None):
        super().__init__()
        self.env = env

    def run(self) -> None:
        try:
            report: HardwareHealthReport = run_health_checks(env=self.env)
            self.health_finished.emit(report)
        except Exception as exc:
            self.health_failed.emit(str(exc))
