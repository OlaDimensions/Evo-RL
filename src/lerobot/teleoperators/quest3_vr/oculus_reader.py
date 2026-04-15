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
import subprocess
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from .buttons_parser import parse_buttons


class OculusReader:
    """Quest3/Oculus input reader implemented entirely inside Evo-RL.

    This reader follows the original quest3VR_ws data format: it listens to the
    Quest3 companion app logcat stream, extracts the tagged payload, parses the
    controller transforms and buttons, and exposes the latest sample through
    ``get_transformations_and_buttons``.

    The implementation is intentionally ROS-free.
    """

    def __init__(
        self,
        ip_address: str | None = None,
        port: int = 5555,
        apk_name: str = "com.rail.oculus.teleop",
        print_fps: bool = False,
        run: bool = True,
        tag: str = "wE9ryARX",
    ):
        self.running = False
        self.last_transforms: dict[str, np.ndarray] = {}
        self.last_buttons: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.tag = tag
        self.ip_address = ip_address
        self.port = port
        self.apk_name = apk_name
        self.print_fps = print_fps
        self._fps_count = 0
        self._fps_t0 = time.monotonic()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._device_serial = self._resolve_device_serial()
        self._app_started = False
        logger.info("[VR_HEALTH] OculusReader initialized device_serial=%s ip=%s", self._device_serial, self.ip_address)
        if run:
            self.run()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    def _resolve_device_serial(self) -> str:
        if self.ip_address:
            return f"{self.ip_address}:{self.port}"
        return self._pick_usb_device_serial()

    @staticmethod
    def _run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, check=check, capture_output=True, text=True)

    def _pick_usb_device_serial(self) -> str:
        try:
            out = self._run_cmd(["adb", "devices"]).stdout.splitlines()
        except FileNotFoundError as exc:
            raise RuntimeError("`adb` was not found on PATH. Install Android platform-tools first.") from exc
        for line in out:
            line = line.strip()
            if not line or line.startswith("List of devices attached"):
                continue
            if "device" in line and "." not in line.split()[0]:
                return line.split()[0]
        raise RuntimeError("No USB Quest device found. Run `adb devices` and ensure the headset is connected.")

    def _ensure_network_mode(self) -> None:
        if not self.ip_address:
            return
        try:
            result = self._run_cmd(["adb", "connect", self._device_serial])
        except FileNotFoundError as exc:
            raise RuntimeError("`adb` was not found on PATH. Install Android platform-tools first.") from exc
        if result.returncode != 0:
            raise RuntimeError(f"Failed to connect to Quest3 via adb: {result.stderr.strip() or result.stdout.strip()}")

    def _ensure_app_started(self) -> None:
        if self._app_started:
            return
        start_cmd = [
            "adb",
            "-s",
            self._device_serial,
            "shell",
            "am",
            "start",
            "-n",
            f"{self.apk_name}/com.rail.oculus.teleop.MainActivity",
            "-a",
            "android.intent.action.MAIN",
            "-c",
            "android.intent.category.LAUNCHER",
        ]
        logger.info("[VR_HEALTH] launching Quest3 companion app command=%s", " ".join(start_cmd))
        try:
            result = self._run_cmd(start_cmd)
        except FileNotFoundError as exc:
            raise RuntimeError("`adb` was not found on PATH. Install Android platform-tools first.") from exc
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start Quest3 companion app: {result.stderr.strip() or result.stdout.strip()}")
        self._app_started = True

    def _extract_data(self, line: str) -> str:
        if self.tag not in line:
            return ""
        try:
            return line.split(self.tag + ": ", 1)[1]
        except IndexError:
            return ""

    @staticmethod
    def _parse_transform_string(transform_string: str) -> np.ndarray | None:
        values = [v for v in transform_string.split(" ") if v]
        if len(values) != 16:
            return None
        mat = np.empty((4, 4), dtype=np.float64)
        idx = 0
        for r in range(4):
            for c in range(4):
                mat[r, c] = float(values[idx])
                idx += 1
        return mat

    @staticmethod
    def process_data(string: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        try:
            transforms_string, buttons_string = string.split("&", 1)
        except ValueError:
            return {}, {}

        split_transform_strings = transforms_string.split("|")
        transforms: dict[str, np.ndarray] = {}
        for pair_string in split_transform_strings:
            pair = pair_string.split(":", 1)
            if len(pair) != 2:
                continue
            left_right_char = pair[0]
            transform = OculusReader._parse_transform_string(pair[1])
            if transform is not None:
                transforms[left_right_char] = transform

        buttons = parse_buttons(buttons_string)
        return transforms, buttons

    def run(self) -> None:
        if self.running:
            return
        self._ensure_network_mode()
        self._ensure_app_started()
        logger.info("[VR_HEALTH] starting Quest3 reader stream serial=%s", self._device_serial)
        self.running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        logger.info("[VR_HEALTH] stopping Quest3 reader stream")
        self.running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._thread is not None and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def _mark_disconnect(self, message: str) -> None:
        logger.warning("[VR_HEALTH] %s", message)
        with self._lock:
            self.last_transforms = {}
            self.last_buttons = {}

    def _restart_reader(self) -> None:
        self.stop()
        time.sleep(0.2)
        self._app_started = False
        self.run()

    def install(self, APK_path: str | None = None, verbose: bool = True, reinstall: bool = False) -> None:  # noqa: ARG002
        raise NotImplementedError("APK install/uninstall is intentionally not bundled in Evo-RL.")

    def uninstall(self, verbose: bool = True) -> None:  # noqa: ARG002
        raise NotImplementedError("APK install/uninstall is intentionally not bundled in Evo-RL.")

    def get_transformations_and_buttons(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        with self._lock:
            return self.last_transforms, self.last_buttons

    def _reader_loop(self) -> None:
        cmd = ["adb", "-s", self._device_serial]
        cmd += ["logcat", "-T", "0", "-s", self.tag]
        logger.info("[VR_HEALTH] launching logcat command=%s", " ".join(cmd))
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except FileNotFoundError as exc:
            self._mark_disconnect("adb was not found on PATH while starting logcat")
            raise RuntimeError("`adb` was not found on PATH. Install Android platform-tools first.") from exc
        except Exception as exc:
            self._mark_disconnect(f"failed to start logcat stream: {exc}")
            raise
        if self._proc.stdout is None:
            self._mark_disconnect("failed to open adb logcat stream")
            raise RuntimeError("Failed to open adb logcat stream.")
        try:
            while self.running:
                raw_line = self._proc.stdout.readline()
                if raw_line == "":
                    if self._proc.poll() is not None:
                        self._mark_disconnect(
                            f"logcat stream exited unexpectedly with code {self._proc.returncode}"
                        )
                        self._restart_reader()
                        return
                    time.sleep(0.02)
                    continue
                line = raw_line.strip()
                # logger.info("[VR_DEBUG_RAW] line=%s", line)
                data = self._extract_data(line)
                if not data:
                    continue
                try:
                    transforms, buttons = OculusReader.process_data(data)
                except Exception as exc:
                    logger.warning("[VR_HEALTH] failed to parse Quest3 payload: %s raw_line=%s", exc, line)
                    continue
                # if not transforms:
                #     logger.info("[VR_DEBUG] parsed empty transforms raw_data=%s", data)
                # else:
                #     logger.info("[VR_DEBUG] transform keys=%s raw_data=%s", list(transforms.keys()), data)
                # if not buttons:
                #     logger.info("[VR_DEBUG] parsed empty buttons raw_data=%s", data)
                # else:
                #     logger.info("[VR_DEBUG] button keys=%s raw_data=%s", list(buttons.keys()), data)
                with self._lock:
                    self.last_transforms, self.last_buttons = transforms, buttons
                if self.print_fps:
                    self._fps_count += 1
                    now = time.monotonic()
                    if now - self._fps_t0 >= 1.0:
                        logger.info("[VR_HEALTH] OculusReader FPS=%.1f", self._fps_count / (now - self._fps_t0))
                        self._fps_count = 0
                        self._fps_t0 = now
        finally:
            try:
                if self._proc is not None and self._proc.stdout is not None:
                    self._proc.stdout.close()
            except Exception:
                pass
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._proc = None


def main() -> None:
    reader = OculusReader()
    while True:
        time.sleep(0.3)
        _ = reader.get_transformations_and_buttons()


if __name__ == "__main__":
    main()
