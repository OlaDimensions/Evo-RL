from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterator

from lerobot.gui.hil_recording.models import (
    HardwareHealthReport,
    HealthCheckResult,
    RecordingParameters,
    StatusLevel,
    robot_camera_env_from_params,
)


@dataclass(frozen=True)
class AdbDevice:
    serial: str
    state: str


DEFAULT_ROBOT_LEFT_CAMERAS = (
    '{"wrist": {"type": "intelrealsense", "serial_number_or_name": "152122072280", '
    '"width": 640, "height":480, "fps": 30, "warmup_s": 2}}'
)
DEFAULT_ROBOT_RIGHT_CAMERAS = (
    '{"wrist": {"type": "intelrealsense", "serial_number_or_name": "008222070618", '
    '"width": 640, "height":480, "fps": 30, "warmup_s": 2}, '
    '"front": {"type": "intelrealsense", "serial_number_or_name": "213622074413", '
    '"width": 640, "height":480, "fps": 30, "warmup_s": 2}}'
)
CAN_ACTIVATE_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "can_muti_activate.sh"


def parse_adb_devices(output: str) -> list[AdbDevice]:
    devices: list[AdbDevice] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("List of devices attached"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            devices.append(AdbDevice(serial=parts[0], state=parts[1]))
    return devices


def check_adb_devices(timeout_s: float = 3.0) -> HealthCheckResult:
    if shutil.which("adb") is None:
        return HealthCheckResult("ADB Devices", StatusLevel.ERROR, "`adb` not found on PATH")

    result = subprocess.run(
        ["adb", "devices"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "adb devices failed").strip()
        return HealthCheckResult("ADB Devices", StatusLevel.ERROR, detail)

    devices = parse_adb_devices(result.stdout)
    ready = [device.serial for device in devices if device.state == "device"]
    if ready:
        return HealthCheckResult("ADB Devices", StatusLevel.OK, f"{len(ready)} device(s): {', '.join(ready)}")
    if devices:
        states = ", ".join(f"{device.serial}:{device.state}" for device in devices)
        return HealthCheckResult("ADB Devices", StatusLevel.WARNING, states)
    return HealthCheckResult("ADB Devices", StatusLevel.ERROR, "No ADB devices")


@contextlib.contextmanager
def suppress_native_stderr() -> Iterator[None]:
    """Silence native libraries that write directly to fd 2 during device probing."""

    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)


def _extract_serials_from_value(value: Any) -> list[str]:
    serials: list[str] = []
    if isinstance(value, dict):
        serial = value.get("serial_number_or_name") if value.get("type") in {None, "intelrealsense"} else None
        if serial is not None:
            serials.append(str(serial))
        for item in value.values():
            serials.extend(_extract_serials_from_value(item))
    elif isinstance(value, list):
        for item in value:
            serials.extend(_extract_serials_from_value(item))
    return serials


def _extract_gopro_paths_from_value(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        if value.get("type") == "gopro" and value.get("index_or_path") is not None:
            paths.append(str(value["index_or_path"]))
        for item in value.values():
            paths.extend(_extract_gopro_paths_from_value(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_extract_gopro_paths_from_value(item))
    return paths


def _camera_env_values(env: dict[str, str] | None = None) -> tuple[str, str]:
    env = env or {}
    left = env.get("ROBOT_LEFT_CAMERAS") or os.environ.get("ROBOT_LEFT_CAMERAS")
    right = env.get("ROBOT_RIGHT_CAMERAS") or os.environ.get("ROBOT_RIGHT_CAMERAS")
    if left is not None and right is not None:
        return left, right

    camera_profile = env.get("CAMERA_PROFILE") or os.environ.get("CAMERA_PROFILE") or "gopro"
    if camera_profile == "custom":
        return left or "{}", right or "{}"

    fallback = robot_camera_env_from_params(RecordingParameters({"camera_profile": camera_profile}))
    return (
        left if left is not None else fallback["ROBOT_LEFT_CAMERAS"],
        right if right is not None else fallback["ROBOT_RIGHT_CAMERAS"],
    )


def camera_serials_from_env(env: dict[str, str] | None = None) -> list[str]:
    values = _camera_env_values(env)
    serials: list[str] = []
    for raw in values:
        try:
            serials.extend(_extract_serials_from_value(json.loads(raw)))
        except Exception:
            continue
    return sorted(set(serials))


def gopro_paths_from_env(env: dict[str, str] | None = None) -> list[str]:
    values = _camera_env_values(env)
    paths: list[str] = []
    for raw in values:
        try:
            paths.extend(_extract_gopro_paths_from_value(json.loads(raw)))
        except Exception:
            continue
    return sorted(set(paths))


def _missing_gopro_paths(paths: list[str]) -> list[str]:
    missing = []
    for raw_path in paths:
        if raw_path.isdigit():
            continue
        if not Path(raw_path).exists():
            missing.append(raw_path)
    return missing


def _camera_id(camera: dict[str, Any]) -> str | None:
    for key in ("serial_number", "serial_number_or_name", "id", "name"):
        value = camera.get(key)
        if value is not None:
            return str(value)
    return None


def check_camera_status(env: dict[str, str] | None = None) -> HealthCheckResult:
    target_serials = camera_serials_from_env(env)
    gopro_paths = gopro_paths_from_env(env)
    missing_gopro_paths = _missing_gopro_paths(gopro_paths)
    if missing_gopro_paths:
        return HealthCheckResult("Camera", StatusLevel.ERROR, f"Missing GoPro paths: {', '.join(missing_gopro_paths)}")

    if not target_serials:
        if gopro_paths:
            return HealthCheckResult("Camera", StatusLevel.OK, f"GoPro paths configured: {', '.join(gopro_paths)}")
        return HealthCheckResult("Camera", StatusLevel.WARNING, "No cameras configured")

    try:
        from lerobot.scripts.lerobot_find_cameras import find_all_realsense_cameras
    except Exception as exc:
        return HealthCheckResult("Camera", StatusLevel.ERROR, f"Camera discovery unavailable: {exc}")

    try:
        with suppress_native_stderr():
            cameras = find_all_realsense_cameras()
    except Exception as exc:
        return HealthCheckResult("Camera", StatusLevel.ERROR, f"RealSense discovery failed: {exc}")

    seen = {camera_id for camera in cameras if (camera_id := _camera_id(camera))}
    missing = [serial for serial in target_serials if serial not in seen]
    if missing:
        detail = f"Missing RealSense: {', '.join(missing)}; seen: {', '.join(sorted(seen)) or 'none'}"
        return HealthCheckResult("Camera", StatusLevel.ERROR, detail)

    details = [f"RealSense online: {', '.join(target_serials)}"]
    if gopro_paths:
        details.append(f"GoPro paths configured: {', '.join(gopro_paths)}")
    return HealthCheckResult("Camera", StatusLevel.OK, "; ".join(details))


def _can_statuses(interfaces: tuple[str, ...]) -> tuple[int, list[str]]:
    try:
        from lerobot.scripts.lerobot_setup_can import check_interface_status
    except Exception as exc:
        return 0, [f"CAN check unavailable: {exc}"]

    statuses = []
    ok_count = 0
    for interface in interfaces:
        is_up, status, _is_fd = check_interface_status(interface)
        if is_up:
            ok_count += 1
        statuses.append(f"{interface}:{status}")
    return ok_count, statuses


def check_can_status(
    interfaces: tuple[str, ...] = ("can_left", "can_right"),
    activate_script: Path = CAN_ACTIVATE_SCRIPT,
    timeout_s: float = 30.0,
) -> HealthCheckResult:
    ok_count, statuses = _can_statuses(interfaces)
    if ok_count == len(interfaces):
        return HealthCheckResult("CAN Status", StatusLevel.OK, ", ".join(statuses))

    if not activate_script.is_file():
        return HealthCheckResult(
            "CAN Status",
            StatusLevel.ERROR,
            f"{', '.join(statuses)}; activation script missing: {activate_script}",
        )

    result = subprocess.run(
        ["bash", str(activate_script), "--ignore"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    ok_count_after, statuses_after = _can_statuses(interfaces)
    if result.returncode == 0 and ok_count_after == len(interfaces):
        return HealthCheckResult("CAN Status", StatusLevel.OK, f"Auto-activated: {', '.join(statuses_after)}")

    detail = (result.stderr or result.stdout or "").strip()
    summary = f"{', '.join(statuses_after)}; activation exit={result.returncode}"
    if detail:
        summary = f"{summary}; {detail.splitlines()[-1]}"
    return HealthCheckResult("CAN Status", StatusLevel.ERROR, summary)


def run_health_checks(env: dict[str, str] | None = None) -> HardwareHealthReport:
    return HardwareHealthReport(
        camera=check_camera_status(env),
        can=check_can_status(),
        adb=check_adb_devices(),
    )
