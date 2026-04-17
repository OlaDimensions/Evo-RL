#!/usr/bin/env python
"""Read and print raw Piper SDK end-effector pose feedback."""

from __future__ import annotations

import argparse
import math
import pprint
import time
from typing import Any


POSE_FIELDS = ("X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis")


def public_attrs(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of SDK message objects to printable dicts."""
    result: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:  # noqa: BLE001 - SDK objects may raise on access.
            result[name] = f"<unreadable: {exc}>"
            continue
        if callable(value):
            continue
        result[name] = value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly read Piper SDK raw end-effector pose feedback.",
    )
    parser.add_argument("--can", default="can_right", help="CAN interface name, e.g. can0/can_left/can_right.")
    parser.add_argument("--samples", type=int, default=1, help="Number of pose samples to print.")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between samples.")
    parser.add_argument("--startup-sleep", type=float, default=0.2, help="Seconds to wait after ConnectPort.")
    parser.add_argument("--judge-flag", action="store_true", help="Pass judge_flag=True to the Piper SDK.")
    parser.add_argument("--no-auto-init", action="store_true", help="Pass can_auto_init=False to the Piper SDK.")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "SILENT"),
        help="Piper SDK log level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from piper_sdk import C_PiperInterface_V2, LogLevel
    except ModuleNotFoundError as exc:
        raise SystemExit("Could not import piper_sdk. Run `pip install -e .` in this repo first.") from exc

    logger_level = getattr(LogLevel, args.log_level)
    arm = C_PiperInterface_V2(
        can_name=args.can,
        judge_flag=args.judge_flag,
        can_auto_init=not args.no_auto_init,
        logger_level=logger_level,
    )

    print(f"Connecting Piper SDK on CAN interface: {args.can}")
    arm.ConnectPort(start_thread=True)
    try:
        if args.startup_sleep > 0:
            time.sleep(args.startup_sleep)

        for sample_idx in range(max(1, args.samples)):
            pose_msg = arm.GetArmEndPoseMsgs()
            end_pose = getattr(pose_msg, "end_pose", None)

            print(f"\n=== sample {sample_idx + 1} ===")
            print("raw GetArmEndPoseMsgs():")
            pprint.pp(public_attrs(pose_msg), sort_dicts=False)

            if end_pose is None:
                print("end_pose: <missing>")
            else:
                print("\nraw end_pose object:")
                pprint.pp(public_attrs(end_pose), sort_dicts=False)

                print("\nraw end_pose fields:")
                raw_pose = {field: getattr(end_pose, field, None) for field in POSE_FIELDS}
                pprint.pp(raw_pose, sort_dicts=False)

                print("\ndecoded reference:")
                decoded = {
                    "x_m": float(raw_pose["X_axis"]) * 1e-6,
                    "y_m": float(raw_pose["Y_axis"]) * 1e-6,
                    "z_m": float(raw_pose["Z_axis"]) * 1e-6,
                    "roll_rad": math.radians(float(raw_pose["RX_axis"]) * 1e-3),
                    "pitch_rad": math.radians(float(raw_pose["RY_axis"]) * 1e-3),
                    "yaw_rad": math.radians(float(raw_pose["RZ_axis"]) * 1e-3),
                }
                pprint.pp(decoded, sort_dicts=False)

            if sample_idx + 1 < args.samples:
                time.sleep(max(0.0, args.interval))
    finally:
        print("\nDisconnecting Piper SDK.")
        arm.DisconnectPort()


if __name__ == "__main__":
    main()
