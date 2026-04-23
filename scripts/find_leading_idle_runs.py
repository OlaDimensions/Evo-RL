#!/usr/bin/env python3
"""Find leading idle/wait frames in a local LeRobot v3 dataset.

The detector is intended for EE-pose datasets where SDK feedback may jitter
slightly while no useful command is being executed. It uses policy action intent
when available and falls back to dynamically-thresholded EE feedback motion.

The generated JSONL is compatible with scripts/drop_static_runs.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from find_static_runs import (
    DEFAULT_DATASET_DIR,
    attach_previews,
    load_info,
    read_data_table,
    scalar_column_to_numpy,
    vector_column_to_numpy,
    write_inspection_html,
    write_jsonl,
    write_summary,
)

POLICY_ACTION_KEY = "complementary_info.policy_action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find per-episode leading idle frames in a local LeRobot v3 dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--policy-action-key", default=POLICY_ACTION_KEY)
    parser.add_argument("--output-dir", type=Path, default=Path("leading_idle_report"))
    parser.add_argument(
        "--mode",
        choices=("conservative", "balanced"),
        default="conservative",
        help="Conservative stops at the first possible motion; balanced uses consecutive active frames.",
    )
    parser.add_argument("--pos-delta-threshold", type=float, default=1e-3)
    parser.add_argument("--rot-delta-threshold-deg", type=float, default=1.0)
    parser.add_argument("--gripper-delta-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--max-trim-seconds",
        type=float,
        default=3.0,
        help="Hard cap for automatic leading trim duration. Use 0 or negative to disable.",
    )
    parser.add_argument(
        "--policy-abs-threshold",
        type=float,
        default=1e-8,
        help="Diagnostic threshold for counting nonzero absolute policy-action frames.",
    )
    parser.add_argument(
        "--policy-mad-multiplier",
        type=float,
        default=5.0,
        help="MAD multiplier used to detect meaningful policy-action deltas.",
    )
    parser.add_argument(
        "--consecutive-active-frames",
        type=int,
        default=4,
        help="First useful motion/action must be active for this many consecutive frames.",
    )
    parser.add_argument(
        "--pre-roll-frames",
        type=int,
        default=15,
        help="Keep this many frames before the detected useful start.",
    )
    parser.add_argument(
        "--min-idle-seconds",
        type=float,
        default=0.5,
        help="Only report trims at least this long.",
    )
    parser.add_argument(
        "--min-trim-frames",
        type=int,
        default=3,
        help="Only report trims with at least this many leading frames.",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=2.0,
        help="Initial episode window used to estimate EE feedback jitter.",
    )
    parser.add_argument("--mad-multiplier", type=float, default=5.0)
    parser.add_argument(
        "--max-trim-ratio-warning",
        type=float,
        default=0.2,
        help="Mark episodes whose proposed trim exceeds this fraction of episode length.",
    )
    parser.add_argument("--preview-camera", default="observation.images.right_front")
    parser.add_argument("--preview-count", type=int, default=30)
    parser.add_argument("--preview-pad-frames", type=int, default=30)
    parser.add_argument("--preview-seed", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def available_columns(dataset_dir: Path) -> set[str]:
    columns: set[str] = set()
    for path in sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet")):
        columns.update(pq.ParquetFile(path).schema_arrow.names)
    if not columns:
        raise FileNotFoundError(f"No parquet data files found under {dataset_dir / 'data'}")
    return columns


def feature_names(info: dict[str, Any], key: str, dim: int) -> list[str]:
    names = info.get("features", {}).get(key, {}).get("names")
    if names is None:
        return [str(i) for i in range(dim)]
    return [str(name) for name in names]


def name_groups(names: list[str]) -> list[dict[str, int]]:
    groups: dict[str, dict[str, int]] = {}
    for i, name in enumerate(names):
        if "." not in name:
            continue
        prefix, suffix = name.rsplit(".", 1)
        groups.setdefault(prefix, {})[suffix] = i
    return list(groups.values())


def angle_from_quat_xyzw(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q0_norm = np.linalg.norm(q0, axis=1)
    q1_norm = np.linalg.norm(q1, axis=1)
    valid = (q0_norm > 0.0) & (q1_norm > 0.0)
    dots = np.ones(len(q0), dtype=np.float64)
    dots[valid] = np.sum(q0[valid] * q1[valid], axis=1) / (q0_norm[valid] * q1_norm[valid])
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def vector_motion_score(
    values: np.ndarray,
    names: list[str],
    *,
    pos_threshold: float,
    rot_threshold_rad: float,
    gripper_threshold: float,
    include_gripper: bool = True,
) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(0, dtype=np.float64)

    diffs = np.diff(values, axis=0)
    score = np.zeros(len(diffs), dtype=np.float64)
    for group in name_groups(names):
        if {"x", "y", "z"} <= set(group):
            xyz_delta = np.linalg.norm(diffs[:, [group["x"], group["y"], group["z"]]], axis=1)
            score = np.maximum(score, xyz_delta / max(pos_threshold, 1e-12))

        if {"qx", "qy", "qz", "qw"} <= set(group):
            q_cols = [group["qx"], group["qy"], group["qz"], group["qw"]]
            angle = angle_from_quat_xyzw(values[:-1, q_cols], values[1:, q_cols])
            score = np.maximum(score, angle / max(rot_threshold_rad, 1e-12))
        elif {"roll", "pitch", "yaw"} <= set(group):
            rpy_cols = [group["roll"], group["pitch"], group["yaw"]]
            angle_delta = np.max(np.abs(np.diff(np.unwrap(values[:, rpy_cols], axis=0), axis=0)), axis=1)
            score = np.maximum(score, angle_delta / max(rot_threshold_rad, 1e-12))
        elif {"rx", "ry", "rz"} <= set(group):
            rpy_cols = [group["rx"], group["ry"], group["rz"]]
            angle_delta = np.max(np.abs(np.diff(np.unwrap(values[:, rpy_cols], axis=0), axis=0)), axis=1)
            score = np.maximum(score, angle_delta / max(rot_threshold_rad, 1e-12))

        if include_gripper and "pos" in group and len(group) == 1:
            score = np.maximum(score, np.abs(diffs[:, group["pos"]]) / max(gripper_threshold, 1e-12))

    if not np.any(score) and diffs.shape[1] > 0:
        score = np.max(np.abs(diffs), axis=1)
    return score


def robust_motion_threshold(score: np.ndarray, baseline_frames: int, mad_multiplier: float) -> float:
    if len(score) == 0:
        return float("inf")
    baseline = score[: max(1, min(len(score), baseline_frames))]
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    return max(1.0, median + mad_multiplier * 1.4826 * mad)


def first_consecutive_true(mask: np.ndarray, consecutive: int) -> int | None:
    if consecutive <= 1:
        hits = np.flatnonzero(mask)
        return int(hits[0]) if len(hits) else None

    run = 0
    for i, value in enumerate(mask):
        if bool(value):
            run += 1
            if run >= consecutive:
                return i - consecutive + 1
        else:
            run = 0
    return None


def sort_by_episode_frame(table: pa.Table, columns: list[str]) -> dict[str, np.ndarray]:
    episode_indices = scalar_column_to_numpy(table, "episode_index", np.int64)
    frame_indices = scalar_column_to_numpy(table, "frame_index", np.int64)
    order = np.lexsort((frame_indices, episode_indices))

    arrays: dict[str, np.ndarray] = {
        "episode_index": episode_indices[order],
        "frame_index": frame_indices[order],
        "timestamp": scalar_column_to_numpy(table, "timestamp", np.float64)[order],
        "index": scalar_column_to_numpy(table, "index", np.int64)[order],
    }
    for column in columns:
        arrays[column] = vector_column_to_numpy(table, column)[order]
    return arrays


def frame_mask_from_pair_mask(pair_mask: np.ndarray, num_frames: int) -> np.ndarray:
    mask = np.zeros(num_frames, dtype=bool)
    if num_frames > 1:
        mask[1:] = pair_mask
    return mask


def first_true(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def first_true_run(mask: np.ndarray, consecutive: int) -> int | None:
    if consecutive <= 1:
        return first_true(mask)
    return first_consecutive_true(mask, consecutive)


def leading_trim_decision(
    *,
    ee_score: np.ndarray,
    ee_threshold: float,
    policy_score: np.ndarray | None,
    policy_threshold: float | None,
    num_frames: int,
    fps: float,
    mode: str,
    consecutive_active_frames: int,
    pre_roll_frames: int,
    max_trim_seconds: float,
) -> tuple[int | None, int, str]:
    if num_frames < 2:
        return None, 0, "too_short"

    max_trim_frames = int(round(max_trim_seconds * fps)) if max_trim_seconds > 0 else num_frames
    max_trim_frames = max(0, min(max_trim_frames, num_frames - 1))

    if mode == "balanced":
        ee_active = frame_mask_from_pair_mask(ee_score > ee_threshold, num_frames)
        if policy_score is not None and policy_threshold is not None:
            policy_active = frame_mask_from_pair_mask(policy_score > policy_threshold, num_frames)
        else:
            policy_active = np.zeros(num_frames, dtype=bool)
        active = ee_active | policy_active
        useful_start = first_consecutive_true(active, consecutive_active_frames)
        if useful_start is None:
            if max_trim_frames > 0:
                return max_trim_frames, max_trim_frames, "max_trim_seconds"
            return None, 0, "no_motion_found"
        trim_end = max(0, min(useful_start - pre_roll_frames, max_trim_frames))
        reason = "policy_or_ee_motion"
        if trim_end == max_trim_frames and useful_start - pre_roll_frames > max_trim_frames:
            reason = "max_trim_seconds"
        return useful_start, trim_end, reason

    possible_ee_motion = frame_mask_from_pair_mask(ee_score > (0.5 * ee_threshold), num_frames)
    possible_policy_motion = np.zeros(num_frames, dtype=bool)
    if policy_score is not None and policy_threshold is not None:
        possible_policy_motion = frame_mask_from_pair_mask(
            policy_score > (0.5 * policy_threshold), num_frames
        )

    first_ee = first_true_run(possible_ee_motion, consecutive_active_frames)
    first_policy = first_true(possible_policy_motion)
    candidates = [
        (idx, reason)
        for idx, reason in ((first_policy, "policy_delta"), (first_ee, "ee_motion"))
        if idx is not None
    ]
    if not candidates:
        if max_trim_frames > 0:
            return max_trim_frames, max_trim_frames, "max_trim_seconds"
        return None, 0, "no_motion_found"

    useful_start, reason = min(candidates, key=lambda item: item[0])
    trim_end = max(0, min(useful_start - pre_roll_frames, max_trim_frames))
    if trim_end == max_trim_frames and useful_start - pre_roll_frames > max_trim_frames:
        reason = "max_trim_seconds"
    return useful_start, trim_end, reason


def detect_leading_idle_runs(
    table: pa.Table,
    *,
    info: dict[str, Any],
    state_key: str,
    action_key: str,
    policy_action_key: str | None,
    fps: float,
    pos_threshold: float,
    rot_threshold_rad: float,
    gripper_threshold: float,
    policy_abs_threshold: float,
    policy_mad_multiplier: float,
    mode: str,
    consecutive_active_frames: int,
    pre_roll_frames: int,
    min_trim_frames: int,
    min_idle_seconds: float,
    max_trim_seconds: float,
    baseline_seconds: float,
    mad_multiplier: float,
    max_trim_ratio_warning: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_columns = [state_key, action_key]
    if policy_action_key is not None:
        data_columns.append(policy_action_key)
    data = sort_by_episode_frame(table, data_columns)

    state_names = feature_names(info, state_key, data[state_key].shape[1])
    action_names = feature_names(info, action_key, data[action_key].shape[1])
    policy_names = (
        feature_names(info, policy_action_key, data[policy_action_key].shape[1])
        if policy_action_key is not None
        else []
    )

    runs: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    baseline_frames = max(1, int(round(baseline_seconds * fps)))

    for episode_index in np.unique(data["episode_index"]):
        ep_positions = np.flatnonzero(data["episode_index"] == episode_index)
        num_frames = len(ep_positions)
        if num_frames < max(2, consecutive_active_frames):
            continue

        ep_state = data[state_key][ep_positions]
        ep_action = data[action_key][ep_positions]
        state_score = vector_motion_score(
            ep_state,
            state_names,
            pos_threshold=pos_threshold,
            rot_threshold_rad=rot_threshold_rad,
            gripper_threshold=gripper_threshold,
            include_gripper=False,
        )
        action_score = vector_motion_score(
            ep_action,
            action_names,
            pos_threshold=pos_threshold,
            rot_threshold_rad=rot_threshold_rad,
            gripper_threshold=gripper_threshold,
            include_gripper=False,
        )
        ee_score = np.maximum(state_score, action_score)
        ee_threshold = robust_motion_threshold(ee_score, baseline_frames, mad_multiplier)
        max_intent_score = 0.0
        policy_motion_threshold = None
        policy_active_frames = 0
        policy_nonzero_frames = 0
        policy_score = None
        if policy_action_key is not None:
            ep_policy = data[policy_action_key][ep_positions]
            policy_nonzero = np.max(np.abs(ep_policy), axis=1) > policy_abs_threshold
            policy_nonzero_frames = int(np.count_nonzero(policy_nonzero))
            policy_score = vector_motion_score(
                ep_policy,
                policy_names,
                pos_threshold=pos_threshold,
                rot_threshold_rad=rot_threshold_rad,
                gripper_threshold=gripper_threshold,
                include_gripper=True,
            )
            max_intent_score = float(np.max(policy_score)) if len(policy_score) else 0.0
            policy_motion_threshold = robust_motion_threshold(
                policy_score, baseline_frames, policy_mad_multiplier
            )
            policy_active_frames = int(np.count_nonzero(policy_score > policy_motion_threshold))

        useful_start, trim_end_exclusive, trim_stop_reason = leading_trim_decision(
            ee_score=ee_score,
            ee_threshold=ee_threshold,
            policy_score=policy_score,
            policy_threshold=policy_motion_threshold,
            num_frames=num_frames,
            fps=fps,
            mode=mode,
            consecutive_active_frames=consecutive_active_frames,
            pre_roll_frames=pre_roll_frames,
            max_trim_seconds=max_trim_seconds,
        )
        min_idle_frames = max(min_trim_frames, int(round(min_idle_seconds * fps)))
        if trim_end_exclusive < min_idle_frames:
            trim_end_exclusive = 0

        row = {
            "episode_index": int(episode_index),
            "num_frames": int(num_frames),
            "useful_start_frame_index": None if useful_start is None else int(useful_start),
            "trim_end_exclusive": int(trim_end_exclusive),
            "trim_frames": int(trim_end_exclusive),
            "trim_duration_s": float(trim_end_exclusive / fps),
            "trim_ratio": float(trim_end_exclusive / num_frames),
            "ee_motion_threshold": float(ee_threshold),
            "max_ee_motion_score": float(np.max(ee_score)) if len(ee_score) else 0.0,
            "max_policy_motion_score": max_intent_score,
            "policy_motion_threshold": policy_motion_threshold,
            "policy_active_frames": policy_active_frames,
            "policy_nonzero_frames": policy_nonzero_frames,
            "trim_stop_reason": trim_stop_reason,
            "needs_review": bool(num_frames > 0 and trim_end_exclusive / num_frames > max_trim_ratio_warning),
        }
        episode_rows.append(row)

        if trim_end_exclusive < min_trim_frames:
            continue

        run_positions = ep_positions[:trim_end_exclusive]
        frame_indices = data["frame_index"][run_positions].astype(int).tolist()
        global_indices = data["index"][run_positions].astype(int).tolist()
        run = {
            "episode_index": int(episode_index),
            "start_frame_index": int(frame_indices[0]),
            "end_frame_index": int(frame_indices[-1]),
            "frame_indices": frame_indices,
            "global_indices": global_indices,
            "num_static_frames": int(trim_end_exclusive),
            "duration_s": float(trim_end_exclusive / fps),
            "start_timestamp": float(data["timestamp"][run_positions[0]]),
            "end_timestamp": float(data["timestamp"][run_positions[-1]]),
            "anchor_frame_index": int(data["frame_index"][ep_positions[trim_end_exclusive]]),
            "anchor_global_index": int(data["index"][ep_positions[trim_end_exclusive]]),
            "max_delta": float(row["max_ee_motion_score"]),
            "detector": "leading_idle",
            "useful_start_frame_index": int(useful_start),
            "pre_roll_frames": int(pre_roll_frames),
            "ee_motion_threshold": float(ee_threshold),
            "max_policy_motion_score": max_intent_score,
            "policy_motion_threshold": policy_motion_threshold,
            "policy_active_frames": policy_active_frames,
            "trim_stop_reason": trim_stop_reason,
            "needs_review": bool(row["needs_review"]),
        }
        runs.append(run)

    runs.sort(key=lambda x: (x["episode_index"], x["start_frame_index"]))
    episode_rows.sort(key=lambda x: x["episode_index"])
    return runs, episode_rows


def write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_index",
        "num_frames",
        "useful_start_frame_index",
        "trim_end_exclusive",
        "trim_frames",
        "trim_duration_s",
        "trim_ratio",
        "ee_motion_threshold",
        "max_ee_motion_score",
        "max_policy_motion_score",
        "policy_motion_threshold",
        "policy_active_frames",
        "policy_nonzero_frames",
        "trim_stop_reason",
        "needs_review",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    info: dict[str, Any],
    runs: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    policy_action_key: str | None,
) -> dict[str, Any]:
    total_trim = sum(int(row["trim_frames"]) for row in episode_rows)
    total_frames = int(info.get("total_frames", 0))
    review_episodes = [int(row["episode_index"]) for row in episode_rows if row["needs_review"]]
    return {
        "dataset": {
            "codebase_version": info.get("codebase_version"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": total_frames,
        },
        "detection": {
            "state_key": args.state_key,
            "action_key": args.action_key,
            "policy_action_key": policy_action_key,
            "mode": args.mode,
            "pos_delta_threshold": args.pos_delta_threshold,
            "rot_delta_threshold_deg": args.rot_delta_threshold_deg,
            "gripper_delta_threshold": args.gripper_delta_threshold,
            "max_trim_seconds": args.max_trim_seconds,
            "policy_abs_threshold": args.policy_abs_threshold,
            "policy_mad_multiplier": args.policy_mad_multiplier,
            "consecutive_active_frames": args.consecutive_active_frames,
            "pre_roll_frames": args.pre_roll_frames,
            "min_trim_frames": args.min_trim_frames,
            "min_idle_seconds": args.min_idle_seconds,
            "baseline_seconds": args.baseline_seconds,
            "mad_multiplier": args.mad_multiplier,
        },
        "result": {
            "num_runs": len(runs),
            "num_trimmed_frames": total_trim,
            "trimmed_frame_ratio": total_trim / total_frames if total_frames else None,
            "review_episodes": review_episodes,
            "longest_run": max(runs, key=lambda x: x["num_static_frames"], default=None),
        },
    }


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = load_info(dataset_dir)
    columns = available_columns(dataset_dir)
    policy_action_key = args.policy_action_key if args.policy_action_key in columns else None
    required_columns = ["episode_index", "frame_index", "timestamp", "index", args.state_key, args.action_key]
    if policy_action_key is not None:
        required_columns.append(policy_action_key)
    table = read_data_table(dataset_dir, required_columns)

    runs, episode_rows = detect_leading_idle_runs(
        table,
        info=info,
        state_key=args.state_key,
        action_key=args.action_key,
        policy_action_key=policy_action_key,
        fps=float(info["fps"]),
        pos_threshold=args.pos_delta_threshold,
        rot_threshold_rad=np.deg2rad(args.rot_delta_threshold_deg),
        gripper_threshold=args.gripper_delta_threshold,
        policy_abs_threshold=args.policy_abs_threshold,
        policy_mad_multiplier=args.policy_mad_multiplier,
        mode=args.mode,
        consecutive_active_frames=args.consecutive_active_frames,
        pre_roll_frames=args.pre_roll_frames,
        min_trim_frames=args.min_trim_frames,
        min_idle_seconds=args.min_idle_seconds,
        max_trim_seconds=args.max_trim_seconds,
        baseline_seconds=args.baseline_seconds,
        mad_multiplier=args.mad_multiplier,
        max_trim_ratio_warning=args.max_trim_ratio_warning,
    )

    write_jsonl(output_dir / "leading_idle_runs.jsonl", runs)
    write_episode_csv(output_dir / "episodes.csv", episode_rows)

    preview_runs = runs
    if not args.no_preview:
        preview_runs = attach_previews(
            dataset_dir=dataset_dir,
            camera_key=args.preview_camera,
            runs=runs,
            fps=float(info["fps"]),
            preview_count=args.preview_count,
            pad_frames=args.preview_pad_frames,
            seed=args.preview_seed,
            inspection_dir=output_dir,
        )
        write_jsonl(output_dir / "leading_idle_runs_with_previews.jsonl", preview_runs)

    summary = build_summary(
        info=info,
        runs=runs,
        episode_rows=episode_rows,
        args=args,
        policy_action_key=policy_action_key,
    )
    write_summary(output_dir / "summary.json", summary)
    if not args.no_preview:
        write_inspection_html(
            path=output_dir / "index.html",
            runs=preview_runs,
            summary={
                "result": {
                    "num_runs": len(runs),
                    "num_static_frames_in_runs": summary["result"]["num_trimmed_frames"],
                    "static_frame_ratio": summary["result"]["trimmed_frame_ratio"],
                }
            },
            camera_key=args.preview_camera,
            preview_pad_frames=args.preview_pad_frames,
        )

    print(f"Detected leading idle runs: {len(runs)}")
    print(f"Frames proposed for trimming: {summary['result']['num_trimmed_frames']}")
    print(f"Report written to: {output_dir}")
    print(f"Drop file: {output_dir / 'leading_idle_runs.jsonl'}")


if __name__ == "__main__":
    main()
