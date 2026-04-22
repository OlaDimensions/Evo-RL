#!/usr/bin/env python3
"""Rewrite a LeRobot v3 dataset with detected static-frame runs removed.

This script never edits the source dataset in place. It creates a new dataset by
copying only kept frames through LeRobotDataset.add_frame/save_episode so data,
videos, timestamps, and metadata are regenerated together.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES, write_info


DEFAULT_SRC_DATASET_DIR = (
    Path.home()
    / ".cache/huggingface/lerobot/ruanafan/evo-rl-data-pnp-vr-ee-pose-round0-0418-1"
)


CORE_EPISODE_METADATA_PREFIXES = (
    "data/",
    "videos/",
    "stats/",
    "meta/episodes/",
)
CORE_EPISODE_METADATA_KEYS = {
    "episode_index",
    "tasks",
    "length",
    "dataset_from_index",
    "dataset_to_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a filtered LeRobot v3 dataset with static-run frame_indices removed."
    )
    parser.add_argument("--src-dataset-dir", type=Path, default=DEFAULT_SRC_DATASET_DIR)
    parser.add_argument("--static-runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Repo id to store in the filtered dataset metadata. Defaults to output directory name.",
    )
    parser.add_argument(
        "--src-repo-id",
        default=None,
        help="Repo id for loading the source dataset. Defaults to source directory name.",
    )
    parser.add_argument(
        "--drop-mode",
        choices=["all_static_frames"],
        default="all_static_frames",
        help="Deletion policy. v1 deletes every frame listed in each static run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print deletion statistics.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output directory first.")
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        choices=["h264", "hevc", "libsvtav1"],
        help="Video codec for re-encoding the filtered dataset.",
    )
    parser.add_argument(
        "--no-parallel-encoding",
        action="store_true",
        help="Encode camera videos sequentially when saving each episode.",
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=20,
        help="Number of kept-frame action/state mappings to spot-check after writing.",
    )
    return parser.parse_args()


def read_static_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Static runs file does not exist: {path}")

    runs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            run = json.loads(line)
            if "episode_index" not in run or "frame_indices" not in run:
                raise ValueError(f"Line {lineno} is missing episode_index or frame_indices")
            runs.append(run)
    return runs


def build_drop_map(runs: list[dict[str, Any]]) -> dict[int, set[int]]:
    drop_map: dict[int, set[int]] = defaultdict(set)
    for run in runs:
        episode_index = int(run["episode_index"])
        for frame_index in run["frame_indices"]:
            drop_map[episode_index].add(int(frame_index))
    return dict(drop_map)


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().numpy()
    if isinstance(value, np.generic):
        return value.item()
    return value


def as_feature_value(value: Any, feature: dict[str, Any]) -> Any:
    dtype = feature["dtype"]
    shape = tuple(feature["shape"])

    if dtype in {"image", "video"}:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        # LeRobot decodes videos as CHW tensors, while the feature shape is HWC.
        if array.ndim == 3 and len(shape) == 3 and array.shape[0] == shape[-1]:
            array = np.moveaxis(array, 0, -1)
        return array

    if dtype == "string":
        return str(value)

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    array = array.astype(np.dtype(dtype), copy=False)
    if array.shape == () and shape == (1,):
        array = array.reshape(1)
    if array.shape != shape:
        array = array.reshape(shape)
    return array


def frame_for_rewrite(src: LeRobotDataset, absolute_index: int) -> dict[str, Any]:
    item = src[absolute_index]
    frame: dict[str, Any] = {"task": item["task"]}
    for key, feature in src.features.items():
        if key in DEFAULT_FEATURES:
            continue
        frame[key] = as_feature_value(item[key], feature)
    return frame


def preserved_episode_metadata(src: LeRobotDataset, episode_index: int) -> dict[str, Any]:
    src_episode = src.meta.episodes[episode_index]
    preserved: dict[str, Any] = {}
    for key, value in src_episode.items():
        if key in CORE_EPISODE_METADATA_KEYS:
            continue
        if key.startswith(CORE_EPISODE_METADATA_PREFIXES):
            continue
        if key.startswith("__index_level_"):
            continue
        value = to_python_scalar(value)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        preserved[key] = value
    return preserved


def compute_episode_plan(src: LeRobotDataset, drop_map: dict[int, set[int]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for episode_index in range(src.meta.total_episodes):
        ep = src.meta.episodes[episode_index]
        length = int(ep["length"])
        to_drop = {idx for idx in drop_map.get(episode_index, set()) if 0 <= idx < length}
        kept = length - len(to_drop)
        plan.append(
            {
                "episode_index": episode_index,
                "old_length": length,
                "drop_count": len(to_drop),
                "new_length": kept,
                "drop_frame_indices": sorted(to_drop),
            }
        )
    return plan


def print_plan_summary(src: LeRobotDataset, plan: list[dict[str, Any]]) -> None:
    total_old = sum(row["old_length"] for row in plan)
    total_drop = sum(row["drop_count"] for row in plan)
    total_new = sum(row["new_length"] for row in plan)
    print(f"Source episodes: {src.meta.total_episodes}")
    print(f"Source frames: {total_old}")
    print(f"Frames to drop: {total_drop}")
    print(f"Filtered frames: {total_new}")
    if total_old:
        print(f"Drop ratio: {total_drop / total_old:.4%}")
    print("Per-episode drop counts:")
    for row in plan:
        print(
            f"  ep {row['episode_index']:04d}: "
            f"{row['old_length']} -> {row['new_length']} "
            f"(drop {row['drop_count']})"
        )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)


def write_filtered_dataset(
    *,
    src: LeRobotDataset,
    plan: list[dict[str, Any]],
    output_dir: Path,
    repo_id: str,
    vcodec: str,
    parallel_encoding: bool,
) -> list[dict[str, int]]:
    dst = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=src.fps,
        features=src.meta.features,
        robot_type=src.meta.robot_type,
        use_videos=len(src.meta.video_keys) > 0,
        vcodec=vcodec,
    )
    dst.meta.info["chunks_size"] = src.meta.chunks_size
    dst.meta.info["data_files_size_in_mb"] = src.meta.data_files_size_in_mb
    dst.meta.info["video_files_size_in_mb"] = src.meta.video_files_size_in_mb
    write_info(dst.meta.info, dst.meta.root)

    kept_mapping: list[dict[str, int]] = []
    try:
        for row in tqdm(plan, desc="Rewriting episodes"):
            if row["new_length"] <= 0:
                raise ValueError(
                    f"Episode {row['episode_index']} would be empty after filtering; refusing to write."
                )

            episode_index = row["episode_index"]
            drop_set = set(row["drop_frame_indices"])
            ep = src.meta.episodes[episode_index]
            start = int(ep["dataset_from_index"])
            old_length = int(ep["length"])
            new_frame_index = 0

            for old_frame_index in range(old_length):
                if old_frame_index in drop_set:
                    continue
                absolute_index = start + old_frame_index
                dst.add_frame(frame_for_rewrite(src, absolute_index))
                kept_mapping.append(
                    {
                        "episode_index": episode_index,
                        "old_frame_index": old_frame_index,
                        "new_frame_index": new_frame_index,
                        "old_global_index": absolute_index,
                    }
                )
                new_frame_index += 1

            dst.save_episode(
                parallel_encoding=parallel_encoding,
                extra_episode_metadata=preserved_episode_metadata(src, episode_index),
            )
    finally:
        dst.finalize()

    return kept_mapping


def read_all_data_table(dataset_dir: Path) -> pa.Table:
    files = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet files found under {dataset_dir / 'data'}")
    tables = [pq.read_table(path) for path in files]
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")


def read_all_episodes_table(dataset_dir: Path) -> pa.Table:
    files = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_dir / 'meta' / 'episodes'}")
    tables = [pq.read_table(path) for path in files]
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")


def verify_filtered_dataset(
    *,
    src: LeRobotDataset,
    output_dir: Path,
    plan: list[dict[str, Any]],
    kept_mapping: list[dict[str, int]],
    verify_samples: int,
) -> None:
    info_path = output_dir / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    data = read_all_data_table(output_dir).to_pydict()
    episodes = read_all_episodes_table(output_dir).to_pydict()
    fps = float(info["fps"])

    expected_total = sum(row["new_length"] for row in plan)
    if int(info["total_frames"]) != expected_total:
        raise AssertionError(f"total_frames mismatch: {info['total_frames']} != {expected_total}")

    data_indices = np.asarray(data["index"], dtype=np.int64)
    if not np.array_equal(data_indices, np.arange(expected_total, dtype=np.int64)):
        raise AssertionError("Global index is not contiguous from 0 to total_frames - 1")

    episode_lengths = {
        int(ep): int(length)
        for ep, length in zip(episodes["episode_index"], episodes["length"], strict=True)
    }
    from_indices = {
        int(ep): int(start)
        for ep, start in zip(episodes["episode_index"], episodes["dataset_from_index"], strict=True)
    }
    to_indices = {
        int(ep): int(end)
        for ep, end in zip(episodes["episode_index"], episodes["dataset_to_index"], strict=True)
    }

    expected_cursor = 0
    for row in plan:
        ep = row["episode_index"]
        expected_length = row["new_length"]
        if episode_lengths.get(ep) != expected_length:
            raise AssertionError(f"Episode {ep} length mismatch: {episode_lengths.get(ep)} != {expected_length}")
        if from_indices.get(ep) != expected_cursor or to_indices.get(ep) != expected_cursor + expected_length:
            raise AssertionError(f"Episode {ep} dataset_from/to_index are not contiguous")
        expected_cursor += expected_length

        mask = np.asarray(data["episode_index"], dtype=np.int64) == ep
        frame_indices = np.asarray(data["frame_index"], dtype=np.int64)[mask]
        timestamps = np.asarray(data["timestamp"], dtype=np.float64)[mask]
        if not np.array_equal(frame_indices, np.arange(expected_length, dtype=np.int64)):
            raise AssertionError(f"Episode {ep} frame_index is not contiguous")
        if not np.allclose(timestamps, frame_indices / fps, atol=1e-4):
            raise AssertionError(f"Episode {ep} timestamp is not frame_index / fps")

    for camera_key in src.meta.video_keys:
        from_key = f"videos/{camera_key}/from_timestamp"
        to_key = f"videos/{camera_key}/to_timestamp"
        if from_key not in episodes or to_key not in episodes:
            raise AssertionError(f"Missing video timestamps for {camera_key}")
        for ep, start_ts, end_ts in zip(
            episodes["episode_index"], episodes[from_key], episodes[to_key], strict=True
        ):
            length = episode_lengths[int(ep)]
            duration = float(end_ts) - float(start_ts)
            expected_duration = length / fps
            if not math.isclose(duration, expected_duration, abs_tol=(1.0 / fps + 1e-3)):
                raise AssertionError(
                    f"Video duration mismatch for {camera_key} episode {ep}: "
                    f"{duration:.6f} != {expected_duration:.6f}"
                )

    if verify_samples > 0 and kept_mapping:
        sample_indices = np.linspace(0, len(kept_mapping) - 1, num=min(verify_samples, len(kept_mapping)))
        sample_indices = sorted({int(round(idx)) for idx in sample_indices})
        dst_action = data.get("action")
        dst_state = data.get("observation.state")
        if dst_action is not None and dst_state is not None:
            for map_index in sample_indices:
                mapping = kept_mapping[map_index]
                new_global_index = map_index
                src_item = src[mapping["old_global_index"]]
                if not np.allclose(
                    np.asarray(dst_action[new_global_index], dtype=np.float32),
                    src_item["action"].detach().cpu().numpy(),
                    atol=1e-6,
                ):
                    raise AssertionError(f"Action mismatch at kept mapping index {map_index}")
                if not np.allclose(
                    np.asarray(dst_state[new_global_index], dtype=np.float32),
                    src_item["observation.state"].detach().cpu().numpy(),
                    atol=1e-6,
                ):
                    raise AssertionError(f"State mismatch at kept mapping index {map_index}")


def main() -> None:
    args = parse_args()
    src_dataset_dir = args.src_dataset_dir.expanduser().resolve()
    static_runs_path = args.static_runs.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    src_repo_id = args.src_repo_id or src_dataset_dir.name
    repo_id = args.repo_id or output_dir.name

    src = LeRobotDataset(src_repo_id, root=src_dataset_dir, download_videos=False)
    runs = read_static_runs(static_runs_path)
    drop_map = build_drop_map(runs)
    plan = compute_episode_plan(src, drop_map)
    print_plan_summary(src, plan)

    empty_episodes = [row["episode_index"] for row in plan if row["new_length"] <= 0]
    if empty_episodes:
        raise ValueError(f"These episodes would become empty after filtering: {empty_episodes}")

    if args.dry_run:
        return

    prepare_output_dir(output_dir, args.overwrite)
    kept_mapping = write_filtered_dataset(
        src=src,
        plan=plan,
        output_dir=output_dir,
        repo_id=repo_id,
        vcodec=args.vcodec,
        parallel_encoding=not args.no_parallel_encoding,
    )
    verify_filtered_dataset(
        src=src,
        output_dir=output_dir,
        plan=plan,
        kept_mapping=kept_mapping,
        verify_samples=args.verify_samples,
    )
    print(f"Filtered dataset written to: {output_dir}")
    print("Verification passed.")


if __name__ == "__main__":
    main()
