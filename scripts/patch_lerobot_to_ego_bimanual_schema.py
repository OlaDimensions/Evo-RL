#!/usr/bin/env python

"""Patch a local LeRobot dataset so shared fields match the ego-bimanual loader schema.

The script creates a new dataset directory, keeps source values intact, and only
renames shared semantic fields. Extra `complementary_info.*` columns are preserved.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.utils import write_stats


STATE_KEYS = ("action", "observation.state")
POLICY_ACTION_KEY = "complementary_info.policy_action"
BASE_STATE_KEY = "observation.base_state"
PREV_STATE_KEY = "prev_state"

VIDEO_KEY_MAP = {
    "observation.images.right_front": "observation.images.main",
    "observation.images.left_wrist": "observation.images.camera0",
    "observation.images.right_wrist": "observation.images.camera1",
}

ROBOT0_ROBOT1_NAMES = [
    [
        "robot0_eef_pos_x",
        "robot0_eef_pos_y",
        "robot0_eef_pos_z",
        "robot0_eef_rotvec_x",
        "robot0_eef_rotvec_y",
        "robot0_eef_rotvec_z",
        "robot0_gripper",
        "robot1_eef_pos_x",
        "robot1_eef_pos_y",
        "robot1_eef_pos_z",
        "robot1_eef_rotvec_x",
        "robot1_eef_rotvec_y",
        "robot1_eef_rotvec_z",
        "robot1_gripper",
    ]
]

PREV_STATE_NAMES = [[f"prev_{name}" for name in ROBOT0_ROBOT1_NAMES[0]]]
BASE_STATE_NAMES = [
    "base_camera_pos_x",
    "base_camera_pos_y",
    "base_camera_pos_z",
    "base_camera_rotvec_x",
    "base_camera_rotvec_y",
    "base_camera_rotvec_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a local LeRobot dataset and map shared field names to the ego-bimanual schema while "
            "preserving complementary_info fields."
        )
    )
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--target-info", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def default_output_root(src_root: Path) -> Path:
    return src_root.with_name(f"{src_root.name}-ego-bimanual-schema")


def remap_feature_key(key: str) -> str:
    return VIDEO_KEY_MAP.get(key, key)


def remap_stats_keys(stats: dict[str, Any]) -> dict[str, Any]:
    return {remap_feature_key(key): value for key, value in stats.items()}


def remap_episode_column(column: str) -> str:
    for old_key, new_key in VIDEO_KEY_MAP.items():
        old_prefix = f"videos/{old_key}/"
        if column.startswith(old_prefix):
            return f"videos/{new_key}/{column[len(old_prefix):]}"
        old_stats_prefix = f"stats/{old_key}/"
        if column.startswith(old_stats_prefix):
            return f"stats/{new_key}/{column[len(old_stats_prefix):]}"
    return column


def feature_from_target(
    target_features: dict[str, Any],
    key: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    return deepcopy(target_features.get(key, fallback))


def update_video_feature_from_source(
    dst_features: dict[str, Any],
    src_features: dict[str, Any],
    old_key: str,
    new_key: str,
) -> None:
    feature = deepcopy(src_features[old_key])
    feature["shape"] = list(feature["shape"])
    if "info" in feature:
        info = deepcopy(feature["info"])
        if len(feature["shape"]) >= 2:
            info["video.height"] = int(feature["shape"][0])
            info["video.width"] = int(feature["shape"][1])
        feature["info"] = info
    dst_features[new_key] = feature


def build_patched_info(src_info: dict[str, Any], target_info: dict[str, Any]) -> dict[str, Any]:
    patched = deepcopy(src_info)
    src_features = src_info["features"]
    target_features = target_info.get("features", {})

    patched["robot_type"] = "ego_bimanual"
    patched_features: dict[str, Any] = {}

    for key, feature in src_features.items():
        if key in VIDEO_KEY_MAP:
            update_video_feature_from_source(patched_features, src_features, key, VIDEO_KEY_MAP[key])
            continue

        new_feature = deepcopy(feature)
        if key in STATE_KEYS:
            new_feature["names"] = deepcopy(
                target_features.get(key, {}).get("names", ROBOT0_ROBOT1_NAMES)
            )
        elif key == POLICY_ACTION_KEY:
            new_feature["names"] = deepcopy(
                target_features.get("action", {}).get("names", ROBOT0_ROBOT1_NAMES)
            )
        patched_features[key] = new_feature

    base_fallback = {"dtype": "float32", "shape": [6], "names": BASE_STATE_NAMES}
    prev_fallback = {"dtype": "float32", "shape": [14], "names": PREV_STATE_NAMES}
    patched_features[BASE_STATE_KEY] = feature_from_target(target_features, BASE_STATE_KEY, base_fallback)
    patched_features[PREV_STATE_KEY] = feature_from_target(target_features, PREV_STATE_KEY, prev_fallback)

    patched["features"] = patched_features
    return patched


def copy_dataset_tree(src_root: Path, output_root: Path, overwrite: bool) -> None:
    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {src_root}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    shutil.copytree(src_root, output_root)


def as_float_matrix(series: pd.Series, width: int | None = None) -> np.ndarray:
    values = np.stack(series.map(lambda value: np.asarray(value, dtype=np.float32)).to_list())
    if width is not None and values.shape[1] != width:
        raise ValueError(f"Expected width {width}, got {values.shape[1]}")
    return values.astype(np.float32, copy=False)


def build_prev_state(df: pd.DataFrame, previous_by_episode: dict[int, np.ndarray]) -> np.ndarray:
    states = as_float_matrix(df["observation.state"], width=14)
    prev_states = np.empty_like(states)

    for row_offset, (episode_index, state) in enumerate(zip(df["episode_index"], states, strict=True)):
        ep_idx = int(episode_index)
        prev_states[row_offset] = previous_by_episode.get(ep_idx, state)
        previous_by_episode[ep_idx] = state

    return prev_states


def fixed_size_float_list_array(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    flat_values = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat_values, values.shape[1])


def write_data_parquet(
    table: pa.Table,
    path: Path,
    prev_state: np.ndarray,
    base_state: np.ndarray,
    features: dict[str, Any],
) -> None:
    table = table.append_column(PREV_STATE_KEY, fixed_size_float_list_array(prev_state))
    table = table.append_column(BASE_STATE_KEY, fixed_size_float_list_array(base_state))
    data_feature_order = [key for key, feature in features.items() if feature.get("dtype") != "video"]
    missing = set(data_feature_order) - set(table.column_names)
    extra = set(table.column_names) - set(data_feature_order)
    if missing or extra:
        raise ValueError(
            "Data parquet columns do not match non-video features: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    table = table.select(data_feature_order)
    writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def rewrite_data_files(
    root: Path,
    features: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    data_paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    previous_by_episode: dict[int, np.ndarray] = {}
    all_prev_state: list[np.ndarray] = []
    all_base_state: list[np.ndarray] = []
    episode_values: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    for path in tqdm(data_paths, desc="Rewriting data parquet"):
        table = pq.read_table(path)
        df = table.to_pandas().reset_index(drop=True)
        prev_state = build_prev_state(df, previous_by_episode)
        base_state = np.zeros((len(df), 6), dtype=np.float32)

        all_prev_state.append(prev_state)
        all_base_state.append(base_state)

        for ep_idx in sorted(df["episode_index"].unique()):
            ep_mask = df["episode_index"] == ep_idx
            episode_values[int(ep_idx)][PREV_STATE_KEY].append(prev_state[ep_mask.to_numpy()])
            episode_values[int(ep_idx)][BASE_STATE_KEY].append(base_state[ep_mask.to_numpy()])

        write_data_parquet(table, path, prev_state, base_state, features)

    all_arrays = {
        PREV_STATE_KEY: np.concatenate(all_prev_state, axis=0),
        BASE_STATE_KEY: np.concatenate(all_base_state, axis=0),
    }
    episode_arrays = {
        ep_idx: {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in values_by_key.items()
        }
        for ep_idx, values_by_key in episode_values.items()
    }
    return all_arrays, episode_arrays


def stats_for_arrays(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return {key: get_feature_stats(value, axis=0, keepdims=False) for key, value in arrays.items()}


def rewrite_stats(root: Path, added_stats: dict[str, dict[str, np.ndarray]]) -> None:
    stats_path = root / "meta" / "stats.json"
    stats = load_json(stats_path) if stats_path.exists() else {}
    stats = remap_stats_keys(stats)
    stats.update(added_stats)
    write_stats(stats, root)


def value_for_parquet(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def rewrite_episode_files(root: Path, episode_stats: dict[int, dict[str, dict[str, np.ndarray]]]) -> None:
    episode_paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'meta/episodes'}")

    for path in tqdm(episode_paths, desc="Rewriting episode metadata"):
        df = pd.read_parquet(path)
        df = df.rename(columns={column: remap_episode_column(column) for column in df.columns})

        stat_columns = [
            f"stats/{feature_key}/{stat_key}"
            for stats in episode_stats.values()
            for feature_key, feature_stats in stats.items()
            for stat_key in feature_stats
        ]
        for column in sorted(set(stat_columns)):
            if column not in df.columns:
                df[column] = pd.Series([None] * len(df), dtype=object)

        for row_idx, ep_idx in enumerate(df["episode_index"]):
            stats = episode_stats[int(ep_idx)]
            for feature_key, feature_stats in stats.items():
                for stat_key, value in feature_stats.items():
                    df.at[row_idx, f"stats/{feature_key}/{stat_key}"] = value_for_parquet(value)

        df = df.map(value_for_parquet)
        df.to_parquet(path, index=False)


def rewrite_video_dirs(root: Path) -> None:
    videos_root = root / "videos"
    if not videos_root.exists():
        return

    for old_key, new_key in VIDEO_KEY_MAP.items():
        old_path = videos_root / old_key
        new_path = videos_root / new_key
        if not old_path.exists():
            continue
        if new_path.exists():
            raise FileExistsError(f"Destination video directory already exists: {new_path}")
        old_path.rename(new_path)


def main() -> None:
    args = parse_args()
    src_root = args.src_root.expanduser().resolve()
    output_root = (args.output_root or default_output_root(src_root)).expanduser().resolve()
    target_info_path = args.target_info.expanduser().resolve()

    src_info = load_json(src_root / "meta" / "info.json")
    target_info = load_json(target_info_path)
    patched_info = build_patched_info(src_info, target_info)

    copy_dataset_tree(src_root, output_root, overwrite=args.overwrite)
    write_json(patched_info, output_root / "meta" / "info.json")

    all_arrays, episode_arrays = rewrite_data_files(output_root, patched_info["features"])
    added_stats = stats_for_arrays(all_arrays)
    rewrite_stats(output_root, added_stats)

    added_episode_stats = {
        ep_idx: stats_for_arrays(values_by_key)
        for ep_idx, values_by_key in episode_arrays.items()
    }
    rewrite_episode_files(output_root, added_episode_stats)
    rewrite_video_dirs(output_root)

    print(f"Patched dataset written to: {output_root}")


if __name__ == "__main__":
    main()
