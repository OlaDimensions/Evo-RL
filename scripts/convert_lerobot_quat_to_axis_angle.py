#!/usr/bin/env python

"""Copy a LeRobot dataset and convert quaternion pose vectors to Euler angles."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.utils import (
    DATA_DIR,
    flatten_dict,
    get_hf_features_from_features,
    load_info,
    load_stats,
    write_info,
    write_stats,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

DEFAULT_SRC_REPO_ID = "ruanafan/evo-rl-data-pnp-vr-ee-pose-round0-0418-all"
DEFAULT_DST_REPO_ID = f"{DEFAULT_SRC_REPO_ID}-rpy"
DEFAULT_COLUMNS = ("action", "observation.state", "complementary_info.policy_action")
EULER_FORMAT_RPY = "rpy"
EULER_FORMAT_RXRYRZ = "rxryrz"
EULER_FORMATS = (EULER_FORMAT_RPY, EULER_FORMAT_RXRYRZ)


def quaternion_xyzw_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Convert one quaternion `[x, y, z, w]` to Euler `[roll, pitch, yaw]` in radians."""
    return rotation_matrix_to_rpy(quaternion_xyzw_to_rotation_matrix(quat))


def quaternion_xyzw_to_rxryrz(quat: np.ndarray) -> np.ndarray:
    """Convert one quaternion `[x, y, z, w]` to XYZ Euler `[rx, ry, rz]` in radians."""
    return rotation_matrix_to_rxryrz(quaternion_xyzw_to_rotation_matrix(quat))


def quaternion_xyzw_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0.0 or not np.isfinite(norm):
        return np.eye(3, dtype=np.float64)

    qx, qy, qz, qw = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to RPY for `R = Rz(yaw) * Ry(pitch) * Rx(roll)`."""
    matrix = np.asarray(matrix, dtype=np.float64)
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float32)


def rotation_matrix_to_rxryrz(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to XYZ Euler for `R = Rx(rx) * Ry(ry) * Rz(rz)`."""
    matrix = np.asarray(matrix, dtype=np.float64)
    rx = np.arctan2(-matrix[1, 2], matrix[2, 2])
    ry = np.arcsin(np.clip(matrix[0, 2], -1.0, 1.0))
    rz = np.arctan2(-matrix[0, 1], matrix[0, 0])
    return np.array([rx, ry, rz], dtype=np.float32)


def euler_names_from_quaternion_names(names: list[str], euler_format: str = EULER_FORMAT_RPY) -> list[str]:
    if euler_format not in EULER_FORMATS:
        raise ValueError(f"`euler_format` must be one of {EULER_FORMATS}, got {euler_format!r}.")
    suffixes = (".roll", ".pitch", ".yaw") if euler_format == EULER_FORMAT_RPY else (".rx", ".ry", ".rz")
    converted: list[str] = []
    index = 0
    while index < len(names):
        name = names[index]
        if name.endswith(".qx"):
            quat_names = names[index : index + 4]
            expected = [name[:-3] + suffix for suffix in (".qx", ".qy", ".qz", ".qw")]
            if quat_names != expected:
                raise ValueError(f"Quaternion names must be contiguous as {expected}, got {quat_names}")
            converted.extend([name[:-3] + suffix for suffix in suffixes])
            index += 4
        else:
            converted.append(name)
            index += 1
    return converted


def rpy_names_from_quaternion_names(names: list[str]) -> list[str]:
    return euler_names_from_quaternion_names(names, EULER_FORMAT_RPY)


def rxryrz_names_from_quaternion_names(names: list[str]) -> list[str]:
    return euler_names_from_quaternion_names(names, EULER_FORMAT_RXRYRZ)


def convert_pose_batch(values: np.ndarray, names: list[str], euler_format: str = EULER_FORMAT_RPY) -> np.ndarray:
    if euler_format not in EULER_FORMATS:
        raise ValueError(f"`euler_format` must be one of {EULER_FORMATS}, got {euler_format!r}.")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D pose array, got shape {values.shape}")
    if values.shape[1] != len(names):
        raise ValueError(f"Pose width {values.shape[1]} does not match {len(names)} names")

    parts: list[np.ndarray] = []
    index = 0
    while index < len(names):
        name = names[index]
        if name.endswith(".qx"):
            quat_names = names[index : index + 4]
            expected = [name[:-3] + suffix for suffix in (".qx", ".qy", ".qz", ".qw")]
            if quat_names != expected:
                raise ValueError(f"Quaternion values must be contiguous as {expected}, got {quat_names}")
            converter = quaternion_xyzw_to_rpy if euler_format == EULER_FORMAT_RPY else quaternion_xyzw_to_rxryrz
            euler = np.stack([converter(quat) for quat in values[:, index : index + 4]])
            parts.append(euler)
            index += 4
        else:
            parts.append(values[:, index : index + 1])
            index += 1

    return np.concatenate(parts, axis=1).astype(np.float32)


def _copy_dataset_tree(src_root: Path, dst_root: Path, overwrite: bool) -> None:
    if not src_root.exists():
        raise FileNotFoundError(
            f"Source dataset root does not exist: {src_root}. "
            "Load or download the source dataset first, or pass --root."
        )
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst_root}. Pass --overwrite to replace it.")
        shutil.rmtree(dst_root)
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, dst_root)


def _converted_features(
    info: dict,
    columns: tuple[str, ...],
    euler_format: str,
) -> tuple[dict, dict[str, list[str]]]:
    features = dict(info["features"])
    source_names: dict[str, list[str]] = {}
    for column in columns:
        if column not in features:
            logging.info("Skipping missing feature %s", column)
            continue
        feature = dict(features[column])
        names = list(feature.get("names") or [])
        if not names:
            raise ValueError(f"Feature {column} has no names; cannot locate quaternion fields")
        new_names = euler_names_from_quaternion_names(names, euler_format)
        if len(new_names) == len(names):
            raise ValueError(f"Feature {column} does not contain quaternion names ending in .qx/.qy/.qz/.qw")
        feature["names"] = new_names
        feature["shape"] = (len(new_names),)
        features[column] = feature
        source_names[column] = names
    info = dict(info)
    info["features"] = features
    return info, source_names


def _write_data_parquet(df: pd.DataFrame, path: Path, features: dict) -> None:
    hf_features = get_hf_features_from_features(features)
    dataset = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")
    table = dataset.with_format("arrow")[:]
    writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def _rewrite_data_files(
    dst_root: Path,
    features: dict,
    source_names: dict[str, list[str]],
    euler_format: str,
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    all_values = {column: [] for column in source_names}
    episode_values: dict[int, dict[str, list[np.ndarray]]] = {}
    data_files = sorted((dst_root / DATA_DIR).glob("*/*.parquet"))
    if not data_files:
        raise ValueError(f"No parquet data files found under {dst_root / DATA_DIR}")

    for path in tqdm(data_files, desc="Converting data"):
        df = pd.read_parquet(path).reset_index(drop=True)
        for column, names in source_names.items():
            values = np.stack(df[column].map(lambda value: np.asarray(value, dtype=np.float32)).to_list())
            converted = convert_pose_batch(values, names, euler_format=euler_format)
            df[column] = list(converted)
            all_values[column].append(converted)

            for ep_idx in sorted(df["episode_index"].unique()):
                ep_mask = df["episode_index"] == ep_idx
                episode_values.setdefault(int(ep_idx), {}).setdefault(column, []).append(converted[ep_mask])

        _write_data_parquet(df, path, features)

    all_arrays = {column: np.concatenate(chunks, axis=0) for column, chunks in all_values.items()}
    episode_arrays = {
        ep_idx: {column: np.concatenate(chunks, axis=0) for column, chunks in values.items()}
        for ep_idx, values in episode_values.items()
    }
    return all_arrays, episode_arrays


def _stats_for_arrays(arrays: dict[str, np.ndarray]) -> dict[str, dict]:
    return {
        column: get_feature_stats(values, axis=0, keepdims=False)
        for column, values in arrays.items()
    }


def _rewrite_dataset_stats(dst_root: Path, converted_stats: dict[str, dict]) -> None:
    stats = load_stats(dst_root) or {}
    stats.update(converted_stats)
    write_stats(stats, dst_root)


def _rewrite_episode_stats(dst_root: Path, episode_stats: dict[int, dict[str, dict]]) -> None:
    episodes_dir = dst_root / "meta" / "episodes"
    for path in tqdm(sorted(episodes_dir.glob("*/*.parquet")), desc="Updating episode stats"):
        df = pd.read_parquet(path)
        for row_idx, ep_idx in enumerate(df["episode_index"]):
            stats = episode_stats[int(ep_idx)]
            for flat_key, value in flatten_dict({"stats": stats}).items():
                if flat_key in df.columns:
                    df.at[row_idx, flat_key] = value.tolist() if isinstance(value, np.ndarray) else value
        path.parent.mkdir(parents=True, exist_ok=True)
        df = df.map(lambda value: value.tolist() if isinstance(value, np.ndarray) else value)
        df.to_parquet(path, index=False)


def convert_dataset(
    src_repo_id: str = DEFAULT_SRC_REPO_ID,
    dst_repo_id: str = DEFAULT_DST_REPO_ID,
    root: Path = HF_LEROBOT_HOME,
    columns: tuple[str, ...] = DEFAULT_COLUMNS,
    euler_format: str = EULER_FORMAT_RPY,
    overwrite: bool = False,
    push_to_hub: bool = False,
) -> Path:
    src_root = root / src_repo_id
    dst_root = root / dst_repo_id

    _copy_dataset_tree(src_root, dst_root, overwrite=overwrite)

    info = load_info(dst_root)
    new_info, source_names = _converted_features(info, columns, euler_format=euler_format)
    if not source_names:
        raise ValueError("None of the requested columns were found in the dataset")
    write_info(new_info, dst_root)

    all_arrays, episode_arrays = _rewrite_data_files(
        dst_root,
        new_info["features"],
        source_names,
        euler_format=euler_format,
    )
    dataset_stats = _stats_for_arrays(all_arrays)
    episode_stats = {ep_idx: _stats_for_arrays(arrays) for ep_idx, arrays in episode_arrays.items()}

    _rewrite_dataset_stats(dst_root, dataset_stats)
    _rewrite_episode_stats(dst_root, episode_stats)

    # Validate the final copy can be loaded by LeRobot.
    LeRobotDataset(dst_repo_id, root=dst_root)

    if push_to_hub:
        LeRobotDataset(dst_repo_id, root=dst_root).push_to_hub()

    return dst_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-repo-id", default=DEFAULT_SRC_REPO_ID)
    parser.add_argument("--dst-repo-id", default=DEFAULT_DST_REPO_ID)
    parser.add_argument("--root", type=Path, default=HF_LEROBOT_HOME)
    parser.add_argument("--columns", nargs="+", default=list(DEFAULT_COLUMNS))
    parser.add_argument("--euler-format", choices=EULER_FORMATS, default=EULER_FORMAT_RPY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    output_root = convert_dataset(
        src_repo_id=args.src_repo_id,
        dst_repo_id=args.dst_repo_id,
        root=args.root,
        columns=tuple(args.columns),
        euler_format=args.euler_format,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
    )
    logging.info("Converted dataset written to %s", output_root)


if __name__ == "__main__":
    main()
