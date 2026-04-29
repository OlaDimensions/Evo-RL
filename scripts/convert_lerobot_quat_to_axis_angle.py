#!/usr/bin/env python

"""Copy a LeRobot dataset and convert pose rotation vectors between supported formats."""

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
from lerobot.utils.rotation import Rotation

DEFAULT_SRC_REPO_ID = "ruanafan/evo-rl-data-pnp-vr-ee-pose-round0-0418-all"
DEFAULT_DST_REPO_ID = f"{DEFAULT_SRC_REPO_ID}-rpy"
DEFAULT_COLUMNS = ("action", "observation.state", "complementary_info.policy_action")
EULER_FORMAT_RPY = "rpy"
EULER_FORMAT_RXRYRZ = "rxryrz"
EULER_FORMATS = (EULER_FORMAT_RPY, EULER_FORMAT_RXRYRZ)
ROTATION_FORMAT_AUTO = "auto"
ROTATION_FORMAT_QUAT = "quat"
ROTATION_FORMAT_ROTVEC = "rotvec"
ROTATION_FORMATS = (ROTATION_FORMAT_QUAT, EULER_FORMAT_RPY, EULER_FORMAT_RXRYRZ, ROTATION_FORMAT_ROTVEC)
INPUT_ROTATION_FORMATS = (ROTATION_FORMAT_AUTO, *ROTATION_FORMATS)
TARGET_ROTATION_FORMATS = (EULER_FORMAT_RPY, EULER_FORMAT_RXRYRZ, ROTATION_FORMAT_ROTVEC)
ROTATION_SUFFIXES = {
    ROTATION_FORMAT_QUAT: ("qx", "qy", "qz", "qw"),
    EULER_FORMAT_RPY: ("roll", "pitch", "yaw"),
    EULER_FORMAT_RXRYRZ: ("rx", "ry", "rz"),
    ROTATION_FORMAT_ROTVEC: ("rotvec_x", "rotvec_y", "rotvec_z"),
}


def quaternion_xyzw_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Convert one quaternion `[x, y, z, w]` to Euler `[roll, pitch, yaw]` in radians."""
    return rotation_matrix_to_rpy(quaternion_xyzw_to_rotation_matrix(quat))


def quaternion_xyzw_to_rxryrz(quat: np.ndarray) -> np.ndarray:
    """Convert one quaternion `[x, y, z, w]` to XYZ Euler `[rx, ry, rz]` in radians."""
    return rotation_matrix_to_rxryrz(quaternion_xyzw_to_rotation_matrix(quat))


def quaternion_xyzw_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Convert one quaternion `[x, y, z, w]` to a rotation vector in radians."""
    return rotation_matrix_to_rotvec(quaternion_xyzw_to_rotation_matrix(quat))


def rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert RPY `[roll, pitch, yaw]` to a matrix for `R = Rz(yaw) * Ry(pitch) * Rx(roll)`."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rxryrz_to_rotation_matrix(rxryrz: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler `[rx, ry, rz]` to a matrix for `R = Rx(rx) * Ry(ry) * Rz(rz)`."""
    rx, ry, rz = np.asarray(rxryrz, dtype=np.float64)
    sa, ca = np.sin(rx), np.cos(rx)
    sb, cb = np.sin(ry), np.cos(ry)
    sc, cc = np.sin(rz), np.cos(rz)
    return np.array(
        [
            [cb * cc, -cb * sc, sb],
            [sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb],
            [-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb],
        ],
        dtype=np.float64,
    )


def rotvec_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert one rotation vector to a rotation matrix."""
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()


def rpy_to_rxryrz(rpy: np.ndarray) -> np.ndarray:
    """Convert one RPY vector `[roll, pitch, yaw]` to XYZ Euler `[rx, ry, rz]` in radians."""
    return rotation_matrix_to_rxryrz(rpy_to_rotation_matrix(rpy))


def rpy_to_rotvec(rpy: np.ndarray) -> np.ndarray:
    """Convert one RPY vector `[roll, pitch, yaw]` to a rotation vector in radians."""
    return rotation_matrix_to_rotvec(rpy_to_rotation_matrix(rpy))


def rxryrz_to_rpy(rxryrz: np.ndarray) -> np.ndarray:
    """Convert one XYZ Euler vector `[rx, ry, rz]` to RPY `[roll, pitch, yaw]` in radians."""
    return rotation_matrix_to_rpy(rxryrz_to_rotation_matrix(rxryrz))


def rxryrz_to_rotvec(rxryrz: np.ndarray) -> np.ndarray:
    """Convert one XYZ Euler vector `[rx, ry, rz]` to a rotation vector in radians."""
    return rotation_matrix_to_rotvec(rxryrz_to_rotation_matrix(rxryrz))


def rotvec_to_rpy(rotvec: np.ndarray) -> np.ndarray:
    """Convert one rotation vector to RPY `[roll, pitch, yaw]` in radians."""
    return rotation_matrix_to_rpy(rotvec_to_rotation_matrix(rotvec))


def rotvec_to_rxryrz(rotvec: np.ndarray) -> np.ndarray:
    """Convert one rotation vector to XYZ Euler `[rx, ry, rz]` in radians."""
    return rotation_matrix_to_rxryrz(rotvec_to_rotation_matrix(rotvec))


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


def rotation_matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a rotation vector in radians."""
    return Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_rotvec().astype(np.float32)


def _rotation_group_at(names: list[str], index: int) -> tuple[str, str, int] | None:
    name = names[index]
    if "." not in name:
        return None

    prefix, suffix = name.rsplit(".", 1)
    for rotation_format, suffixes in ROTATION_SUFFIXES.items():
        expected = [f"{prefix}.{group_suffix}" for group_suffix in suffixes]
        if suffix == suffixes[0]:
            actual = names[index : index + len(suffixes)]
            if actual != expected:
                raise ValueError(f"{rotation_format} names must be contiguous as {expected}, got {actual}")
            return rotation_format, prefix, len(suffixes)
        if suffix in suffixes[1:]:
            raise ValueError(f"Rotation names must start with {expected[0]!r}, got {name!r}")

    return None


def pose_rotation_format(names: list[str]) -> str | None:
    formats: set[str] = set()
    index = 0
    while index < len(names):
        group = _rotation_group_at(names, index)
        if group is None:
            index += 1
            continue
        rotation_format, _, width = group
        formats.add(rotation_format)
        index += width

    if not formats:
        return None
    if len(formats) > 1:
        raise ValueError(f"Pose feature mixes rotation formats {sorted(formats)}")
    return next(iter(formats))


def _validate_input_rotation_format(input_rotation_format: str) -> None:
    if input_rotation_format not in INPUT_ROTATION_FORMATS:
        raise ValueError(
            f"`input_rotation_format` must be one of {INPUT_ROTATION_FORMATS}, got {input_rotation_format!r}."
        )


def _resolve_target_rotation_format(
    euler_format: str | None = None,
    target_rotation_format: str | None = None,
) -> str:
    if euler_format is not None and euler_format not in EULER_FORMATS:
        raise ValueError(f"`euler_format` must be one of {EULER_FORMATS}, got {euler_format!r}.")
    if target_rotation_format is not None and target_rotation_format not in TARGET_ROTATION_FORMATS:
        raise ValueError(
            "`target_rotation_format` must be one of "
            f"{TARGET_ROTATION_FORMATS}, got {target_rotation_format!r}."
        )
    if (
        euler_format is not None
        and target_rotation_format is not None
        and euler_format != target_rotation_format
    ):
        raise ValueError(
            f"`euler_format` ({euler_format!r}) and `target_rotation_format` "
            f"({target_rotation_format!r}) must match when both are provided."
        )
    return target_rotation_format or euler_format or EULER_FORMAT_RPY


def _validate_detected_input_format(names: list[str], input_rotation_format: str) -> str | None:
    _validate_input_rotation_format(input_rotation_format)
    source_format = pose_rotation_format(names)
    if input_rotation_format != ROTATION_FORMAT_AUTO and source_format != input_rotation_format:
        raise ValueError(
            f"Pose feature uses {source_format!r} rotations, expected {input_rotation_format!r}."
        )
    return source_format


def rotation_names_from_pose_names(
    names: list[str],
    target_rotation_format: str = EULER_FORMAT_RPY,
) -> list[str]:
    if target_rotation_format not in TARGET_ROTATION_FORMATS:
        raise ValueError(
            "`target_rotation_format` must be one of "
            f"{TARGET_ROTATION_FORMATS}, got {target_rotation_format!r}."
        )
    suffixes = ROTATION_SUFFIXES[target_rotation_format]
    converted: list[str] = []
    index = 0
    while index < len(names):
        group = _rotation_group_at(names, index)
        if group is None:
            converted.append(names[index])
            index += 1
            continue

        _, prefix, width = group
        converted.extend([f"{prefix}.{suffix}" for suffix in suffixes])
        index += width
    return converted


def euler_names_from_pose_names(names: list[str], euler_format: str = EULER_FORMAT_RPY) -> list[str]:
    if euler_format not in EULER_FORMATS:
        raise ValueError(f"`euler_format` must be one of {EULER_FORMATS}, got {euler_format!r}.")
    return rotation_names_from_pose_names(names, euler_format)


def euler_names_from_quaternion_names(names: list[str], euler_format: str = EULER_FORMAT_RPY) -> list[str]:
    return euler_names_from_pose_names(names, euler_format)


def rpy_names_from_quaternion_names(names: list[str]) -> list[str]:
    return euler_names_from_quaternion_names(names, EULER_FORMAT_RPY)


def rxryrz_names_from_quaternion_names(names: list[str]) -> list[str]:
    return euler_names_from_quaternion_names(names, EULER_FORMAT_RXRYRZ)


def rotvec_names_from_pose_names(names: list[str]) -> list[str]:
    return rotation_names_from_pose_names(names, ROTATION_FORMAT_ROTVEC)


def rotvec_names_from_quaternion_names(names: list[str]) -> list[str]:
    return rotvec_names_from_pose_names(names)


def _rotation_matrix_from_values(rotation: np.ndarray, source_format: str) -> np.ndarray:
    if source_format == ROTATION_FORMAT_QUAT:
        return quaternion_xyzw_to_rotation_matrix(rotation)
    if source_format == EULER_FORMAT_RPY:
        return rpy_to_rotation_matrix(rotation)
    if source_format == EULER_FORMAT_RXRYRZ:
        return rxryrz_to_rotation_matrix(rotation)
    if source_format == ROTATION_FORMAT_ROTVEC:
        return rotvec_to_rotation_matrix(rotation)
    raise ValueError(f"Unsupported source rotation format {source_format!r}")


def _rotation_values_from_matrix(matrix: np.ndarray, target_rotation_format: str) -> np.ndarray:
    if target_rotation_format == EULER_FORMAT_RPY:
        return rotation_matrix_to_rpy(matrix)
    if target_rotation_format == EULER_FORMAT_RXRYRZ:
        return rotation_matrix_to_rxryrz(matrix)
    if target_rotation_format == ROTATION_FORMAT_ROTVEC:
        return rotation_matrix_to_rotvec(matrix)
    raise ValueError(f"Unsupported target rotation format {target_rotation_format!r}")


def convert_rotation_values(
    rotation: np.ndarray,
    source_format: str,
    target_rotation_format: str,
) -> np.ndarray:
    """Convert one rotation between supported pose rotation formats."""
    if source_format == target_rotation_format:
        return np.asarray(rotation, dtype=np.float32)
    matrix = _rotation_matrix_from_values(rotation, source_format)
    return _rotation_values_from_matrix(matrix, target_rotation_format)


def convert_pose_batch(
    values: np.ndarray,
    names: list[str],
    euler_format: str | None = None,
    target_rotation_format: str | None = None,
    input_rotation_format: str = ROTATION_FORMAT_AUTO,
) -> np.ndarray:
    target_rotation_format = _resolve_target_rotation_format(euler_format, target_rotation_format)
    _validate_detected_input_format(names, input_rotation_format)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D pose array, got shape {values.shape}")
    if values.shape[1] != len(names):
        raise ValueError(f"Pose width {values.shape[1]} does not match {len(names)} names")

    parts: list[np.ndarray] = []
    index = 0
    while index < len(names):
        group = _rotation_group_at(names, index)
        if group is None:
            parts.append(values[:, index : index + 1])
            index += 1
            continue

        source_format, _, width = group
        source_values = values[:, index : index + width]
        if source_format == target_rotation_format:
            parts.append(source_values)
        else:
            converted = np.stack(
                [
                    convert_rotation_values(rotation, source_format, target_rotation_format)
                    for rotation in source_values
                ]
            )
            parts.append(converted)
        index += width

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
    target_rotation_format: str,
    input_rotation_format: str,
) -> tuple[dict, dict[str, list[str]]]:
    features = dict(info["features"])
    source_names: dict[str, list[str]] = {}
    found_feature = False
    for column in columns:
        if column not in features:
            logging.info("Skipping missing feature %s", column)
            continue
        found_feature = True
        feature = dict(features[column])
        names = list(feature.get("names") or [])
        if not names:
            raise ValueError(f"Feature {column} has no names; cannot locate pose rotation fields")
        source_format = _validate_detected_input_format(names, input_rotation_format)
        if source_format is None:
            raise ValueError(f"Feature {column} does not contain recognized pose rotation fields")
        if source_format == target_rotation_format:
            logging.info(
                "Keeping feature %s unchanged; already uses %s rotations",
                column,
                target_rotation_format,
            )
            continue

        new_names = rotation_names_from_pose_names(names, target_rotation_format)
        feature["names"] = new_names
        feature["shape"] = (len(new_names),)
        features[column] = feature
        source_names[column] = names
    if not found_feature:
        raise ValueError("None of the requested columns were found in the dataset")
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
    target_rotation_format: str,
    input_rotation_format: str,
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
            converted = convert_pose_batch(
                values,
                names,
                target_rotation_format=target_rotation_format,
                input_rotation_format=input_rotation_format,
            )
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
    euler_format: str | None = None,
    target_rotation_format: str | None = None,
    input_rotation_format: str = ROTATION_FORMAT_AUTO,
    overwrite: bool = False,
    push_to_hub: bool = False,
) -> Path:
    target_rotation_format = _resolve_target_rotation_format(euler_format, target_rotation_format)
    src_root = root / src_repo_id
    dst_root = root / dst_repo_id

    _copy_dataset_tree(src_root, dst_root, overwrite=overwrite)

    info = load_info(dst_root)
    new_info, source_names = _converted_features(
        info,
        columns,
        target_rotation_format=target_rotation_format,
        input_rotation_format=input_rotation_format,
    )
    write_info(new_info, dst_root)

    if source_names:
        all_arrays, episode_arrays = _rewrite_data_files(
            dst_root,
            new_info["features"],
            source_names,
            target_rotation_format=target_rotation_format,
            input_rotation_format=input_rotation_format,
        )
        dataset_stats = _stats_for_arrays(all_arrays)
        episode_stats = {ep_idx: _stats_for_arrays(arrays) for ep_idx, arrays in episode_arrays.items()}

        _rewrite_dataset_stats(dst_root, dataset_stats)
        _rewrite_episode_stats(dst_root, episode_stats)
    else:
        logging.info(
            "All requested pose columns already use %s rotations; no parquet rewrite needed",
            target_rotation_format,
        )

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
    parser.add_argument(
        "--input-rotation-format",
        choices=INPUT_ROTATION_FORMATS,
        default=ROTATION_FORMAT_AUTO,
    )
    parser.add_argument("--target-rotation-format", choices=TARGET_ROTATION_FORMATS, default=None)
    parser.add_argument("--euler-format", choices=EULER_FORMATS, default=None)
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
        target_rotation_format=args.target_rotation_format,
        input_rotation_format=args.input_rotation_format,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
    )
    logging.info("Converted dataset written to %s", output_root)


if __name__ == "__main__":
    main()
