#!/usr/bin/env python

"""Audit LeRobot dataset endpoint frames and export episode first/last images."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
import torch
from PIL import Image
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm


@dataclass(frozen=True)
class StructureIssue:
    issue_type: str
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class EndpointIssue:
    episode_index: int
    video_key: str
    issue_type: str
    diff: float
    threshold: float
    details: dict[str, Any]


def _load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    with info_path.open() as f:
        return json.load(f)


def _video_keys(info: dict) -> list[str]:
    return [key for key, ft in info["features"].items() if ft.get("dtype") == "video"]


def _load_episodes(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'meta' / 'episodes'}")
    return pd.concat(frames, ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def _count_data_rows(root: Path) -> int:
    return int(pa_ds.dataset(root / "data", format="parquet").count_rows())


def _ffprobe_video_frames(video_path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    stream = json.loads(out)["streams"][0]
    if "nb_frames" not in stream or stream["nb_frames"] in {None, "N/A"}:
        raise RuntimeError(f"ffprobe did not report nb_frames for {video_path}")
    return int(stream["nb_frames"])


def _video_path(root: Path, info: dict, row: pd.Series, video_key: str) -> Path:
    chunk_idx = int(row[f"videos/{video_key}/chunk_index"])
    file_idx = int(row[f"videos/{video_key}/file_index"])
    return root / info["video_path"].format(
        video_key=video_key,
        chunk_index=chunk_idx,
        file_index=file_idx,
    )


def _first_frame_index(row: pd.Series, video_key: str, fps: int) -> int:
    return round(float(row[f"videos/{video_key}/from_timestamp"]) * fps)


def _last_frame_index(row: pd.Series, video_key: str, fps: int) -> int:
    return _first_frame_index(row, video_key, fps) + int(row["length"]) - 1


def _frame_to_float(frame: torch.Tensor) -> torch.Tensor:
    frame = frame.float()
    return frame / 255.0 if float(frame.max()) > 1.0 else frame


def _frame_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((_frame_to_float(a) - _frame_to_float(b)).abs().mean())


def _frame_to_image(frame: torch.Tensor) -> Image.Image:
    array = frame.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


class DecoderCache:
    def __init__(self) -> None:
        self._decoders: dict[Path, VideoDecoder] = {}

    def get(self, video_path: Path) -> VideoDecoder:
        video_path = video_path.resolve()
        if video_path not in self._decoders:
            self._decoders[video_path] = VideoDecoder(str(video_path), seek_mode="approximate")
        return self._decoders[video_path]

    def frame(self, video_path: Path, frame_index: int) -> torch.Tensor:
        return self.get(video_path).get_frames_at(indices=[int(frame_index)]).data[0]


def audit_structure(root: Path, info: dict, episodes: pd.DataFrame) -> list[StructureIssue]:
    issues: list[StructureIssue] = []
    total_frames = int(info.get("total_frames", 0))
    total_episodes = int(info.get("total_episodes", 0))
    fps = int(info["fps"])

    if len(episodes) != total_episodes:
        issues.append(
            StructureIssue(
                "episode_count_mismatch",
                "meta/episodes row count does not match info.total_episodes",
                {"episodes_rows": int(len(episodes)), "info_total_episodes": total_episodes},
            )
        )

    sum_length = int(episodes["length"].sum())
    if sum_length != total_frames:
        issues.append(
            StructureIssue(
                "total_length_mismatch",
                "sum(meta/episodes.length) does not match info.total_frames",
                {"sum_length": sum_length, "info_total_frames": total_frames},
            )
        )

    data_rows = _count_data_rows(root)
    if data_rows != total_frames:
        issues.append(
            StructureIssue(
                "data_row_count_mismatch",
                "data parquet row count does not match info.total_frames",
                {"data_rows": data_rows, "info_total_frames": total_frames},
            )
        )

    span = episodes["dataset_to_index"].astype(int) - episodes["dataset_from_index"].astype(int)
    bad_span = episodes.loc[span != episodes["length"].astype(int), "episode_index"].astype(int).tolist()
    if bad_span:
        issues.append(
            StructureIssue(
                "episode_span_mismatch",
                "dataset_to_index - dataset_from_index does not equal length",
                {"episode_indices": bad_span},
            )
        )

    for video_key in _video_keys(info):
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        from_col = f"videos/{video_key}/from_timestamp"
        to_col = f"videos/{video_key}/to_timestamp"

        span_frames = ((episodes[to_col].astype(float) - episodes[from_col].astype(float)) * fps).round()
        bad_episode_span = episodes.loc[
            span_frames.astype(int) != episodes["length"].astype(int), "episode_index"
        ].astype(int).tolist()
        if bad_episode_span:
            issues.append(
                StructureIssue(
                    "video_episode_span_mismatch",
                    "video timestamp span does not equal episode length",
                    {"video_key": video_key, "episode_indices": bad_episode_span},
                )
            )

        for (chunk_idx, file_idx), group in episodes.groupby([chunk_col, file_col], sort=True):
            video_path = root / info["video_path"].format(
                video_key=video_key,
                chunk_index=int(chunk_idx),
                file_index=int(file_idx),
            )
            expected_frames = int(group["length"].sum())
            actual_frames = _ffprobe_video_frames(video_path)
            if actual_frames != expected_frames:
                issues.append(
                    StructureIssue(
                        "video_chunk_frame_count_mismatch",
                        "video chunk frame count does not equal summed episode lengths",
                        {
                            "video_key": video_key,
                            "chunk_index": int(chunk_idx),
                            "file_index": int(file_idx),
                            "actual_frames": actual_frames,
                            "expected_frames": expected_frames,
                        },
                    )
                )

    return issues


def export_endpoint_images(
    root: Path,
    info: dict,
    episodes: pd.DataFrame,
    endpoint_dir: Path,
    max_episodes: int | None,
    decoder_cache: DecoderCache,
) -> None:
    fps = int(info["fps"])
    rows = episodes if max_episodes is None else episodes.head(max_episodes)
    video_keys = _video_keys(info)

    for _, row in tqdm(list(rows.iterrows()), desc="Export endpoint frames"):
        ep_idx = int(row["episode_index"])
        ep_dir = endpoint_dir / f"episode-{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        for video_key in video_keys:
            video_path = _video_path(root, info, row, video_key)
            first_idx = _first_frame_index(row, video_key, fps)
            last_idx = _last_frame_index(row, video_key, fps)
            first_frame = decoder_cache.frame(video_path, first_idx)
            last_frame = decoder_cache.frame(video_path, last_idx)
            _frame_to_image(first_frame).save(ep_dir / f"{video_key}_first.png")
            _frame_to_image(last_frame).save(ep_dir / f"{video_key}_last.png")


def audit_endpoint_frames(
    root: Path,
    info: dict,
    episodes: pd.DataFrame,
    same_threshold: float,
    jump_threshold: float,
    max_episodes: int | None,
    decoder_cache: DecoderCache,
) -> list[EndpointIssue]:
    fps = int(info["fps"])
    issues: list[EndpointIssue] = []
    video_keys = _video_keys(info)
    rows_to_check = len(episodes) if max_episodes is None else min(len(episodes), max_episodes)

    for idx in tqdm(range(rows_to_check), desc="Audit endpoint frames"):
        row = episodes.iloc[idx]
        ep_idx = int(row["episode_index"])
        length = int(row["length"])
        if length < 2:
            issues.append(
                EndpointIssue(
                    ep_idx,
                    "*",
                    "episode_too_short",
                    0.0,
                    2.0,
                    {"length": length},
                )
            )
            continue

        for video_key in video_keys:
            video_path = _video_path(root, info, row, video_key)
            first_idx = _first_frame_index(row, video_key, fps)
            last_idx = _last_frame_index(row, video_key, fps)

            first = decoder_cache.frame(video_path, first_idx)
            second = decoder_cache.frame(video_path, first_idx + 1)
            penultimate = decoder_cache.frame(video_path, last_idx - 1)
            last = decoder_cache.frame(video_path, last_idx)

            first_second = _frame_diff(first, second)
            if first_second > jump_threshold:
                issues.append(
                    EndpointIssue(
                        ep_idx,
                        video_key,
                        "head_jump",
                        first_second,
                        jump_threshold,
                        {"first_vs_second_diff": first_second},
                    )
                )

            prev_last = _frame_diff(penultimate, last)
            if prev_last > jump_threshold:
                issues.append(
                    EndpointIssue(
                        ep_idx,
                        video_key,
                        "tail_jump",
                        prev_last,
                        jump_threshold,
                        {"prev_vs_last_diff": prev_last},
                    )
                )

            if idx > 0:
                prev_row = episodes.iloc[idx - 1]
                prev_video_path = _video_path(root, info, prev_row, video_key)
                prev_last_idx = _last_frame_index(prev_row, video_key, fps)
                previous_last = decoder_cache.frame(prev_video_path, prev_last_idx)
                prev_tail_first = _frame_diff(previous_last, first)
                if prev_tail_first < same_threshold:
                    issues.append(
                        EndpointIssue(
                            ep_idx,
                            video_key,
                            "duplicate_prev_tail",
                            prev_tail_first,
                            same_threshold,
                            {"prev_last_vs_first_diff": prev_tail_first, "previous_episode_index": ep_idx - 1},
                        )
                    )

            if idx + 1 < len(episodes):
                next_row = episodes.iloc[idx + 1]
                next_video_path = _video_path(root, info, next_row, video_key)
                next_first_idx = _first_frame_index(next_row, video_key, fps)
                next_first = decoder_cache.frame(next_video_path, next_first_idx)
                last_next = _frame_diff(last, next_first)
                if last_next < same_threshold:
                    issues.append(
                        EndpointIssue(
                            ep_idx,
                            video_key,
                            "duplicate_next_start",
                            last_next,
                            same_threshold,
                            {"last_vs_next_first_diff": last_next, "next_episode_index": ep_idx + 1},
                        )
                    )

    return issues


def _json_ready(info: dict, structure: list[StructureIssue], endpoint: list[EndpointIssue]) -> dict:
    return {
        "summary": {
            "structure_issue_count": len(structure),
            "endpoint_issue_count": len(endpoint),
            "has_issues": bool(structure or endpoint),
        },
        "structure_issues": [asdict(issue) for issue in structure],
        "endpoint_issues": [asdict(issue) for issue in endpoint],
        "config": info,
    }


def _print_report(structure_issues: list[StructureIssue], endpoint_issues: list[EndpointIssue]) -> None:
    if not structure_issues and not endpoint_issues:
        print("No structure or endpoint-frame issues found.")
        return

    if structure_issues:
        print("[Structure issues]")
        for issue in structure_issues:
            print(f"- {issue.issue_type}: {issue.message} {issue.details}")

    if endpoint_issues:
        print("[Endpoint frame issues]")
        for issue in endpoint_issues:
            print(
                f"- episode={issue.episode_index} video={issue.video_key} "
                f"type={issue.issue_type} diff={issue.diff:.6f} threshold={issue.threshold:.6f} "
                f"details={issue.details}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Path to a local LeRobot dataset.")
    parser.add_argument("--endpoint-dir", required=True, type=Path, help="Directory for endpoint PNG export.")
    parser.add_argument("--same-threshold", type=float, default=0.001)
    parser.add_argument("--jump-threshold", type=float, default=0.05)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite-endpoint-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    endpoint_dir = args.endpoint_dir.expanduser().resolve()

    if endpoint_dir.exists():
        if not args.overwrite_endpoint_dir:
            raise FileExistsError(
                f"Endpoint directory already exists: {endpoint_dir}. Use --overwrite-endpoint-dir to replace it."
            )
        shutil.rmtree(endpoint_dir)
    endpoint_dir.mkdir(parents=True, exist_ok=False)

    info = _load_info(dataset_root)
    episodes = _load_episodes(dataset_root)
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes must be positive when provided")

    decoder_cache = DecoderCache()
    structure_issues = audit_structure(dataset_root, info, episodes)
    endpoint_issues = audit_endpoint_frames(
        dataset_root,
        info,
        episodes,
        same_threshold=args.same_threshold,
        jump_threshold=args.jump_threshold,
        max_episodes=args.max_episodes,
        decoder_cache=decoder_cache,
    )
    export_endpoint_images(
        dataset_root,
        info,
        episodes,
        endpoint_dir=endpoint_dir,
        max_episodes=args.max_episodes,
        decoder_cache=decoder_cache,
    )

    report = _json_ready(info, structure_issues, endpoint_issues)
    if args.json_output is not None:
        args.json_output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.expanduser().open("w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    _print_report(structure_issues, endpoint_issues)
    raise SystemExit(1 if structure_issues or endpoint_issues else 0)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2) from exc
