#!/usr/bin/env python

"""Audit and optionally repair LeRobot v3 video/episode length mismatches."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm


@dataclass(frozen=True)
class VideoIssue:
    video_key: str
    chunk_index: int
    file_index: int
    first_episode: int
    last_episode: int
    video_frames: int
    expected_frames: int
    extra_frames: int


@dataclass(frozen=True)
class EpisodeIssue:
    video_key: str
    episode_index: int
    chunk_index: int
    file_index: int
    length: int
    span_frames: int
    extra_frames: int


def _load_info(root: Path) -> dict:
    with (root / "meta" / "info.json").open() as f:
        return json.load(f)


def _video_keys(info: dict) -> list[str]:
    return [key for key, ft in info["features"].items() if ft.get("dtype") == "video"]


def _episode_files(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))


def _load_episodes(root: Path) -> pd.DataFrame:
    frames = []
    for path in _episode_files(root):
        df = pd.read_parquet(path)
        df["__meta_path"] = str(path.relative_to(root))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'meta' / 'episodes'}")
    return pd.concat(frames, ignore_index=True)


def _ffprobe_video_frames(video_path: Path) -> tuple[int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    stream = json.loads(out)["streams"][0]
    return int(stream["nb_frames"]), float(stream["duration"])


def audit_dataset(root: Path) -> tuple[list[VideoIssue], list[EpisodeIssue]]:
    info = _load_info(root)
    episodes = _load_episodes(root)
    fps = int(info["fps"])
    video_path_template = info["video_path"]
    video_issues: list[VideoIssue] = []
    episode_issues: list[EpisodeIssue] = []

    for video_key in _video_keys(info):
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        from_col = f"videos/{video_key}/from_timestamp"
        to_col = f"videos/{video_key}/to_timestamp"

        for _, row in episodes.iterrows():
            span_frames = round((float(row[to_col]) - float(row[from_col])) * fps)
            length = int(row["length"])
            extra_frames = span_frames - length
            if extra_frames != 0:
                episode_issues.append(
                    EpisodeIssue(
                        video_key=video_key,
                        episode_index=int(row["episode_index"]),
                        chunk_index=int(row[chunk_col]),
                        file_index=int(row[file_col]),
                        length=length,
                        span_frames=span_frames,
                        extra_frames=extra_frames,
                    )
                )

        group_cols = [chunk_col, file_col]
        for (chunk_idx, file_idx), group in episodes.groupby(group_cols, sort=True):
            video_path = root / video_path_template.format(
                video_key=video_key,
                chunk_index=int(chunk_idx),
                file_index=int(file_idx),
            )
            video_frames, _ = _ffprobe_video_frames(video_path)
            expected_frames = int(group["length"].sum())
            extra_frames = video_frames - expected_frames
            if extra_frames != 0:
                video_issues.append(
                    VideoIssue(
                        video_key=video_key,
                        chunk_index=int(chunk_idx),
                        file_index=int(file_idx),
                        first_episode=int(group["episode_index"].min()),
                        last_episode=int(group["episode_index"].max()),
                        video_frames=video_frames,
                        expected_frames=expected_frames,
                        extra_frames=extra_frames,
                    )
                )

    return video_issues, episode_issues


def _vcodec_from_info(info: dict, video_key: str) -> str:
    codec = info["features"][video_key].get("info", {}).get("video.codec", "")
    if codec == "av1":
        return "libsvtav1"
    if codec in {"h264", "hevc"}:
        return codec
    return "libsvtav1"


def _encode_frames_to_video(
    frames: torch.Tensor,
    output_path: Path,
    fps: int,
    vcodec: str,
    encoder: av.video.stream.VideoStream | None,
    container: av.container.OutputContainer | None,
) -> tuple[av.video.stream.VideoStream, av.container.OutputContainer]:
    if container is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(output_path), "w")
    if encoder is None:
        options = {"g": "2", "crf": "30"}
        if vcodec == "libsvtav1":
            options["preset"] = "12"
        encoder = container.add_stream(vcodec, fps, options=options)
        encoder.pix_fmt = "yuv420p"
        encoder.width = int(frames.shape[-1])
        encoder.height = int(frames.shape[-2])

    np_frames = frames.detach().cpu().numpy()
    if np_frames.shape[1] in (1, 3):
        np_frames = np.transpose(np_frames, (0, 2, 3, 1))
    if np_frames.dtype != np.uint8:
        np_frames = np.clip(np_frames, 0, 255).astype(np.uint8)

    for np_frame in np_frames:
        image = Image.fromarray(np_frame).convert("RGB")
        av_frame = av.VideoFrame.from_image(image)
        for packet in encoder.encode(av_frame):
            container.mux(packet)

    return encoder, container


def _repair_video_file(
    source_root: Path,
    output_root: Path,
    info: dict,
    episodes: pd.DataFrame,
    video_key: str,
    chunk_idx: int,
    file_idx: int,
    batch_size: int,
    vcodec: str,
) -> None:
    fps = int(info["fps"])
    video_path_template = info["video_path"]
    source_video = source_root / video_path_template.format(
        video_key=video_key,
        chunk_index=chunk_idx,
        file_index=file_idx,
    )
    output_video = output_root / video_path_template.format(
        video_key=video_key,
        chunk_index=chunk_idx,
        file_index=file_idx,
    )
    decoder = VideoDecoder(str(source_video), seek_mode="approximate")

    encoder = None
    container = None
    try:
        for _, row in episodes.sort_values("episode_index").iterrows():
            start_frame = round(float(row[f"videos/{video_key}/from_timestamp"]) * fps)
            length = int(row["length"])
            for start in range(0, length, batch_size):
                indices = list(range(start_frame + start, start_frame + min(start + batch_size, length)))
                frames = decoder.get_frames_at(indices=indices).data
                encoder, container = _encode_frames_to_video(
                    frames=frames,
                    output_path=output_video,
                    fps=fps,
                    vcodec=vcodec,
                    encoder=encoder,
                    container=container,
                )
        if container is None or encoder is None:
            raise RuntimeError(f"No frames encoded for {output_video}")
        for packet in encoder.encode():
            container.mux(packet)
    finally:
        if container is not None:
            container.close()


def repair_dataset(
    source_root: Path,
    output_root: Path,
    video_issues: list[VideoIssue],
    overwrite_output: bool,
    batch_size: int,
) -> None:
    if output_root.exists():
        if not overwrite_output:
            raise FileExistsError(f"Output directory already exists: {output_root}")
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)

    info = _load_info(source_root)
    source_episodes = _load_episodes(source_root)
    repaired_episodes = _load_episodes(output_root)
    fps = int(info["fps"])
    issue_keys = {
        (issue.video_key, issue.chunk_index, issue.file_index)
        for issue in video_issues
    }

    for video_key, chunk_idx, file_idx in tqdm(sorted(issue_keys), desc="Repair video files"):
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        group_mask = (source_episodes[chunk_col] == chunk_idx) & (source_episodes[file_col] == file_idx)
        source_group = source_episodes[group_mask]
        _repair_video_file(
            source_root=source_root,
            output_root=output_root,
            info=info,
            episodes=source_group,
            video_key=video_key,
            chunk_idx=chunk_idx,
            file_idx=file_idx,
            batch_size=batch_size,
            vcodec=_vcodec_from_info(info, video_key),
        )

        cumulative_ts = 0.0
        for row_idx in source_group.sort_values("episode_index").index:
            length = int(source_episodes.at[row_idx, "length"])
            from_ts = cumulative_ts
            to_ts = cumulative_ts + length / fps
            episode_index = int(source_episodes.at[row_idx, "episode_index"])
            repair_idx = repaired_episodes.index[repaired_episodes["episode_index"] == episode_index][0]
            repaired_episodes.at[repair_idx, f"videos/{video_key}/from_timestamp"] = from_ts
            repaired_episodes.at[repair_idx, f"videos/{video_key}/to_timestamp"] = to_ts
            cumulative_ts = to_ts

    for meta_path, group in repaired_episodes.groupby("__meta_path", sort=False):
        output_path = output_root / meta_path
        group.drop(columns=["__meta_path"]).to_parquet(output_path, index=False)


def _print_report(video_issues: list[VideoIssue], episode_issues: list[EpisodeIssue]) -> None:
    if not video_issues and not episode_issues:
        print("No video/episode length mismatches found.")
        return

    if episode_issues:
        print("[Episode span mismatches]")
        for issue in episode_issues:
            print(
                f"{issue.video_key} episode={issue.episode_index} "
                f"file={issue.chunk_index:03d}/{issue.file_index:03d} "
                f"length={issue.length} span_frames={issue.span_frames} extra_frames={issue.extra_frames}"
            )
    if video_issues:
        print("[Video file mismatches]")
        for issue in video_issues:
            print(
                f"{issue.video_key} file={issue.chunk_index:03d}/{issue.file_index:03d} "
                f"episodes={issue.first_episode}-{issue.last_episode} "
                f"video_frames={issue.video_frames} expected_frames={issue.expected_frames} "
                f"extra_frames={issue.extra_frames}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Path to a local LeRobot dataset.")
    parser.add_argument(
        "--repair-output-dir",
        type=Path,
        default=None,
        help="If set, copy the dataset here and repair mismatched video files in the copy.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Remove --repair-output-dir first if it already exists.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Frames decoded per repair batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dataset_root = args.dataset_root.expanduser().resolve()
    video_issues, episode_issues = audit_dataset(dataset_root)
    _print_report(video_issues, episode_issues)

    if args.repair_output_dir is None:
        if video_issues or episode_issues:
            raise SystemExit(1)
        return

    repair_dataset(
        source_root=dataset_root,
        output_root=args.repair_output_dir.expanduser().resolve(),
        video_issues=video_issues,
        overwrite_output=args.overwrite_output,
        batch_size=args.batch_size,
    )
    repaired_video_issues, repaired_episode_issues = audit_dataset(args.repair_output_dir.expanduser().resolve())
    _print_report(repaired_video_issues, repaired_episode_issues)
    if repaired_video_issues or repaired_episode_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
