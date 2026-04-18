#!/usr/bin/env python3
"""Find contiguous static state/action runs in a local LeRobot v3 dataset.

The script is read-only with respect to the source dataset. It writes a report
directory containing JSONL/CSV run summaries and, when videos are available,
short preview clips for quick manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_DATASET_DIR = (
    Path.home()
    / ".cache/huggingface/lerobot/ruanafan/evo-rl-data-pnp-vr-ee-pose-round0-0418-1"
)


@dataclass(frozen=True)
class EpisodeVideo:
    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find contiguous static observation.state/action runs in a local LeRobot v3 dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--min-run-frames",
        type=int,
        default=3,
        help="Only report runs with at least this many consecutive static frames.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("static_runs_report"))
    parser.add_argument("--preview-camera", default="observation.images.right_front")
    parser.add_argument(
        "--preview-count",
        type=int,
        default=30,
        help="Number of detected runs to include in the HTML inspection page.",
    )
    parser.add_argument(
        "--preview-pad-frames",
        type=int,
        default=15,
        help="Frames of context to include before and after each preview clip.",
    )
    parser.add_argument("--preview-seed", type=int, default=0)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip video previews and only write JSONL/CSV/summary outputs.",
    )
    return parser.parse_args()


def load_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata file: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_data_files(dataset_dir: Path) -> list[Path]:
    files = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_dir / 'data'}")
    return files


def read_data_table(dataset_dir: Path, columns: list[str]) -> pa.Table:
    files = find_data_files(dataset_dir)
    available = set()
    for path in files:
        available.update(pq.ParquetFile(path).schema_arrow.names)

    missing = [col for col in columns if col not in available]
    if missing:
        available_str = ", ".join(sorted(available))
        raise ValueError(f"Missing required parquet columns {missing}. Available columns: {available_str}")

    tables = [pq.read_table(path, columns=columns) for path in files]
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options="default")


def vector_column_to_numpy(table: pa.Table, key: str) -> np.ndarray:
    values = table[key].to_pylist()
    if not values:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def scalar_column_to_numpy(table: pa.Table, key: str, dtype: Any) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=dtype)


def iter_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if bool(value) and start is None:
            start = i
        elif not bool(value) and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def detect_static_runs(
    table: pa.Table,
    *,
    state_key: str,
    action_key: str,
    tol: float,
    min_run_frames: int,
    fps: float,
) -> list[dict[str, Any]]:
    states = vector_column_to_numpy(table, state_key)
    actions = vector_column_to_numpy(table, action_key)
    episode_indices = scalar_column_to_numpy(table, "episode_index", np.int64)
    frame_indices = scalar_column_to_numpy(table, "frame_index", np.int64)
    timestamps = scalar_column_to_numpy(table, "timestamp", np.float64)
    global_indices = scalar_column_to_numpy(table, "index", np.int64)

    order = np.lexsort((frame_indices, episode_indices))
    states = states[order]
    actions = actions[order]
    episode_indices = episode_indices[order]
    frame_indices = frame_indices[order]
    timestamps = timestamps[order]
    global_indices = global_indices[order]

    runs: list[dict[str, Any]] = []
    for episode_index in np.unique(episode_indices):
        ep_positions = np.flatnonzero(episode_indices == episode_index)
        if len(ep_positions) < 2:
            continue

        ep_states = states[ep_positions]
        ep_actions = actions[ep_positions]
        diffs = np.concatenate(
            [
                np.abs(np.diff(ep_states, axis=0)),
                np.abs(np.diff(ep_actions, axis=0)),
            ],
            axis=1,
        )
        pair_max_delta = np.max(diffs, axis=1)
        static_current_frame = pair_max_delta <= tol

        for pair_start, pair_end in iter_true_runs(static_current_frame):
            start_pos = pair_start + 1
            end_pos = pair_end + 1
            num_static_frames = end_pos - start_pos + 1
            if num_static_frames < min_run_frames:
                continue

            run_frame_indices = frame_indices[ep_positions[start_pos : end_pos + 1]].astype(int).tolist()
            run_global_indices = global_indices[ep_positions[start_pos : end_pos + 1]].astype(int).tolist()
            start_timestamp = float(timestamps[ep_positions[start_pos]])
            end_timestamp = float(timestamps[ep_positions[end_pos]])
            run_max_delta = float(np.max(pair_max_delta[pair_start : pair_end + 1]))
            run = {
                "episode_index": int(episode_index),
                "start_frame_index": int(frame_indices[ep_positions[start_pos]]),
                "end_frame_index": int(frame_indices[ep_positions[end_pos]]),
                "frame_indices": run_frame_indices,
                "global_indices": run_global_indices,
                "num_static_frames": int(num_static_frames),
                "duration_s": float(num_static_frames / fps),
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "anchor_frame_index": int(frame_indices[ep_positions[start_pos - 1]]),
                "anchor_global_index": int(global_indices[ep_positions[start_pos - 1]]),
                "max_delta": run_max_delta,
            }
            runs.append(run)

    runs.sort(key=lambda x: (x["episode_index"], x["start_frame_index"]))
    return runs


def load_episode_videos(dataset_dir: Path, camera_key: str) -> dict[int, EpisodeVideo]:
    episode_files = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        return {}

    columns = [
        "episode_index",
        f"videos/{camera_key}/chunk_index",
        f"videos/{camera_key}/file_index",
        f"videos/{camera_key}/from_timestamp",
        f"videos/{camera_key}/to_timestamp",
    ]
    tables = []
    for path in episode_files:
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if all(column in names for column in columns):
            tables.append(pq.read_table(path, columns=columns))
    if not tables:
        return {}

    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
    episode_indices = scalar_column_to_numpy(table, "episode_index", np.int64)
    chunk_indices = scalar_column_to_numpy(table, f"videos/{camera_key}/chunk_index", np.int64)
    file_indices = scalar_column_to_numpy(table, f"videos/{camera_key}/file_index", np.int64)
    from_timestamps = scalar_column_to_numpy(table, f"videos/{camera_key}/from_timestamp", np.float64)
    to_timestamps = scalar_column_to_numpy(table, f"videos/{camera_key}/to_timestamp", np.float64)

    return {
        int(ep): EpisodeVideo(int(chunk), int(file), float(start), float(end))
        for ep, chunk, file, start, end in zip(
            episode_indices, chunk_indices, file_indices, from_timestamps, to_timestamps, strict=True
        )
    }


def choose_preview_runs(
    runs: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if count <= 0 or not runs:
        return []
    if len(runs) <= count:
        return list(runs)

    longest_count = max(1, count // 2)
    longest = sorted(runs, key=lambda x: x["num_static_frames"], reverse=True)[:longest_count]
    longest_ids = {id(run) for run in longest}
    remaining = [run for run in runs if id(run) not in longest_ids]

    rng = random.Random(seed)
    sampled = rng.sample(remaining, k=min(count - len(longest), len(remaining)))
    selected = longest + sampled
    return sorted(selected, key=lambda x: (x["episode_index"], x["start_frame_index"]))


def camera_video_path(dataset_dir: Path, camera_key: str, episode_video: EpisodeVideo) -> Path:
    return (
        dataset_dir
        / "videos"
        / camera_key
        / f"chunk-{episode_video.chunk_index:03d}"
        / f"file-{episode_video.file_index:03d}.mp4"
    )


def make_preview_clip(
    *,
    dataset_dir: Path,
    camera_key: str,
    episode_video: EpisodeVideo,
    run: dict[str, Any],
    fps: float,
    pad_frames: int,
    output_path: Path,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False

    source = camera_video_path(dataset_dir, camera_key, episode_video)
    if not source.exists():
        return False

    # Video files use cumulative dataset time; frame_index/timestamp are episode-local.
    start_s = episode_video.from_timestamp + max(0, run["start_frame_index"] - pad_frames) / fps
    end_s = episode_video.from_timestamp + (run["end_frame_index"] + pad_frames + 1) / fps
    start_s = max(episode_video.from_timestamp, start_s)
    end_s = min(episode_video.to_timestamp, end_s)
    if end_s <= start_s:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.6f}",
        "-to",
        f"{end_s:.6f}",
        "-i",
        str(source),
        "-vf",
        (
            "scale=640:-2,"
            f"drawtext=text='ep {run['episode_index']} frames "
            f"{run['start_frame_index']}-{run['end_frame_index']}':"
            "x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55"
        ),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        str(output_path),
    ]
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0 and output_path.exists()


def attach_previews(
    *,
    dataset_dir: Path,
    camera_key: str,
    runs: list[dict[str, Any]],
    fps: float,
    preview_count: int,
    pad_frames: int,
    seed: int,
    inspection_dir: Path,
) -> list[dict[str, Any]]:
    episode_videos = load_episode_videos(dataset_dir, camera_key)
    selected = choose_preview_runs(runs, count=preview_count, seed=seed)
    preview_runs: list[dict[str, Any]] = []
    for i, run in enumerate(selected):
        preview_run = dict(run)
        episode_video = episode_videos.get(run["episode_index"])
        if episode_video is not None:
            clip_path = inspection_dir / "previews" / (
                f"run_{i:03d}_ep{run['episode_index']:04d}_"
                f"{run['start_frame_index']:06d}-{run['end_frame_index']:06d}.mp4"
            )
            if make_preview_clip(
                dataset_dir=dataset_dir,
                camera_key=camera_key,
                episode_video=episode_video,
                run=run,
                fps=fps,
                pad_frames=pad_frames,
                output_path=clip_path,
            ):
                preview_run["preview_path"] = str(clip_path.relative_to(inspection_dir))
        preview_runs.append(preview_run)
    return preview_runs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_index",
        "start_frame_index",
        "end_frame_index",
        "frame_indices",
        "num_static_frames",
        "duration_s",
        "start_timestamp",
        "end_timestamp",
        "anchor_frame_index",
        "max_delta",
        "preview_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["frame_indices"] = json.dumps(out["frame_indices"], ensure_ascii=False)
            writer.writerow(out)


def build_summary(
    *,
    info: dict[str, Any],
    runs: list[dict[str, Any]],
    tol: float,
    min_run_frames: int,
    state_key: str,
    action_key: str,
    preview_camera: str,
) -> dict[str, Any]:
    def compact_run(run: dict[str, Any] | None) -> dict[str, Any] | None:
        if run is None:
            return None
        return {
            "episode_index": run["episode_index"],
            "start_frame_index": run["start_frame_index"],
            "end_frame_index": run["end_frame_index"],
            "num_static_frames": run["num_static_frames"],
            "duration_s": run["duration_s"],
            "anchor_frame_index": run["anchor_frame_index"],
            "max_delta": run["max_delta"],
        }

    frames_by_episode: dict[int, int] = {}
    longest_by_episode: dict[int, dict[str, Any]] = {}
    for run in runs:
        episode_index = int(run["episode_index"])
        frames_by_episode[episode_index] = frames_by_episode.get(episode_index, 0) + int(
            run["num_static_frames"]
        )
        current = longest_by_episode.get(episode_index)
        if current is None or run["num_static_frames"] > current["num_static_frames"]:
            longest_by_episode[episode_index] = {
                "start_frame_index": run["start_frame_index"],
                "end_frame_index": run["end_frame_index"],
                "num_static_frames": run["num_static_frames"],
                "duration_s": run["duration_s"],
            }

    total_frames = int(info.get("total_frames", 0))
    detected_frames = sum(int(run["num_static_frames"]) for run in runs)
    return {
        "dataset": {
            "codebase_version": info.get("codebase_version"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": total_frames,
        },
        "detection": {
            "state_key": state_key,
            "action_key": action_key,
            "tol": tol,
            "min_run_frames": min_run_frames,
            "preview_camera": preview_camera,
        },
        "result": {
            "num_runs": len(runs),
            "num_static_frames_in_runs": detected_frames,
            "static_frame_ratio": detected_frames / total_frames if total_frames else None,
            "longest_run": compact_run(max(runs, key=lambda x: x["num_static_frames"], default=None)),
            "static_frames_by_episode": {str(k): v for k, v in sorted(frames_by_episode.items())},
            "longest_run_by_episode": {
                str(k): v for k, v in sorted(longest_by_episode.items())
            },
        },
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_inspection_html(
    *,
    path: Path,
    runs: list[dict[str, Any]],
    summary: dict[str, Any],
    camera_key: str,
    preview_pad_frames: int,
) -> None:
    cards = []
    for i, run in enumerate(runs):
        frame_indices = ", ".join(str(v) for v in run["frame_indices"])
        preview_path = run.get("preview_path")
        if preview_path:
            media = (
                f'<video src="{html.escape(preview_path)}" controls preload="metadata"></video>'
            )
        else:
            media = '<p class="missing">No preview clip generated for this run.</p>'

        cards.append(
            f"""
            <article class="run">
              <h2>#{i + 1} episode {run['episode_index']} · frames {run['start_frame_index']} - {run['end_frame_index']}</h2>
              {media}
              <dl>
                <dt>Static frames</dt><dd>{run['num_static_frames']}</dd>
                <dt>Duration</dt><dd>{run['duration_s']:.3f}s</dd>
                <dt>Max delta</dt><dd>{run['max_delta']:.8g}</dd>
                <dt>Anchor frame</dt><dd>{run['anchor_frame_index']}</dd>
                <dt>Frame indices</dt><dd class="frames">{html.escape(frame_indices)}</dd>
              </dl>
            </article>
            """
        )

    summary_result = summary["result"]
    body = "\n".join(cards) if cards else "<p>No runs selected for preview.</p>"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Static Run Inspection</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #111; background: #fafafa; }}
    header {{ margin-bottom: 24px; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; }}
    .meta div, .run {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    .run {{ margin: 18px 0; }}
    video {{ width: min(960px, 100%); display: block; background: black; margin: 8px 0 12px; }}
    h1 {{ margin: 0 0 12px; }}
    h2 {{ margin: 0 0 8px; font-size: 18px; }}
    dl {{ display: grid; grid-template-columns: 140px 1fr; gap: 6px 12px; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; }}
    .frames {{ overflow-wrap: anywhere; font-family: monospace; }}
    .missing {{ color: #8a1f11; }}
  </style>
</head>
<body>
  <header>
    <h1>Static Run Inspection</h1>
    <div class="meta">
      <div>Runs: {summary_result['num_runs']}</div>
      <div>Static frames in runs: {summary_result['num_static_frames_in_runs']}</div>
      <div>Camera: {html.escape(camera_key)}</div>
      <div>Preview context: {preview_pad_frames} frames each side</div>
    </div>
  </header>
  {body}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.min_run_frames < 1:
        raise ValueError("--min-run-frames must be >= 1")
    if args.tol < 0:
        raise ValueError("--tol must be >= 0")

    info = load_info(dataset_dir)
    fps = float(info.get("fps", 30))
    columns = [
        args.action_key,
        args.state_key,
        "episode_index",
        "frame_index",
        "timestamp",
        "index",
    ]
    table = read_data_table(dataset_dir, columns=columns)
    runs = detect_static_runs(
        table,
        state_key=args.state_key,
        action_key=args.action_key,
        tol=args.tol,
        min_run_frames=args.min_run_frames,
        fps=fps,
    )

    summary = build_summary(
        info=info,
        runs=runs,
        tol=args.tol,
        min_run_frames=args.min_run_frames,
        state_key=args.state_key,
        action_key=args.action_key,
        preview_camera=args.preview_camera,
    )

    preview_runs: list[dict[str, Any]] = []
    if not args.no_preview and runs and args.preview_count > 0:
        inspection_dir = output_dir / "inspection"
        preview_runs = attach_previews(
            dataset_dir=dataset_dir,
            camera_key=args.preview_camera,
            runs=runs,
            fps=fps,
            preview_count=args.preview_count,
            pad_frames=args.preview_pad_frames,
            seed=args.preview_seed,
            inspection_dir=inspection_dir,
        )
        preview_by_key = {
            (
                run["episode_index"],
                run["start_frame_index"],
                run["end_frame_index"],
            ): run.get("preview_path")
            for run in preview_runs
        }
        for run in runs:
            preview_path = preview_by_key.get(
                (run["episode_index"], run["start_frame_index"], run["end_frame_index"])
            )
            if preview_path:
                run["preview_path"] = str(Path("inspection") / preview_path)

        write_inspection_html(
            path=inspection_dir / "index.html",
            runs=preview_runs,
            summary=summary,
            camera_key=args.preview_camera,
            preview_pad_frames=args.preview_pad_frames,
        )

    write_jsonl(output_dir / "static_runs.jsonl", runs)
    write_csv(output_dir / "static_runs.csv", runs)
    write_summary(output_dir / "summary.json", summary)

    print(f"Dataset: {dataset_dir}")
    print(f"Detected runs: {len(runs)}")
    print(f"Static frames in detected runs: {summary['result']['num_static_frames_in_runs']}")
    print(f"Wrote: {output_dir / 'static_runs.jsonl'}")
    print(f"Wrote: {output_dir / 'static_runs.csv'}")
    print(f"Wrote: {output_dir / 'summary.json'}")
    inspection_index = output_dir / "inspection" / "index.html"
    if inspection_index.exists():
        print(f"Wrote: {inspection_index}")


if __name__ == "__main__":
    main()
