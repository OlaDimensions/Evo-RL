#!/usr/bin/env python

"""Export value-function diagnostics overlays from an already annotated LeRobot dataset.

This script expects `lerobot-value-infer` to have written at least a value column
into the dataset. It does not run the value model and does not modify parquet
data; it only creates videos and a CSV summary for inspection.
"""

from __future__ import annotations

import argparse
import csv
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import encode_video_frames
from lerobot.scripts.lerobot_value_infer import (
    _binarize_advantages,
    _build_episode_info,
    _compute_dense_rewards_from_targets,
    _compute_n_step_advantages,
    _compute_task_thresholds,
)
from lerobot.scripts.value_infer_viz import (
    _build_output_video_path,
    _build_output_video_path_multiview,
    _decode_frames_at_timestamps,
    _encode_pil_to_video,
    _get_episode_video_time_bounds,
    _load_font,
    _parse_episodes_arg,
    _select_video_keys,
    _smooth_1d,
    _to_1d_float,
    _to_1d_int,
)
from lerobot.utils.recording_annotations import resolve_episode_success_label


@dataclass
class DiagnosticsArgs:
    dataset_repo_id: str
    dataset_root: str | None
    dataset_episodes: list[int] | None
    dataset_revision: str | None
    dataset_download_videos: bool
    value_field: str
    advantage_field: str
    indicator_field: str
    intervention_field: str
    success_field: str
    default_success: str
    c_fail_coef: float
    bin_min: float
    bin_max: float
    n_step: int
    positive_ratio: float
    force_intervention_positive: bool
    viz_episodes: str
    viz_video_key: str | None
    viz_video_keys: str | None
    output_dir: Path
    overwrite: bool
    smooth_window: int
    vcodec: str
    frame_storage_mode: str


def _has_lerobot_metadata(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _resolve_dataset_root_arg(repo_id: str, root: str | None) -> str | None:
    """Accept either an exact dataset root or a LeRobot cache/base directory."""
    if root is None:
        return None

    root_path = Path(root).expanduser()
    if _has_lerobot_metadata(root_path):
        return str(root_path)

    candidate = root_path / repo_id
    if _has_lerobot_metadata(candidate) or root_path.exists():
        return str(candidate)

    return str(root_path)


def _parse_episode_list(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    parsed: set[int] = set()
    for token in value.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid episode range '{part}'.")
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(part))
    return sorted(parsed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export value diagnostics videos from a LeRobot dataset annotated by lerobot-value-infer."
    )
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", required=True)
    parser.add_argument("--dataset.root", dest="dataset_root", default=None)
    parser.add_argument("--dataset.episodes", dest="dataset_episodes", default=None)
    parser.add_argument("--dataset.revision", dest="dataset_revision", default=None)
    parser.add_argument("--dataset.download_videos", dest="dataset_download_videos", default="true")

    parser.add_argument("--value_field", default="complementary_info.value")
    parser.add_argument("--advantage_field", default="complementary_info.advantage")
    parser.add_argument("--indicator_field", default="complementary_info.acp_indicator")
    parser.add_argument("--intervention_field", default="complementary_info.is_intervention")
    parser.add_argument("--success_field", default="episode_success")
    parser.add_argument("--default_success", default="failure")

    parser.add_argument("--c_fail_coef", type=float, default=1.0)
    parser.add_argument("--bin_min", type=float, default=-1.0)
    parser.add_argument("--bin_max", type=float, default=0.0)
    parser.add_argument("--n_step", type=int, default=50)
    parser.add_argument("--positive_ratio", type=float, default=0.3)
    parser.add_argument("--force_intervention_positive", default="true")

    parser.add_argument("--viz.episodes", dest="viz_episodes", default="all")
    parser.add_argument("--viz.video_key", dest="viz_video_key", default=None)
    parser.add_argument("--viz.video_keys", dest="viz_video_keys", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--frame_storage_mode", choices=("memory", "disk"), default="memory")
    return parser


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got '{raw}'.")


def _parse_args() -> DiagnosticsArgs:
    ns = _build_parser().parse_args()
    return DiagnosticsArgs(
        dataset_repo_id=ns.dataset_repo_id,
        dataset_root=ns.dataset_root,
        dataset_episodes=_parse_episode_list(ns.dataset_episodes),
        dataset_revision=ns.dataset_revision,
        dataset_download_videos=_parse_bool(ns.dataset_download_videos),
        value_field=ns.value_field,
        advantage_field=ns.advantage_field,
        indicator_field=ns.indicator_field,
        intervention_field=ns.intervention_field,
        success_field=ns.success_field,
        default_success=ns.default_success,
        c_fail_coef=ns.c_fail_coef,
        bin_min=ns.bin_min,
        bin_max=ns.bin_max,
        n_step=ns.n_step,
        positive_ratio=ns.positive_ratio,
        force_intervention_positive=_parse_bool(ns.force_intervention_positive),
        viz_episodes=ns.viz_episodes,
        viz_video_key=ns.viz_video_key,
        viz_video_keys=ns.viz_video_keys,
        output_dir=ns.output_dir,
        overwrite=ns.overwrite,
        smooth_window=ns.smooth_window,
        vcodec=ns.vcodec,
        frame_storage_mode=ns.frame_storage_mode,
    )


def _safe_column_1d(raw_dataset: Any, field: str, *, dtype: str, default: np.ndarray | None = None):
    if field not in raw_dataset.column_names:
        if default is None:
            raise KeyError(f"Missing field '{field}' in dataset.")
        return default.copy()
    if dtype == "float":
        return _to_1d_float(raw_dataset[field])
    if dtype == "int":
        return _to_1d_int(raw_dataset[field])
    raise ValueError(f"Unknown dtype kind: {dtype}")


def _resolve_episode_labels(dataset: LeRobotDataset, success_field: str, default_success: str) -> dict[int, str]:
    episodes_ds = dataset.meta.episodes.with_format(None)
    episodes = episodes_ds[:]
    has_success = success_field in episodes_ds.column_names
    labels: dict[int, str] = {}
    for i in range(len(episodes_ds)):
        ep_idx = int(episodes["episode_index"][i])
        explicit = episodes[success_field][i] if has_success else None
        labels[ep_idx] = str(
            resolve_episode_success_label(
                explicit,
                default_label=default_success,
                require_label=True,
            )
        )
    return labels


def _line_points(
    values: np.ndarray,
    current_step: int,
    rect: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = rect
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    denom_x = max(1, values.shape[0] - 1)
    denom_y = max(1e-6, y_max - y_min)
    last = min(current_step, values.shape[0] - 1)
    points: list[tuple[int, int]] = []
    for i in range(last + 1):
        x = int(round(x0 + width * (i / denom_x)))
        y_norm = np.clip((float(values[i]) - y_min) / denom_y, 0.0, 1.0)
        y = int(round(y0 + (1.0 - y_norm) * height))
        points.append((x, y))
    return points


def _format_panel_number(value: float, *, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_value_badges(
    draw: ImageDraw.ImageDraw,
    chart: tuple[int, int, int, int],
    badges: list[tuple[str, int, int, tuple[int, int, int, int], bool]],
    font,
) -> None:
    if not badges:
        return

    chart_x0, chart_y0, chart_x1, chart_y1 = chart
    label_pad_x = 4
    label_pad_y = 2
    gap = 2
    placed: list[dict[str, Any]] = []

    for text, px, py, color, prefer_left in badges:
        text_w, text_h = _text_size(draw, text, font)
        box_w = text_w + label_pad_x * 2
        box_h = text_h + label_pad_y * 2
        if prefer_left:
            x0 = max(chart_x0, px - box_w - 8)
        else:
            x0 = min(px + 8, chart_x1 - box_w)
        x0 = max(chart_x0, min(x0, chart_x1 - box_w))
        y0 = py - box_h // 2
        y0 = max(chart_y0, min(y0, chart_y1 - box_h))
        placed.append({"text": text, "x0": x0, "y0": y0, "w": box_w, "h": box_h, "color": color})

    placed.sort(key=lambda item: item["y0"])
    prev_bottom = chart_y0 - gap
    for item in placed:
        item["y0"] = max(item["y0"], prev_bottom + gap)
        prev_bottom = item["y0"] + item["h"]

    overflow = prev_bottom - chart_y1
    if overflow > 0:
        for item in placed:
            item["y0"] = max(chart_y0, item["y0"] - overflow)
        prev_bottom = chart_y0 - gap
        for item in placed:
            item["y0"] = max(item["y0"], prev_bottom + gap)
            prev_bottom = item["y0"] + item["h"]

    for item in placed:
        x0 = int(item["x0"])
        y0 = int(item["y0"])
        x1 = x0 + int(item["w"])
        y1 = y0 + int(item["h"])
        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 175), outline=item["color"], width=1)
        draw.text((x0 + label_pad_x, y0 + label_pad_y), item["text"], fill=item["color"], font=font)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, np.ndarray, tuple[int, int, int, int]]],
    current_step: int,
    *,
    threshold: float | None = None,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, fill=(0, 0, 0, 145), outline=(255, 255, 255, 70), width=1)
    font = _load_font(max(12, (y1 - y0) // 7))
    draw.text((x0 + 8, y0 + 5), title, fill=(235, 235, 235, 230), font=font)

    all_values = [values for _, values, _ in series if values.size > 0 and np.isfinite(values).any()]
    if not all_values:
        return
    finite = np.concatenate([values[np.isfinite(values)] for values in all_values])
    if threshold is not None and np.isfinite(threshold):
        finite = np.concatenate([finite, np.asarray([threshold], dtype=np.float32)])
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0
    padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding

    tick_values = (y_max, (y_min + y_max) / 2.0, y_min)
    tick_labels = [_format_panel_number(value) for value in tick_values]
    tick_label_w = max(_text_size(draw, label, font)[0] for label in tick_labels)
    chart = (x0 + 12 + tick_label_w, y0 + 28, x1 - 8, y1 - 22)
    for frac in (0.25, 0.5, 0.75):
        gy = int(round(chart[1] + (chart[3] - chart[1]) * frac))
        draw.line([(chart[0], gy), (chart[2], gy)], fill=(255, 255, 255, 25), width=1)

    tick_y_positions = (chart[1], (chart[1] + chart[3]) // 2, chart[3])
    for label, ty in zip(tick_labels, tick_y_positions, strict=True):
        text_w, text_h = _text_size(draw, label, font)
        draw.text((chart[0] - text_w - 5, ty - text_h // 2), label, fill=(210, 210, 210, 210), font=font)
        draw.line([(chart[0] - 3, ty), (chart[0], ty)], fill=(255, 255, 255, 45), width=1)

    if threshold is not None and np.isfinite(threshold):
        ty_norm = np.clip((threshold - y_min) / max(1e-6, y_max - y_min), 0.0, 1.0)
        ty = int(round(chart[1] + (1.0 - ty_norm) * (chart[3] - chart[1])))
        draw.line([(chart[0], ty), (chart[2], ty)], fill=(255, 210, 80, 160), width=1)
        threshold_text = f"thr={_format_panel_number(float(threshold), signed=True)}"
        text_w, text_h = _text_size(draw, threshold_text, font)
        tx = max(chart[0] + 2, chart[2] - text_w - 6)
        ty_text = max(chart[1], min(ty - text_h - 2, chart[3] - text_h))
        draw.rectangle((tx - 3, ty_text - 1, tx + text_w + 3, ty_text + text_h + 1), fill=(0, 0, 0, 150))
        draw.text((tx, ty_text), threshold_text, fill=(255, 210, 80, 230), font=font)

    value_badges: list[tuple[str, int, int, tuple[int, int, int, int], bool]] = []
    for label, values, color in series:
        points = _line_points(values, current_step, chart, y_min, y_max)
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        if points:
            px, py = points[-1]
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
            last = min(current_step, values.shape[0] - 1)
            if values.size > 0 and np.isfinite(values[last]):
                signed = label in {"error", "adv"}
                short_label = "err" if label == "error" else label
                value_badges.append(
                    (
                        f"{short_label}={_format_panel_number(float(values[last]), signed=signed)}",
                        px,
                        py,
                        color,
                        px > (chart[0] + chart[2]) // 2,
                    )
                )

    _draw_value_badges(draw, chart, value_badges, font)

    legend_x = x0 + 8
    legend_y = y1 - 18
    for label, _, color in series:
        draw.rectangle((legend_x, legend_y + 4, legend_x + 10, legend_y + 10), fill=color)
        draw.text((legend_x + 14, legend_y), label, fill=(230, 230, 230, 220), font=font)
        legend_x += max(74, int(draw.textlength(label, font=font)) + 28)


def _draw_indicator_bar(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    indicators: np.ndarray,
    interventions: np.ndarray,
    current_step: int,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, fill=(0, 0, 0, 145), outline=(255, 255, 255, 70), width=1)
    n = indicators.shape[0]
    if n == 0:
        return
    width = max(1, x1 - x0)
    for i in range(n):
        left = int(round(x0 + width * (i / n)))
        right = int(round(x0 + width * ((i + 1) / n)))
        if right <= left:
            right = left + 1
        color = (70, 210, 120, 210) if int(indicators[i]) == 1 else (120, 120, 120, 150)
        draw.rectangle((left, y0, right, y1), fill=color)
        if i < interventions.shape[0] and float(interventions[i]) > 0.5:
            draw.line([(left, y0), (left, y1)], fill=(255, 220, 80, 240), width=2)
    cursor_x = int(round(x0 + width * (min(current_step, n - 1) / max(1, n - 1))))
    draw.line([(cursor_x, y0 - 4), (cursor_x, y1 + 4)], fill=(255, 255, 255, 240), width=2)


def _draw_diagnostics_overlay(
    frame: Image.Image,
    *,
    current_step: int,
    ep_idx: int,
    frame_indices: np.ndarray,
    success_label: str,
    task_index: int,
    values: np.ndarray,
    targets: np.ndarray,
    errors: np.ndarray,
    advantages: np.ndarray,
    indicators: np.ndarray,
    interventions: np.ndarray,
    threshold: float | None,
    episode_mae: float,
) -> Image.Image:
    rgba = frame.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = rgba.size
    margin = max(10, width // 110)
    panel_h = max(96, height // 5)
    gap = max(8, height // 100)
    value_panel = (margin, margin, width - margin, margin + panel_h)
    adv_panel = (margin, value_panel[3] + gap, width - margin, value_panel[3] + gap + panel_h)
    bar_h = max(16, height // 38)
    bar = (margin, height - margin - bar_h, width - margin, height - margin)

    _draw_panel(
        draw,
        value_panel,
        "value fit",
        [
            ("pred", values, (100, 200, 255, 230)),
            ("target", targets, (255, 210, 90, 230)),
            ("error", errors, (255, 120, 120, 210)),
        ],
        current_step,
    )
    _draw_panel(
        draw,
        adv_panel,
        "advantage",
        [("adv", advantages, (150, 230, 140, 230))],
        current_step,
        threshold=threshold,
    )
    _draw_indicator_bar(draw, bar, indicators, interventions, current_step)

    i = min(current_step, values.shape[0] - 1)
    text_lines = [
        f"episode={ep_idx} frame={int(frame_indices[i])} success={success_label} task={task_index}",
        f"value={float(values[i]): .4f} target={float(targets[i]): .4f} error={float(errors[i]): .4f} mae={episode_mae:.4f}",
        f"adv={float(advantages[i]): .4f} threshold={threshold if threshold is not None else float('nan'): .4f} acp={int(indicators[i])} intervention={float(interventions[i]) > 0.5}",
    ]
    font = _load_font(max(13, height // 42))
    line_h = max(16, font.size + 3 if hasattr(font, "size") else height // 38)
    text_w = max(int(draw.textlength(line, font=font)) for line in text_lines)
    x0 = margin
    y0 = adv_panel[3] + gap
    draw.rectangle(
        (x0, y0, x0 + text_w + 12, y0 + line_h * len(text_lines) + 8),
        fill=(0, 0, 0, 155),
    )
    for row, line in enumerate(text_lines):
        draw.text((x0 + 6, y0 + 4 + row * line_h), line, fill=(245, 245, 245, 235), font=font)

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _episode_summary_row(
    ep_idx: int,
    success_label: str,
    task_index: int,
    values: np.ndarray,
    targets: np.ndarray,
    advantages: np.ndarray,
    indicators: np.ndarray,
    interventions: np.ndarray,
) -> dict[str, Any]:
    errors = values - targets
    x = np.arange(values.shape[0], dtype=np.float32)
    value_slope = float(np.polyfit(x, values.astype(np.float32), 1)[0]) if values.shape[0] > 1 else 0.0
    return {
        "episode_index": ep_idx,
        "success_label": success_label,
        "task_index": task_index,
        "num_frames": int(values.shape[0]),
        "positive_ratio": float(np.mean(indicators.astype(np.float32))) if indicators.size else 0.0,
        "intervention_ratio": float(np.mean((interventions > 0.5).astype(np.float32)))
        if interventions.size
        else 0.0,
        "value_mae": float(np.mean(np.abs(errors))) if errors.size else 0.0,
        "value_error_mean": float(np.mean(errors)) if errors.size else 0.0,
        "value_error_std": float(np.std(errors)) if errors.size else 0.0,
        "value_slope": value_slope,
        "advantage_mean": float(np.mean(advantages)) if advantages.size else 0.0,
        "advantage_std": float(np.std(advantages)) if advantages.size else 0.0,
    }


def _write_summary_csv(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "diagnostics_summary.csv"
    fieldnames = [
        "episode_index",
        "success_label",
        "task_index",
        "num_frames",
        "positive_ratio",
        "intervention_ratio",
        "value_mae",
        "value_error_mean",
        "value_error_std",
        "value_slope",
        "advantage_mean",
        "advantage_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _export_episode_video(
    *,
    dataset: LeRobotDataset,
    src_video_paths: list[Path],
    dst_video_path: Path,
    camera_labels: list[str],
    ep_idx: int,
    frame_indices: np.ndarray,
    timestamps_per_cam: list[np.ndarray],
    success_label: str,
    task_index: int,
    values: np.ndarray,
    targets: np.ndarray,
    advantages: np.ndarray,
    indicators: np.ndarray,
    interventions: np.ndarray,
    threshold: float | None,
    overwrite: bool,
    vcodec: str,
    frame_storage_mode: str,
    output_dir: Path,
    smooth_window: int,
) -> Path | None:
    if dst_video_path.exists() and not overwrite:
        logging.info("Skip existing output: %s", dst_video_path)
        return None

    values = _smooth_1d(values, smooth_window)
    targets = _smooth_1d(targets, smooth_window)
    advantages = _smooth_1d(advantages, smooth_window)
    errors = values - targets
    episode_mae = float(np.mean(np.abs(errors))) if errors.size else 0.0

    all_cam_frames: list[list[Image.Image]] = []
    for src_path, timestamps in zip(src_video_paths, timestamps_per_cam, strict=True):
        frames = _decode_frames_at_timestamps(
            video_file=src_path,
            timestamps_s=timestamps,
            tolerance_s=float(getattr(dataset, "tolerance_s", 1e-4)),
            backend=getattr(dataset, "video_backend", None),
        )
        all_cam_frames.append(frames)

    n_frames = min(len(frames) for frames in all_cam_frames)
    n_frames = min(n_frames, values.shape[0])
    if n_frames == 0:
        logging.warning("No frames decoded for episode %s, skipping.", ep_idx)
        return None

    single_w = all_cam_frames[0][0].width
    single_h = all_cam_frames[0][0].height
    total_w = single_w * len(all_cam_frames)
    label_font = _load_font(max(14, single_h // 32))

    def compose_frame(i: int) -> Image.Image:
        if len(all_cam_frames) == 1:
            base = all_cam_frames[0][i]
        else:
            base = Image.new("RGB", (total_w, single_h))
            label_draw = ImageDraw.Draw(base)
            for cam_idx, frames in enumerate(all_cam_frames):
                base.paste(frames[i].resize((single_w, single_h)), (cam_idx * single_w, 0))
                label = camera_labels[cam_idx].split(".")[-1]
                lx = cam_idx * single_w + 8
                ly = 4
                bbox = label_draw.textbbox((lx, ly), label, font=label_font)
                label_draw.rectangle(
                    (bbox[0] - 2, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
                    fill=(0, 0, 0),
                )
                label_draw.text((lx, ly), label, fill=(255, 255, 210), font=label_font)

        return _draw_diagnostics_overlay(
            base,
            current_step=i,
            ep_idx=ep_idx,
            frame_indices=frame_indices,
            success_label=success_label,
            task_index=task_index,
            values=values,
            targets=targets,
            errors=errors,
            advantages=advantages,
            indicators=indicators,
            interventions=interventions,
            threshold=threshold,
            episode_mae=episode_mae,
        )

    dst_video_path.parent.mkdir(parents=True, exist_ok=True)
    if frame_storage_mode == "disk":
        with tempfile.TemporaryDirectory(dir=str(output_dir), prefix=f"{dst_video_path.stem}-frames-") as tmp:
            tmp_path = Path(tmp)
            for i in range(n_frames):
                compose_frame(i).save(tmp_path / f"frame-{i:06d}.png")
            encode_video_frames(
                imgs_dir=tmp_path,
                video_path=dst_video_path,
                fps=int(dataset.fps),
                vcodec=vcodec,
                overwrite=True,
            )
    else:
        _encode_pil_to_video([compose_frame(i) for i in range(n_frames)], dst_video_path, int(dataset.fps), vcodec)

    return dst_video_path


def _build_diagnostics_arrays(args: DiagnosticsArgs, dataset: LeRobotDataset) -> dict[str, Any]:
    raw = dataset.hf_dataset.with_format(None)
    columns = set(raw.column_names)
    if args.value_field not in columns:
        raise KeyError(
            f"Missing value field '{args.value_field}'. Run lerobot-value-infer first or pass --value_field."
        )

    values = _to_1d_float(raw[args.value_field])
    episode_indices = np.asarray(raw["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices = np.asarray(raw["frame_index"], dtype=np.int64).reshape(-1)
    task_indices = np.asarray(raw["task_index"], dtype=np.int64).reshape(-1)
    timestamps = (
        np.asarray(raw["timestamp"], dtype=np.float64).reshape(-1)
        if "timestamp" in columns
        else frame_indices.astype(np.float64) / float(dataset.fps)
    )

    interventions = _safe_column_1d(
        raw,
        args.intervention_field,
        dtype="float",
        default=np.zeros(values.shape[0], dtype=np.float32),
    )

    episode_info, task_max_lengths = _build_episode_info(
        dataset=dataset,
        success_field=args.success_field,
        default_success=args.default_success,
    )
    targets = _compute_value_targets_for_diagnostics(
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        task_max_lengths=task_max_lengths,
        c_fail_coef=args.c_fail_coef,
        bin_min=args.bin_min,
        bin_max=args.bin_max,
    )
    rewards = _compute_dense_rewards_from_targets(targets, episode_indices, frame_indices)

    if args.advantage_field in columns:
        advantages = _to_1d_float(raw[args.advantage_field])
    else:
        advantages = _compute_n_step_advantages(
            rewards=rewards,
            values=values,
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            n_step=args.n_step,
        )

    thresholds = _compute_task_thresholds(
        task_indices=task_indices,
        advantages=advantages,
        positive_ratio=args.positive_ratio,
    )

    if args.indicator_field in columns:
        indicators = _to_1d_int(raw[args.indicator_field])
    else:
        indicators = _binarize_advantages(
            task_indices=task_indices,
            advantages=advantages,
            thresholds=thresholds,
            interventions=interventions,
            force_intervention_positive=args.force_intervention_positive,
        )

    return {
        "values": values,
        "targets": targets,
        "advantages": advantages,
        "indicators": indicators,
        "interventions": interventions,
        "episode_indices": episode_indices,
        "frame_indices": frame_indices,
        "task_indices": task_indices,
        "timestamps": timestamps,
        "thresholds": thresholds,
        "success_labels": _resolve_episode_labels(dataset, args.success_field, args.default_success),
    }


def _compute_value_targets_for_diagnostics(
    *,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    episode_info,
    task_max_lengths: dict[int, int],
    c_fail_coef: float,
    bin_min: float,
    bin_max: float,
) -> np.ndarray:
    from lerobot.values.pistar06.modeling_pistar06 import compute_normalized_value_targets

    return compute_normalized_value_targets(
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        task_max_lengths=task_max_lengths,
        c_fail_coef=c_fail_coef,
        clip_min=bin_min,
        clip_max=bin_max,
    )


def export_diagnostics(args: DiagnosticsArgs) -> list[Path]:
    logging.info("Loading dataset %s", args.dataset_repo_id)
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=_resolve_dataset_root_arg(args.dataset_repo_id, args.dataset_root),
        episodes=args.dataset_episodes,
        revision=args.dataset_revision,
        download_videos=args.dataset_download_videos,
    )
    arrays = _build_diagnostics_arrays(args, dataset)

    selected_video_keys = _select_video_keys(
        camera_keys=list(dataset.meta.camera_keys),
        requested_video_keys=args.viz_video_keys,
        requested_video_key=args.viz_video_key,
    )
    multiview = len(selected_video_keys) > 1

    if dataset.episodes is not None:
        available_episodes = sorted(dataset.episodes)
    else:
        available_episodes = list(range(dataset.meta.total_episodes))

    if args.viz_episodes.strip().lower() == "all":
        episodes = available_episodes
    else:
        requested = _parse_episodes_arg(args.viz_episodes, dataset.meta.total_episodes)
        available = set(available_episodes)
        episodes = [ep for ep in requested if ep in available]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    written: list[Path] = []

    for ep in tqdm(episodes, desc="Export value diagnostics", leave=False):
        positions = np.flatnonzero(arrays["episode_indices"] == ep)
        if positions.shape[0] == 0:
            continue
        ep_frame_indices = arrays["frame_indices"][positions]
        if bool(np.any(np.diff(ep_frame_indices) < 0)):
            positions = positions[np.argsort(ep_frame_indices, kind="stable")]
            ep_frame_indices = arrays["frame_indices"][positions]

        task_index = int(arrays["task_indices"][positions[0]])
        values = arrays["values"][positions]
        targets = arrays["targets"][positions]
        advantages = arrays["advantages"][positions]
        indicators = arrays["indicators"][positions]
        interventions = arrays["interventions"][positions]
        success_label = arrays["success_labels"].get(int(ep), "unknown")
        threshold = arrays["thresholds"].get(task_index)

        rows.append(
            _episode_summary_row(
                ep_idx=int(ep),
                success_label=success_label,
                task_index=task_index,
                values=values,
                targets=targets,
                advantages=advantages,
                indicators=indicators,
                interventions=interventions,
            )
        )

        ep_timestamps = arrays["timestamps"][positions]
        src_paths: list[Path] = []
        ts_per_cam: list[np.ndarray] = []
        for cam_key in selected_video_keys:
            src_path = Path(dataset.root) / dataset.meta.get_video_file_path(int(ep), cam_key)
            from_ts, to_ts = _get_episode_video_time_bounds(dataset, int(ep), cam_key)
            cam_ts = from_ts + ep_timestamps
            if to_ts is not None:
                cam_ts = np.minimum(cam_ts, to_ts)
            src_paths.append(src_path)
            ts_per_cam.append(cam_ts)

        if multiview:
            dst_path = _build_output_video_path_multiview(
                output_dir=args.output_dir,
                repo_id=dataset.repo_id,
                video_keys=selected_video_keys,
                episode_index=int(ep),
            )
        else:
            dst_path = _build_output_video_path(
                output_dir=args.output_dir,
                repo_id=dataset.repo_id,
                video_key=selected_video_keys[0],
                episode_index=int(ep),
            )

        out = _export_episode_video(
            dataset=dataset,
            src_video_paths=src_paths,
            dst_video_path=dst_path,
            camera_labels=selected_video_keys,
            ep_idx=int(ep),
            frame_indices=ep_frame_indices,
            timestamps_per_cam=ts_per_cam,
            success_label=success_label,
            task_index=task_index,
            values=values,
            targets=targets,
            advantages=advantages,
            indicators=indicators,
            interventions=interventions,
            threshold=threshold,
            overwrite=args.overwrite,
            vcodec=args.vcodec,
            frame_storage_mode=args.frame_storage_mode,
            output_dir=args.output_dir,
            smooth_window=args.smooth_window,
        )
        if out is not None:
            written.append(out)

    summary_path = _write_summary_csv(args.output_dir, rows)
    logging.info("Wrote diagnostics summary: %s", summary_path)
    logging.info("Wrote %d diagnostics videos to %s", len(written), args.output_dir)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export_diagnostics(_parse_args())


if __name__ == "__main__":
    main()
