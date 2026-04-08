#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Horizontally stack same-name value visualization videos from multiple directories.

Example:
    python -m lerobot.scripts.lerobot_concat_value_videos \
        --input 6k=~/autodl-tmp/outputs/value_infer/pnpround2_6k/value/viz \
        --input 4k=~/autodl-tmp/outputs/value_infer/pnpround2_4k/value/viz \
        --input 8k=~/autodl-tmp/outputs/value_infer/pnpround2_8k/value/viz \
        --output-dir ~/autodl-tmp/outputs/value_infer/pnpround2_compare/viz_concat
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm


def parse_input_pair(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(
            f"Invalid --input '{raw}'. Expected format is LABEL=PATH, e.g. 6k=/path/to/viz"
        )
    label, path = raw.split("=", 1)
    label = label.strip()
    path = Path(path.strip()).expanduser().resolve()
    if not label:
        raise ValueError(f"Invalid --input '{raw}': label cannot be empty.")
    return label, path


def escape_drawtext_text(text: str) -> str:
    # Minimal escaping for ffmpeg drawtext text field.
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
        .replace("%", "\\%")
    )


def collect_common_video_names(viz_dirs: list[Path], suffix: str) -> list[str]:
    sets = []
    for d in viz_dirs:
        names = {p.name for p in d.iterdir() if p.is_file() and p.name.endswith(suffix)}
        sets.append(names)

    if not sets:
        return []
    return sorted(set.intersection(*sets))


def build_filter_complex(labels: list[str], font_size: int) -> str:
    streams = []
    for i, label in enumerate(labels):
        safe_text = escape_drawtext_text(label)
        streams.append(
            (
                f"[{i}:v]setpts=PTS-STARTPTS,"
                f"drawtext=text='{safe_text}':x=20:y=20:fontsize={font_size}:"
                "fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8"
                f"[v{i}]"
            )
        )
    stacked_inputs = "".join(f"[v{i}]" for i in range(len(labels)))
    streams.append(f"{stacked_inputs}hstack=inputs={len(labels)}:shortest=1[vout]")
    return ";".join(streams)


def concat_one_video(
    inputs: list[Path],
    labels: list[str],
    output_path: Path,
    overwrite: bool,
    font_size: int,
) -> None:
    filter_complex = build_filter_complex(labels, font_size=font_size)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    cmd.append("-y" if overwrite else "-n")

    for input_path in inputs:
        cmd += ["-i", str(input_path)]

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Horizontally stack same-name videos across multiple viz directories."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input in LABEL=PATH form. Repeat this argument, e.g. --input 6k=/a/viz --input 8k=/b/viz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for concatenated videos.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".mp4",
        help="Only process files ending with this suffix. Default: .mp4",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=32,
        help="Label font size for drawtext. Default: 32",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH.")

    parsed_inputs = [parse_input_pair(raw) for raw in args.input]
    if len(parsed_inputs) < 2:
        raise ValueError("At least two --input LABEL=PATH entries are required.")

    labels = [x[0] for x in parsed_inputs]
    viz_dirs = [x[1] for x in parsed_inputs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate labels are not allowed: {labels}")

    for d in viz_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Input directory does not exist: {d}")
        if not d.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {d}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    common_names = collect_common_video_names(viz_dirs, suffix=args.suffix)
    if not common_names:
        raise RuntimeError(
            f"No common files with suffix '{args.suffix}' were found across all input directories."
        )

    logging.info("Inputs:")
    for label, d in parsed_inputs:
        total_here = len([p for p in d.iterdir() if p.is_file() and p.name.endswith(args.suffix)])
        logging.info("  - %s -> %s (%d files)", label, d, total_here)
    logging.info("Common files to process: %d", len(common_names))
    logging.info("Output directory: %s", output_dir)

    for name in tqdm(common_names, desc="Concatenating videos"):
        out_path = output_dir / name
        if out_path.exists() and not args.overwrite:
            continue

        in_paths = [d / name for d in viz_dirs]
        concat_one_video(
            inputs=in_paths,
            labels=labels,
            output_path=out_path,
            overwrite=args.overwrite,
            font_size=args.font_size,
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()
