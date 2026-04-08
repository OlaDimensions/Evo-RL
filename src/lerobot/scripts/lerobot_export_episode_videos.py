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

"""Export per-episode videos from a LeRobot dataset image/video key.

Example:
    python -m lerobot.scripts.lerobot_export_episode_videos \
        --repo-id local/my_dataset \
        --root ~/.cache/huggingface/lerobot \
        --image-key observation.images.front \
        --output-dir ./episode_videos
"""

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm
from torchvision.transforms import functional as TF

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import encode_video_frames


def save_tensor_frame_as_png(chw_float: torch.Tensor, save_path: Path) -> None:
    """Save a CHW float image in [0, 1] to PNG."""
    if chw_float.ndim != 3:
        raise ValueError(f"Expected 3D CHW image, got shape={chw_float.shape}")

    if chw_float.shape[0] in (1, 3):
        img = TF.to_pil_image(chw_float.clamp(0.0, 1.0))
    else:
        raise ValueError(f"Expected C in (1, 3), got C={chw_float.shape[0]}")

    img.save(save_path)


def export_episode_video(
    dataset: LeRobotDataset,
    episode_index: int,
    image_key: str,
    output_path: Path,
    fps: int,
    overwrite: bool,
    vcodec: str,
    accelerator: str,
) -> None:
    if output_path.exists() and not overwrite:
        logging.info("Skip existing output: %s", output_path)
        return

    episode_meta = dataset.meta.episodes[episode_index]
    start_idx = int(episode_meta["dataset_from_index"])
    end_idx = int(episode_meta["dataset_to_index"])

    if end_idx <= start_idx:
        logging.warning("Episode %d has no frames, skipping.", episode_index)
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for frame_i, dataset_idx in enumerate(range(start_idx, end_idx)):
            item = dataset[dataset_idx]
            frame = item[image_key]
            frame_path = tmp_dir_path / f"frame-{frame_i:06d}.png"
            save_tensor_frame_as_png(frame, frame_path)

        if accelerator in ("apple", "gpu"):
            if accelerator == "apple":
                if vcodec == "h264":
                    hw_codec = "h264_videotoolbox"
                elif vcodec == "hevc":
                    hw_codec = "hevc_videotoolbox"
                else:
                    raise ValueError(
                        "Apple GPU acceleration currently supports only h264/hevc in this script."
                    )
            else:  # gpu (NVIDIA NVENC)
                if vcodec == "h264":
                    hw_codec = "h264_nvenc"
                elif vcodec == "hevc":
                    hw_codec = "hevc_nvenc"
                else:
                    raise ValueError(
                        "NVIDIA GPU acceleration currently supports only h264/hevc in this script."
                    )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ffmpeg_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y" if overwrite else "-n",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_dir_path / "frame-%06d.png"),
                "-c:v",
                hw_codec,
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
            subprocess.run(ffmpeg_cmd, check=True)
        else:
            encode_video_frames(
                imgs_dir=tmp_dir_path,
                video_path=output_path,
                fps=fps,
                vcodec=vcodec,
                overwrite=overwrite,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export each LeRobot episode as a separate MP4 from a camera key."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        type=str,
        help="LeRobot dataset repo id (e.g. 'lerobot/pusht' or 'local/my_dataset').",
    )
    parser.add_argument(
        "--root",
        default=None,
        type=Path,
        help="Root directory that contains datasets locally.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write per-episode videos.",
    )
    parser.add_argument(
        "--image-key",
        default="observation.images.front",
        type=str,
        help="Image/video feature key to export (default: observation.images.front).",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of episode indices. If omitted, exports all episodes.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Output FPS. Defaults to dataset FPS.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        choices=["h264", "hevc", "libsvtav1"],
        help="Video codec for encoding.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        choices=["cpu", "apple", "gpu"],
        help=(
            "Encoding accelerator: 'cpu' uses pyav encoder; "
            "'apple' uses ffmpeg VideoToolbox (h264/hevc only); "
            "'gpu' uses ffmpeg NVIDIA NVENC (h264/hevc only, requires NVIDIA GPU)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output videos if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    if args.image_key not in dataset.features:
        available_camera_like = [k for k in dataset.features if k.startswith("observation.images.")]
        raise KeyError(
            f"Feature key '{args.image_key}' not found in dataset.\n"
            f"Available camera keys: {available_camera_like}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.episodes is None or len(args.episodes) == 0:
        episodes = list(range(dataset.num_episodes))
    else:
        episodes = sorted(set(args.episodes))

    for ep_idx in episodes:
        if ep_idx < 0 or ep_idx >= dataset.num_episodes:
            raise IndexError(
                f"Episode index {ep_idx} is out of range [0, {dataset.num_episodes - 1}]"
            )

    fps = args.fps if args.fps is not None else dataset.fps
    logging.info(
        "Exporting %d episode(s), key='%s', fps=%d, vcodec=%s, accelerator=%s, output_dir=%s",
        len(episodes),
        args.image_key,
        fps,
        args.vcodec,
        args.accelerator,
        output_dir,
    )

    for ep_idx in tqdm(episodes, desc="Export episodes"):
        output_path = output_dir / f"episode_{ep_idx:06d}.mp4"
        export_episode_video(
            dataset=dataset,
            episode_index=ep_idx,
            image_key=args.image_key,
            output_path=output_path,
            fps=fps,
            overwrite=args.overwrite,
            vcodec=args.vcodec,
            accelerator=args.accelerator,
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()

