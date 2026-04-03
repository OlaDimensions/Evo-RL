"""
Save the first frame of observation.images.front for each episode.

Usage:
    python save_first_frames.py --repo ruanafan/pnp0329_success --output-dir first_frames
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--feature", type=str, default="observation.images.front")
    parser.add_argument("--output-dir", type=str, default="first_frames")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = LeRobotDataset(args.repo)
    total_eps = src.meta.total_episodes
    print(f"Dataset has {total_eps} episodes.")

    ep_index_col = src.hf_dataset["episode_index"]

    # Find the first frame index for each episode
    first_frame = {}
    for i, ep in enumerate(ep_index_col):
        ep = int(ep)
        if ep not in first_frame:
            first_frame[ep] = i

    for ep_idx in tqdm(sorted(first_frame.keys()), desc="Saving frames"):
        item = src[first_frame[ep_idx]]
        img = item[args.feature]
        if isinstance(img, torch.Tensor):
            # Convert CHW float [0,1] to HWC uint8
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img = (img * 255).clamp(0, 255).byte().numpy()
        img = Image.fromarray(img)
        img.save(output_dir / f"episode_{ep_idx:04d}.png")

    print(f"Saved {len(first_frame)} images to {output_dir}/")


if __name__ == "__main__":
    main()
