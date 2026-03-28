#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Filter a LeRobot dataset down to successful episodes only.

Examples:

```bash
lerobot-filter-dataset-success \
  --dataset local/my_hil_dataset \
  --output-dir ~/.cache/huggingface/lerobot/local/my_hil_dataset_success_only

lerobot-filter-dataset-success \
  --dataset ~/.cache/huggingface/lerobot/local/my_hil_dataset \
  --output-dir /tmp/my_hil_dataset_success_only
```
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_episodes
from lerobot.scripts.lerobot_dataset_report import resolve_dataset_root
from lerobot.utils.utils import init_logging


def _normalize_label(value: Any) -> str:
    if value is None:
        return ""
    label = str(value).strip().lower()
    return label


def get_success_episode_indices(
    dataset_root: Path,
    label_field: str = "episode_success",
    success_label: str = "success",
) -> list[int]:
    episodes_ds = load_episodes(dataset_root)
    if episodes_ds is None or len(episodes_ds) == 0:
        return []
    if label_field not in episodes_ds.column_names:
        raise ValueError(
            f"Episode metadata field '{label_field}' not found in dataset. "
            f"Available columns: {episodes_ds.column_names}"
        )

    success_label_normalized = _normalize_label(success_label)
    success_episode_indices: list[int] = []

    for row in episodes_ds:
        if _normalize_label(row.get(label_field)) == success_label_normalized:
            success_episode_indices.append(int(row["episode_index"]))

    return success_episode_indices


def filter_dataset_to_success_episodes(
    dataset_root: Path,
    output_dir: Path,
    output_repo_id: str | None = None,
    label_field: str = "episode_success",
    success_label: str = "success",
) -> LeRobotDataset:
    success_episode_indices = get_success_episode_indices(
        dataset_root=dataset_root,
        label_field=label_field,
        success_label=success_label,
    )
    if not success_episode_indices:
        raise ValueError(
            f"No successful episodes found in dataset '{dataset_root}' using "
            f"{label_field} == '{success_label}'."
        )

    source_dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)
    delete_episode_indices = [
        episode_index
        for episode_index in range(source_dataset.meta.total_episodes)
        if episode_index not in set(success_episode_indices)
    ]

    repo_id = output_repo_id if output_repo_id is not None else output_dir.name
    return delete_episodes(
        source_dataset,
        episode_indices=delete_episode_indices,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def main():
    parser = argparse.ArgumentParser(description="Keep only success-labeled episodes from a LeRobot dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset repo id (e.g. local/eval_xxx) or absolute/local filesystem path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the filtered dataset will be written.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional root directory containing datasets. Defaults to HF_LEROBOT_HOME.",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Optional repo_id metadata for the new dataset. Defaults to output directory name.",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default="episode_success",
        help="Episode-level metadata field used to decide whether an episode is successful.",
    )
    parser.add_argument(
        "--success-label",
        type=str,
        default="success",
        help="Label value treated as success after lower/strip normalization.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    args = parser.parse_args()

    init_logging()

    dataset_root = resolve_dataset_root(args.dataset, args.root)
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --force to overwrite it."
            )
        shutil.rmtree(output_dir)

    success_episode_indices = get_success_episode_indices(
        dataset_root=dataset_root,
        label_field=args.label_field,
        success_label=args.success_label,
    )

    logging.info(
        "Found %d successful episodes in %s using %s == %s",
        len(success_episode_indices),
        dataset_root,
        args.label_field,
        args.success_label,
    )

    new_dataset = filter_dataset_to_success_episodes(
        dataset_root=dataset_root,
        output_dir=output_dir,
        output_repo_id=args.output_repo_id,
        label_field=args.label_field,
        success_label=args.success_label,
    )

    logging.info("Filtered dataset saved to %s", output_dir)
    logging.info(
        "New dataset totals: episodes=%d frames=%d",
        new_dataset.meta.total_episodes,
        new_dataset.meta.total_frames,
    )


if __name__ == "__main__":
    main()
