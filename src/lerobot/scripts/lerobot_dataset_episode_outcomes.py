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
List successful and failed episode indices for a LeRobot dataset.

Examples:

```bash
lerobot-dataset-episode-outcomes --dataset local/eval_twl2_10_hil_auto
lerobot-dataset-episode-outcomes --dataset ~/.cache/huggingface/lerobot/local/eval_twl2_10_hil_auto --json
```
"""

import argparse
import json
from pathlib import Path
from typing import Any

from lerobot.datasets.utils import load_episodes
from lerobot.scripts.lerobot_dataset_report import resolve_dataset_root
from lerobot.utils.recording_annotations import EPISODE_FAILURE, EPISODE_SUCCESS


def _normalize_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def build_episode_outcomes(
    dataset_root: Path,
    label_field: str = "episode_success",
) -> dict[str, Any]:
    episodes_ds = load_episodes(dataset_root)
    episodes_df = episodes_ds.to_pandas()

    if label_field not in episodes_df.columns:
        raise ValueError(
            f"Episode metadata field '{label_field}' not found in dataset. "
            f"Available columns: {list(episodes_df.columns)}"
        )
    if "episode_index" not in episodes_df.columns:
        raise ValueError(
            f"Episode metadata field 'episode_index' not found in dataset. "
            f"Available columns: {list(episodes_df.columns)}"
        )

    success_episode_indices: list[int] = []
    failure_episode_indices: list[int] = []
    unlabeled_episode_indices: list[int] = []

    for _, row in episodes_df.iterrows():
        episode_index = int(row["episode_index"])
        label = _normalize_label(row[label_field])

        if label == EPISODE_SUCCESS:
            success_episode_indices.append(episode_index)
        elif label == EPISODE_FAILURE:
            failure_episode_indices.append(episode_index)
        else:
            unlabeled_episode_indices.append(episode_index)

    return {
        "dataset_root": str(dataset_root),
        "label_field": label_field,
        "success_episode_indices": success_episode_indices,
        "failure_episode_indices": failure_episode_indices,
        "unlabeled_episode_indices": unlabeled_episode_indices,
        "success_count": len(success_episode_indices),
        "failure_count": len(failure_episode_indices),
        "unlabeled_count": len(unlabeled_episode_indices),
        "total_episodes": int(len(episodes_df)),
    }


def format_text_outcomes(outcomes: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=== Dataset Episode Outcomes ===")
    lines.append(f"Root: {outcomes['dataset_root']}")
    lines.append(f"Label field: {outcomes['label_field']}")
    lines.append("")
    lines.append(f"Success episodes ({outcomes['success_count']}): {outcomes['success_episode_indices']}")
    lines.append(f"Failure episodes ({outcomes['failure_count']}): {outcomes['failure_episode_indices']}")
    lines.append(
        f"Unlabeled episodes ({outcomes['unlabeled_count']}): "
        f"{outcomes['unlabeled_episode_indices']}"
    )
    lines.append(f"Total episodes: {outcomes['total_episodes']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="List success-labeled and failure-labeled episode indices for a LeRobot dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset repo id (e.g. local/eval_xxx) or absolute/local filesystem path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional root directory containing datasets. Defaults to HF_LEROBOT_HOME.",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default="episode_success",
        help="Episode-level metadata field containing success/failure labels.",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text.")
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset, args.root)
    outcomes = build_episode_outcomes(dataset_root, label_field=args.label_field)

    if args.json:
        print(json.dumps(outcomes, indent=2, ensure_ascii=False))
    else:
        print(format_text_outcomes(outcomes))


if __name__ == "__main__":
    main()
