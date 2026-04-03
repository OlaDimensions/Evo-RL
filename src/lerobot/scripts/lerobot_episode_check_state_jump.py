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
Check LeRobot episodes for large jumps in observation.state.

A jump is detected when, for any dimension d:
abs(state[t + frame_gap, d] - state[t, d]) > threshold

Examples:

```bash
lerobot-episode-check-state-jump \
  --dataset local/my_dataset \
  --output-txt ./state_jump_records.txt
```
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pa_ds

from lerobot.datasets.utils import load_info
from lerobot.scripts.lerobot_dataset_report import resolve_dataset_root


@dataclass(frozen=True)
class StateJumpRecord:
    episode_index: int
    from_frame: int
    to_frame: int
    dim: int
    from_value: float
    to_value: float
    delta: float


def _to_state_vector(value: Any, *, state_key: str, episode_index: int, frame_index: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1D array for '{state_key}', got shape {arr.shape} "
            f"(episode={episode_index}, frame={frame_index})."
        )
    return arr


def _build_episode_sequences(
    dataset_root: Path,
    state_key: str,
) -> dict[int, list[tuple[int, int, np.ndarray]]]:
    data_dataset = pa_ds.dataset(dataset_root / "data", format="parquet")
    schema_names = set(data_dataset.schema.names)
    if state_key not in schema_names:
        raise ValueError(
            f"State key '{state_key}' not found in data parquet schema. "
            f"Available columns include: {sorted(schema_names)[:20]}"
        )
    if "episode_index" not in schema_names:
        raise ValueError("Required column 'episode_index' not found in data parquet schema.")

    order_key = "index" if "index" in schema_names else "frame_index" if "frame_index" in schema_names else None
    if order_key is None:
        raise ValueError("Neither 'index' nor 'frame_index' exists in data parquet schema.")
    frame_key = "frame_index" if "frame_index" in schema_names else order_key

    columns = ["episode_index", order_key, frame_key, state_key]
    unique_columns: list[str] = []
    for column in columns:
        if column not in unique_columns:
            unique_columns.append(column)

    table = data_dataset.to_table(columns=unique_columns)
    episode_indices = table["episode_index"].to_pylist()
    order_values = table[order_key].to_pylist()
    frame_values = table[frame_key].to_pylist()
    state_values = table[state_key].to_pylist()

    episode_sequences: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    for ep_idx_raw, order_raw, frame_raw, state_raw in zip(
        episode_indices, order_values, frame_values, state_values, strict=True
    ):
        ep_idx = int(ep_idx_raw)
        order = int(order_raw)
        frame = int(frame_raw)
        state = _to_state_vector(state_raw, state_key=state_key, episode_index=ep_idx, frame_index=frame)
        episode_sequences.setdefault(ep_idx, []).append((order, frame, state))

    for ep_idx in episode_sequences:
        episode_sequences[ep_idx].sort(key=lambda row: row[0])

    return episode_sequences


def check_dataset_state_jumps(
    dataset_root: Path,
    *,
    state_key: str = "observation.state",
    threshold: float = 40.0,
    frame_gap: int = 13,
) -> list[StateJumpRecord]:
    info = load_info(dataset_root)
    features = info.get("features", {})
    if state_key not in features:
        raise ValueError(
            f"Feature '{state_key}' not found in dataset info features. "
            f"Available features: {sorted(features.keys())[:20]}"
        )
    if frame_gap <= 0:
        raise ValueError(f"frame_gap must be positive, got {frame_gap}")

    episode_sequences = _build_episode_sequences(dataset_root, state_key=state_key)

    records: list[StateJumpRecord] = []
    for ep_idx in sorted(episode_sequences.keys()):
        rows = episode_sequences[ep_idx]
        if len(rows) <= frame_gap:
            continue

        for t in range(0, len(rows) - frame_gap):
            _, from_frame, from_state = rows[t]
            _, to_frame, to_state = rows[t + frame_gap]
            if from_state.shape != to_state.shape:
                raise ValueError(
                    f"Inconsistent '{state_key}' shape in episode {ep_idx}: "
                    f"{from_state.shape} (frame={from_frame}) vs {to_state.shape} (frame={to_frame})."
                )

            deltas = np.abs(to_state - from_state)
            hit_dims = np.flatnonzero(deltas > threshold)
            for dim_idx in hit_dims.tolist():
                records.append(
                    StateJumpRecord(
                        episode_index=ep_idx,
                        from_frame=from_frame,
                        to_frame=to_frame,
                        dim=int(dim_idx),
                        from_value=float(from_state[dim_idx]),
                        to_value=float(to_state[dim_idx]),
                        delta=float(deltas[dim_idx]),
                    )
                )

    return records


def write_state_jump_report(
    output_path: Path,
    *,
    dataset_root: Path,
    state_key: str,
    threshold: float,
    frame_gap: int,
    records: list[StateJumpRecord],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"dataset_root={dataset_root}\n")
        f.write(f"state_key={state_key}\n")
        f.write(f"threshold={threshold}\n")
        f.write(f"frame_gap={frame_gap}\n")
        f.write(f"total_hits={len(records)}\n")
        for record in records:
            f.write(
                " | ".join(
                    [
                        f"episode={record.episode_index}",
                        f"from_frame={record.from_frame}",
                        f"to_frame={record.to_frame}",
                        f"dim={record.dim}",
                        f"from_value={record.from_value:.6f}",
                        f"to_value={record.to_value:.6f}",
                        f"delta={record.delta:.6f}",
                    ]
                )
            )
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Check observation.state jumps between frames t and t+13.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset repo id (e.g. local/eval_xxx) or absolute/local filesystem path.",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path("./episode_state_jump_records.txt"),
        help="Output txt file path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional root directory containing datasets. Defaults to HF_LEROBOT_HOME.",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default="observation.state",
        help="Feature key to inspect.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=40.0,
        help="Trigger threshold, strict > comparison.",
    )
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=13,
        help="Frame gap used in comparisons: state[t] vs state[t+frame_gap].",
    )
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset, args.root)
    records = check_dataset_state_jumps(
        dataset_root,
        state_key=args.state_key,
        threshold=args.threshold,
        frame_gap=args.frame_gap,
    )
    output_path = args.output_txt.expanduser().resolve()
    write_state_jump_report(
        output_path,
        dataset_root=dataset_root,
        state_key=args.state_key,
        threshold=args.threshold,
        frame_gap=args.frame_gap,
        records=records,
    )
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
