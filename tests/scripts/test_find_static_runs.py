from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pyarrow as pa


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "find_static_runs.py"
SPEC = spec_from_file_location("find_static_runs", SCRIPT_PATH)
find_static_runs = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = find_static_runs
SPEC.loader.exec_module(find_static_runs)


def make_table(rows):
    return pa.table(
        {
            "observation.state": [row["state"] for row in rows],
            "action": [row["action"] for row in rows],
            "episode_index": [row["episode_index"] for row in rows],
            "frame_index": [row["frame_index"] for row in rows],
            "timestamp": [row["frame_index"] / 30 for row in rows],
            "index": list(range(len(rows))),
        }
    )


def test_detect_static_runs_respects_min_run_frames_and_tolerance():
    rows = [
        {"episode_index": 0, "frame_index": 0, "state": [0.0], "action": [0.0]},
        {"episode_index": 0, "frame_index": 1, "state": [0.0], "action": [0.0]},
        {"episode_index": 0, "frame_index": 2, "state": [0.0], "action": [0.0]},
        {"episode_index": 0, "frame_index": 3, "state": [0.0], "action": [0.0]},
        {"episode_index": 0, "frame_index": 4, "state": [0.01], "action": [0.0]},
        {"episode_index": 0, "frame_index": 5, "state": [0.01005], "action": [0.0]},
        {"episode_index": 0, "frame_index": 6, "state": [0.01009], "action": [0.0]},
    ]

    runs = find_static_runs.detect_static_runs(
        make_table(rows),
        state_key="observation.state",
        action_key="action",
        tol=1e-4,
        min_run_frames=3,
        fps=30,
    )

    assert len(runs) == 1
    assert runs[0]["start_frame_index"] == 1
    assert runs[0]["end_frame_index"] == 3
    assert runs[0]["frame_indices"] == [1, 2, 3]
    assert runs[0]["anchor_frame_index"] == 0


def test_detect_static_runs_does_not_merge_across_episodes():
    rows = [
        {"episode_index": 0, "frame_index": 0, "state": [0.0], "action": [0.0]},
        {"episode_index": 0, "frame_index": 1, "state": [0.0], "action": [0.0]},
        {"episode_index": 0, "frame_index": 2, "state": [0.0], "action": [0.0]},
        {"episode_index": 1, "frame_index": 0, "state": [0.0], "action": [0.0]},
        {"episode_index": 1, "frame_index": 1, "state": [0.0], "action": [0.0]},
        {"episode_index": 1, "frame_index": 2, "state": [0.0], "action": [0.0]},
    ]

    runs = find_static_runs.detect_static_runs(
        make_table(rows),
        state_key="observation.state",
        action_key="action",
        tol=0.0,
        min_run_frames=3,
        fps=30,
    )

    assert runs == []
