#!/usr/bin/env python

import numpy as np
import pytest

from lerobot.scripts.lerobot_value_infer import (
    _binarize_advantages,
    _compute_dense_rewards_from_targets,
    _compute_n_step_advantages,
    _compute_task_thresholds,
    _load_prediction_progress,
    _save_prediction_progress_atomic,
)


def test_compute_dense_rewards_from_targets_terminal_handling():
    episode_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    frame_indices = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    targets = np.array([-0.6, -0.4, -0.2, -0.8, -0.5], dtype=np.float32)

    rewards = _compute_dense_rewards_from_targets(targets, episode_indices, frame_indices)
    expected = np.array([-0.2, -0.2, -0.2, -0.3, -0.5], dtype=np.float32)
    assert np.allclose(rewards, expected)


def test_compute_n_step_advantages_simple_case():
    rewards = np.array([-0.2, -0.2, -0.2], dtype=np.float32)
    values = np.array([-0.5, -0.3, -0.1], dtype=np.float32)
    episode_indices = np.array([0, 0, 0], dtype=np.int64)
    frame_indices = np.array([0, 1, 2], dtype=np.int64)

    advantages = _compute_n_step_advantages(
        rewards=rewards,
        values=values,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        n_step=2,
    )

    expected = np.array([0.0, -0.1, -0.1], dtype=np.float32)
    assert np.allclose(advantages, expected)


def test_compute_task_thresholds_and_binarize_with_interventions():
    task_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    advantages = np.array([-0.4, -0.1, 0.3, -0.2, 0.2], dtype=np.float32)
    interventions = np.array([0, 1, 0, 0, 0], dtype=np.float32)

    thresholds = _compute_task_thresholds(task_indices, advantages, positive_ratio=0.5)

    indicators = _binarize_advantages(
        task_indices=task_indices,
        advantages=advantages,
        thresholds=thresholds,
        interventions=interventions,
        force_intervention_positive=True,
    )

    assert indicators.tolist() == [0, 1, 1, 0, 1]


def test_prediction_progress_round_trip(tmp_path):
    progress_path = tmp_path / "value_infer_progress.npz"
    absolute_indices = np.array([10, 11, 12, 13], dtype=np.int64)
    prediction_lookup = np.zeros(14, dtype=np.float32)
    prediction_seen = np.zeros(14, dtype=np.bool_)
    prediction_lookup[[10, 12]] = np.array([0.25, -0.75], dtype=np.float32)
    prediction_seen[[10, 12]] = True
    metadata = {"schema_version": 1, "run": "test"}

    saved_count = _save_prediction_progress_atomic(
        progress_path=progress_path,
        prediction_lookup=prediction_lookup,
        prediction_seen=prediction_seen,
        absolute_indices=absolute_indices,
        metadata=metadata,
    )

    assert saved_count == 2
    assert progress_path.is_file()
    assert not (tmp_path / "value_infer_progress.npz.tmp").exists()

    loaded = _load_prediction_progress(progress_path=progress_path, expected_metadata=metadata)

    assert loaded is not None
    indices, values = loaded
    assert indices.tolist() == [10, 12]
    assert np.allclose(values, np.array([0.25, -0.75], dtype=np.float32))


def test_prediction_progress_metadata_mismatch_raises(tmp_path):
    progress_path = tmp_path / "value_infer_progress.npz"
    absolute_indices = np.array([0], dtype=np.int64)
    prediction_lookup = np.array([1.0], dtype=np.float32)
    prediction_seen = np.array([True], dtype=np.bool_)

    _save_prediction_progress_atomic(
        progress_path=progress_path,
        prediction_lookup=prediction_lookup,
        prediction_seen=prediction_seen,
        absolute_indices=absolute_indices,
        metadata={"schema_version": 1, "run": "old"},
    )

    with pytest.raises(ValueError, match="does not match"):
        _load_prediction_progress(
            progress_path=progress_path,
            expected_metadata={"schema_version": 1, "run": "new"},
        )


def test_prediction_progress_missing_keys_raises(tmp_path):
    progress_path = tmp_path / "value_infer_progress.npz"
    with open(progress_path, "wb") as f:
        np.savez_compressed(f, indices=np.array([0], dtype=np.int64))

    with pytest.raises(ValueError, match="missing keys"):
        _load_prediction_progress(progress_path=progress_path, expected_metadata={})
