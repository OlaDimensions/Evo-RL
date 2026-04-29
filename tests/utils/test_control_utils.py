from types import SimpleNamespace

import pytest

from lerobot.utils import control_utils
from lerobot.utils.control_utils import sanity_check_bimanual_piper_pair


@pytest.mark.parametrize(
    ("robot_type", "teleop_type"),
    [
        ("bi_piper_follower", "bi_piper_leader"),
        ("bi_piperx_follower", "bi_piperx_leader"),
        ("so101_follower", "so101_leader"),
    ],
)
def test_sanity_check_bimanual_piper_pair_accepts_valid_pairs(robot_type, teleop_type):
    sanity_check_bimanual_piper_pair(
        SimpleNamespace(type=robot_type),
        SimpleNamespace(type=teleop_type),
    )


def test_sanity_check_bimanual_piper_pair_accepts_missing_teleop():
    sanity_check_bimanual_piper_pair(SimpleNamespace(type="bi_piperx_follower"), None)


@pytest.mark.parametrize(
    ("robot_type", "teleop_type"),
    [
        ("bi_piper_follower", "bi_piperx_leader"),
        ("bi_piperx_follower", "bi_piper_leader"),
        ("so101_follower", "bi_piperx_leader"),
        ("so101_follower", "bi_piper_leader"),
    ],
)
def test_sanity_check_bimanual_piper_pair_rejects_mixed_pairs(robot_type, teleop_type):
    with pytest.raises(ValueError, match="must be paired"):
        sanity_check_bimanual_piper_pair(
            SimpleNamespace(type=robot_type),
            SimpleNamespace(type=teleop_type),
        )


def test_init_keyboard_listener_force_tty_uses_tty_listener(monkeypatch):
    started = []

    class FakeStdin:
        def isatty(self):
            return True

    class FakeTTYKeyboardListener:
        def __init__(self, events, intervention_toggle_key, episode_success_key, episode_failure_key):
            self.events = events
            self.intervention_toggle_key = intervention_toggle_key
            self.episode_success_key = episode_success_key
            self.episode_failure_key = episode_failure_key

        def start(self):
            started.append(self)

    monkeypatch.setenv("LEROBOT_FORCE_TTY_KEYBOARD", "1")
    monkeypatch.setattr(control_utils.sys, "stdin", FakeStdin())
    monkeypatch.setattr(control_utils, "TTYKeyboardListener", FakeTTYKeyboardListener)

    listener, events = control_utils.init_keyboard_listener(
        intervention_toggle_key="i",
        episode_success_key="s",
        episode_failure_key="f",
    )

    assert listener is started[0]
    assert listener.episode_success_key == "s"
    assert events["exit_early"] is False
    assert events["episode_outcome"] is None


def test_init_keyboard_listener_force_tty_without_tty_returns_events(monkeypatch):
    class FakeStdin:
        def isatty(self):
            return False

    monkeypatch.setenv("LEROBOT_FORCE_TTY_KEYBOARD", "1")
    monkeypatch.setattr(control_utils.sys, "stdin", FakeStdin())

    listener, events = control_utils.init_keyboard_listener()

    assert listener is None
    assert events["stop_recording"] is False
