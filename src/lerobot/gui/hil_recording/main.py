from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from lerobot.gui.hil_recording.controllers import RecordingController
from lerobot.gui.hil_recording.theme import apply_gui_theme
from lerobot.gui.hil_recording.views import MainWindow


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def main() -> int:
    app = QApplication(sys.argv)
    apply_gui_theme(app)
    window = MainWindow()
    window.controller = RecordingController(window=window, repo_root=_repo_root())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
