from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtWidgets import QApplication


EXTRA_STYLE = """
QWidget {
    font-size: 13px;
}

QMainWindow, QWidget#MainRoot {
    background: #12181f;
}

QFrame#HeaderFrame {
    background: #17212b;
    border: 1px solid #263545;
    border-radius: 8px;
}

QLabel#AppTitle {
    color: #f4f7fb;
    font-size: 22px;
    font-weight: 700;
}

QLabel#AppSubtitle {
    color: #93a4b8;
}

QLabel#StateBadge {
    border-radius: 12px;
    padding: 4px 12px;
    font-weight: 700;
}

QLabel[state="idle"] {
    background: #2c3744;
    color: #c9d4e2;
}

QLabel[state="starting"] {
    background: #5f4b12;
    color: #ffe8a3;
}

QLabel[state="running"] {
    background: #123d35;
    color: #84f1d4;
}

QLabel[state="resetting"] {
    background: #4d3514;
    color: #ffd08a;
}

QLabel[state="stopped"] {
    background: #243545;
    color: #a8d8ff;
}

QLabel[state="error"] {
    background: #4d1f26;
    color: #ffb3bd;
}

QGroupBox {
    border: 1px solid #263545;
    border-radius: 8px;
    margin-top: 18px;
    padding: 16px 12px 12px 12px;
    font-weight: 700;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #dce6f2;
}

QLineEdit, QTextEdit, QComboBox {
    border: 1px solid #4a6076;
    border-radius: 5px;
    padding: 7px 9px;
    background: #223041;
    color: #eef4fb;
    selection-background-color: #00a783;
}

QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border: 1px solid #00c9a3;
    background: #26384a;
}

QLineEdit:disabled, QTextEdit:disabled, QComboBox:disabled {
    border: 1px solid #2f3f50;
    background: #18222d;
    color: #7f90a3;
}

QComboBox QAbstractItemView {
    border: 1px solid #314255;
    border-radius: 5px;
    background: #223041;
    color: #eef4fb;
    selection-background-color: #00a783;
    selection-color: #07110f;
    outline: 0;
}

QComboBox QAbstractItemView::item {
    min-height: 28px;
    padding: 6px 10px;
}

QTextEdit#TaskEditor {
    min-height: 84px;
}

QTextEdit#LogView {
    border: 1px solid #263545;
    border-radius: 8px;
    background: #080d12;
    color: #b9f6d4;
    padding: 10px;
}

QFrame#ControlCard, QFrame#StatusCard {
    background: #17212b;
    border: 1px solid #263545;
    border-radius: 8px;
}

QLabel#SectionTitle {
    color: #f4f7fb;
    font-size: 15px;
    font-weight: 700;
}

QLabel#SectionHint, QLabel#ShortcutHint, QLabel#StatusDetail {
    color: #93a4b8;
}

QLabel#StatusTitle {
    color: #f4f7fb;
    font-weight: 700;
}

QLabel#FormLabel {
    color: #dce6f2;
    font-weight: 700;
}

QPushButton {
    border-radius: 6px;
    min-height: 34px;
    padding: 7px 12px;
    font-weight: 700;
}

QPushButton[role="primary"] {
    background: #00a783;
    color: #07110f;
}

QPushButton[role="danger"] {
    background: #c2414a;
    color: #fff5f6;
}

QPushButton[role="success"] {
    background: #2f9e44;
    color: #f2fff5;
}

QPushButton[role="warning"] {
    background: #d98b18;
    color: #fff7e6;
}

QPushButton[role="secondary"] {
    background: #263545;
    color: #e8f0fa;
}

QPushButton:disabled {
    background: #1b2631;
    color: #667789;
}
"""


def apply_gui_theme(app: QApplication) -> None:
    """Apply the optional Material theme, then layer Evo-RL GUI refinements."""

    try:
        from qt_material import apply_stylesheet
    except Exception:
        app.setStyleSheet(EXTRA_STYLE)
        return

    resource_dir = Path(tempfile.gettempdir()) / "evo_rl_qt_material"
    try:
        apply_stylesheet(app, theme="dark_teal.xml", extra=EXTRA_STYLE, parent=str(resource_dir))
    except Exception:
        app.setStyleSheet(EXTRA_STYLE)
