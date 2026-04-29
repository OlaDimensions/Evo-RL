from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont, QIcon, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lerobot.gui.hil_recording.config_store import load_config, save_config
from lerobot.gui.hil_recording.models import (
    DEFAULT_PARAMETER_SPECS,
    HardwareHealthReport,
    HealthCheckResult,
    ParameterSpec,
    RecordingParameters,
    RecordingState,
    StatusLevel,
)


STATUS_COLORS = {
    StatusLevel.UNKNOWN: "#7a8794",
    StatusLevel.CHECKING: "#e0a923",
    StatusLevel.OK: "#2fbd73",
    StatusLevel.WARNING: "#f29d38",
    StatusLevel.ERROR: "#e45461",
}

STATE_LABELS = {
    RecordingState.IDLE: "IDLE",
    RecordingState.STARTING: "STARTING",
    RecordingState.RUNNING: "RECORDING",
    RecordingState.RESETTING: "RESETTING",
    RecordingState.STOPPED: "STOPPED",
    RecordingState.ERROR: "ERROR",
}

DATASET_PARAMETER_KEYS = {
    "dataset_name",
    "dataset_single_task",
    "dataset_num_episodes",
    "dataset_episode_time_s",
    "dataset_reset_time_s",
    "resume",
}

def _icon(name: str, color: str = "#e8f0fa") -> QIcon:
    try:
        import qtawesome as qta

        return qta.icon(name, color=color)
    except Exception:
        return QIcon()


def _refresh_dynamic_style(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _form_group(title: str) -> tuple[QGroupBox, QFormLayout]:
    group = QGroupBox(title)
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    layout.setHorizontalSpacing(14)
    layout.setVerticalSpacing(10)
    return group, layout


class StatusLight(QFrame):
    def __init__(self, title: str, icon_name: str):
        super().__init__()
        self.setObjectName("StatusCard")
        self.title = QLabel(title)
        self.title.setObjectName("StatusTitle")
        self.icon = QLabel()
        self.dot = QLabel()
        self.level = QLabel("UNKNOWN")
        self.level.setObjectName("ShortcutHint")
        self.detail = QLabel("Unknown")
        self.detail.setObjectName("StatusDetail")
        self.detail.setWordWrap(True)
        self.detail.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QGridLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(5)
        layout.setColumnStretch(2, 1)

        icon = _icon(icon_name, "#dce6f2")
        if not icon.isNull():
            self.icon.setPixmap(icon.pixmap(QSize(22, 22)))
        self.icon.setFixedSize(24, 24)

        layout.addWidget(self.icon, 0, 0, 2, 1, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.title, 0, 1)
        layout.addWidget(self.dot, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.level, 1, 1)
        layout.addWidget(self.detail, 2, 0, 1, 3)
        self.set_result(HealthCheckResult(title, StatusLevel.UNKNOWN, "Unknown"))

    def set_result(self, result: HealthCheckResult) -> None:
        color = STATUS_COLORS[result.level]
        self.dot.setFixedSize(12, 12)
        self.dot.setStyleSheet(f"background:{color}; border-radius:6px;")
        self.level.setText(result.level.value.upper())
        self.level.setStyleSheet(f"color:{color}; font-weight:700;")
        self.detail.setText(result.detail or result.level.value)


class ParameterPanel(QWidget):
    def __init__(self, specs: list[ParameterSpec] | None = None, values: dict[str, str] | None = None):
        super().__init__()
        self.specs = specs or DEFAULT_PARAMETER_SPECS
        values = values or {}
        self.editors: dict[str, QCheckBox | QLineEdit | QTextEdit | QComboBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        dataset_group, dataset_layout = _form_group("Dataset")
        policy_group, policy_layout = _form_group("Policy")

        for spec in self.specs:
            initial_value = values.get(spec.key, spec.default)
            if spec.key == "resume":
                editor = QCheckBox()
                editor.setChecked(initial_value.strip().lower() in {"1", "true", "yes", "on"})
            elif spec.choices:
                editor = QComboBox()
                for value, label in spec.choices:
                    editor.addItem(label, value)
                default_index = editor.findData(initial_value)
                if default_index >= 0:
                    editor.setCurrentIndex(default_index)
                if spec.key == "policy_mode":
                    editor.currentIndexChanged.connect(self._update_policy_controls)
            elif spec.key == "dataset_single_task":
                editor = QTextEdit(initial_value)
                editor.setObjectName("TaskEditor")
                editor.setPlaceholderText(spec.placeholder)
                editor.setAcceptRichText(False)
                editor.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
            else:
                editor = QLineEdit(initial_value)
                editor.setPlaceholderText(spec.placeholder)
            self.editors[spec.key] = editor

            label = QLabel(spec.label)
            label.setObjectName("FormLabel")
            target_layout = dataset_layout if spec.key in DATASET_PARAMETER_KEYS else policy_layout
            target_layout.addRow(label, editor)

        layout.addWidget(dataset_group)
        layout.addWidget(policy_group)
        layout.addStretch(1)
        self._update_policy_controls()

    def parameters(self) -> RecordingParameters:
        values = {}
        for key, editor in self.editors.items():
            if isinstance(editor, QComboBox):
                values[key] = str(editor.currentData())
            elif isinstance(editor, QCheckBox):
                values[key] = "true" if editor.isChecked() else "false"
            elif isinstance(editor, QTextEdit):
                values[key] = editor.toPlainText()
            else:
                values[key] = editor.text()
        return RecordingParameters(values)

    def set_parameter(self, key: str, value: str) -> None:
        editor = self.editors.get(key)
        if isinstance(editor, QComboBox):
            index = editor.findData(value)
            if index >= 0:
                editor.setCurrentIndex(index)
        elif isinstance(editor, QCheckBox):
            editor.setChecked(value.strip().lower() in {"1", "true", "yes", "on"})
        elif isinstance(editor, QTextEdit):
            editor.setPlainText(value)
        elif isinstance(editor, QLineEdit):
            editor.setText(value)

    def focus_parameter(self, key: str) -> None:
        editor = self.editors.get(key)
        if editor is not None:
            editor.setFocus(Qt.FocusReason.OtherFocusReason)

    def set_editable(self, editable: bool) -> None:
        for editor in self.editors.values():
            editor.setEnabled(editable)
        if editable:
            self._update_policy_controls()

    def _update_policy_controls(self) -> None:
        policy_mode_editor = self.editors.get("policy_mode")
        if not isinstance(policy_mode_editor, QComboBox):
            return
        mode = str(policy_mode_editor.currentData())
        for key in ("policy_path", "openpi_policy_dir", "openpi_server_root", "policy_host", "policy_port"):
            editor = self.editors.get(key)
            if editor is None:
                continue
            editor.setEnabled(
                (key == "policy_path" and mode == "local_path")
                or (
                    key in {"openpi_policy_dir", "openpi_server_root", "policy_host", "policy_port"}
                    and mode == "openpi_remote"
                )
            )


class MainWindow(QMainWindow):
    start_recording_requested = Signal()
    success_requested = Signal()
    fail_requested = Signal()
    rerecord_requested = Signal()
    intervention_requested = Signal()
    advance_requested = Signal()
    stop_recording_requested = Signal()
    health_check_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Evo-RL HIL Recorder")
        self.resize(1160, 760)
        self._config_warning: str | None = None
        saved_values, self._config_warning = load_config()

        root = QWidget()
        root.setObjectName("MainRoot")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        header = self._build_header()
        layout.addWidget(header)

        self.parameter_panel = ParameterPanel(values=saved_values)
        self.parameter_panel.setMinimumWidth(430)
        self.parameter_panel.setMaximumWidth(520)

        body = QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(self.parameter_panel)

        workbench = QVBoxLayout()
        workbench.setSpacing(14)
        workbench.addWidget(self._build_controls())
        workbench.addWidget(self._build_monitor())
        workbench.addWidget(self._build_log_panel(), 1)
        body.addLayout(workbench, 1)
        layout.addLayout(body, 1)

        self.start_button.clicked.connect(self.start_recording_requested.emit)
        self.success_button.clicked.connect(self.success_requested.emit)
        self.fail_button.clicked.connect(self.fail_requested.emit)
        self.rerecord_button.clicked.connect(self.rerecord_requested.emit)
        self.intervention_button.clicked.connect(self.intervention_requested.emit)
        self.advance_button.clicked.connect(self.advance_requested.emit)
        self.stop_recording_button.clicked.connect(self.stop_recording_requested.emit)
        self.health_button.clicked.connect(self.health_check_requested.emit)
        self._build_recording_shortcuts()

        self.set_recording_state(RecordingState.IDLE)
        if self._config_warning:
            self.append_log(f"[gui] {self._config_warning}\n")

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("HeaderFrame")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(14)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title = QLabel("Ola Data Recorder")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Dataset recording console for teleop, policy intervention, and hardware checks")
        subtitle.setObjectName("AppSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        layout.addLayout(title_block, 1)

        self.state_badge = QLabel("IDLE")
        self.state_badge.setObjectName("StateBadge")
        self.state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_badge.setMinimumWidth(108)
        layout.addWidget(self.state_badge)
        return header

    def _build_controls(self) -> QFrame:
        controls = QFrame()
        controls.setObjectName("ControlCard")
        layout = QGridLayout(controls)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)

        title = QLabel("Recording Controls")
        title.setObjectName("SectionTitle")
        hint = QLabel("Shortcuts: S success, F fail, I human/policy, Left rerecord, Right advance, Esc stop")
        hint.setObjectName("SectionHint")
        hint.setWordWrap(True)
        layout.addWidget(title, 0, 0, 1, 2)
        layout.addWidget(hint, 0, 2, 1, 4)

        self.start_button = self._button("开始录制", "fa5s.play", "primary")
        self.stop_recording_button = self._button("停止录制", "fa5s.stop", "danger")
        self.success_button = self._button("Success", "fa5s.check", "success")
        self.fail_button = self._button("Fail", "fa5s.times", "danger")
        self.rerecord_button = self._button("重录", "fa5s.redo-alt", "warning")
        self.intervention_button = self._button("Human/Policy", "fa5s.exchange-alt", "secondary")
        self.advance_button = self._button("结束当前阶段", "fa5s.arrow-right", "secondary")
        self.health_button = self._button("自检", "fa5s.stethoscope", "secondary")
        self.start_button.setMinimumWidth(150)
        self.stop_recording_button.setMinimumWidth(150)
        self.advance_button.setMinimumWidth(112)

        layout.addWidget(self.start_button, 1, 0, 1, 2)
        layout.addWidget(self.stop_recording_button, 1, 2, 1, 2)
        layout.addWidget(self.success_button, 1, 4)
        layout.addWidget(self.fail_button, 1, 5)
        layout.addWidget(self.rerecord_button, 2, 0)
        layout.addWidget(self.intervention_button, 2, 1, 1, 2)
        layout.addWidget(self.advance_button, 2, 3)
        layout.addWidget(self.health_button, 2, 4)

        for column in range(6):
            layout.setColumnStretch(column, 1)
        return controls

    def _build_monitor(self) -> QGroupBox:
        monitor = QGroupBox("Status Monitor")
        monitor_layout = QGridLayout(monitor)
        monitor_layout.setHorizontalSpacing(12)
        monitor_layout.setVerticalSpacing(12)
        self.camera_light = StatusLight("Camera", "fa5s.camera")
        self.can_light = StatusLight("CAN Status", "fa5s.network-wired")
        self.adb_light = StatusLight("ADB Devices", "fa5b.android")
        monitor_layout.addWidget(self.camera_light, 0, 0)
        monitor_layout.addWidget(self.can_light, 0, 1)
        monitor_layout.addWidget(self.adb_light, 0, 2)
        for column in range(3):
            monitor_layout.setColumnStretch(column, 1)
        return monitor

    def _build_log_panel(self) -> QGroupBox:
        log_panel = QGroupBox("Process Log")
        layout = QVBoxLayout(log_panel)
        self.log_view = QTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        log_font = QFont("monospace", 10)
        log_font.setStyleHint(QFont.StyleHint.Monospace)
        self.log_view.setFont(log_font)
        layout.addWidget(self.log_view)
        return log_panel

    def _button(self, text: str, icon_name: str, role: str) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("role", role)
        button.setIcon(_icon(icon_name))
        button.setIconSize(QSize(16, 16))
        button.setMinimumHeight(38)
        return button

    def _build_recording_shortcuts(self) -> None:
        self.recording_shortcuts: list[QShortcut] = []
        shortcut_specs = (
            ("S", self.success_requested.emit),
            ("F", self.fail_requested.emit),
            (Qt.Key.Key_Left, self.rerecord_requested.emit),
            ("I", self.intervention_requested.emit),
            (Qt.Key.Key_Right, self.advance_requested.emit),
            (Qt.Key.Key_Escape, self.stop_recording_requested.emit),
        )
        for key, callback in shortcut_specs:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)
            self.recording_shortcuts.append(shortcut)

    def parameters(self) -> RecordingParameters:
        return self.parameter_panel.parameters()

    def set_parameter(self, key: str, value: str) -> None:
        self.parameter_panel.set_parameter(key, value)

    def focus_parameter(self, key: str) -> None:
        self.parameter_panel.focus_parameter(key)

    def save_parameters(self) -> None:
        try:
            save_config(self.parameters().values)
        except Exception as exc:
            self.append_log(f"[gui] Failed to save config: {exc}\n")

    def append_log(self, message: str) -> None:
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(message)
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    def set_recording_state(self, state: RecordingState) -> None:
        running = state in {RecordingState.STARTING, RecordingState.RUNNING, RecordingState.RESETTING}
        self.state_badge.setText(STATE_LABELS[state])
        self.state_badge.setProperty("state", state.value)
        _refresh_dynamic_style(self.state_badge)
        self.parameter_panel.set_editable(not running)
        self.start_button.setEnabled(not running)
        self.success_button.setEnabled(running)
        self.fail_button.setEnabled(running)
        self.rerecord_button.setEnabled(running)
        self.intervention_button.setEnabled(running)
        self.advance_button.setEnabled(running)
        self.stop_recording_button.setEnabled(running)
        for shortcut in self.recording_shortcuts:
            shortcut.setEnabled(running)

    def set_health_checking(self) -> None:
        checking = HealthCheckResult("", StatusLevel.CHECKING, "Checking...")
        self.camera_light.set_result(checking)
        self.can_light.set_result(checking)
        self.adb_light.set_result(checking)

    def set_health_report(self, report: HardwareHealthReport) -> None:
        self.camera_light.set_result(report.camera)
        self.can_light.set_result(report.can)
        self.adb_light.set_result(report.adb)

    def closeEvent(self, event) -> None:
        controller = getattr(self, "controller", None)
        if controller is not None and hasattr(controller, "shutdown"):
            controller.shutdown()
        self.save_parameters()
        super().closeEvent(event)
