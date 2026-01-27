from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class BodyScaleDialog(QDialog):
    def __init__(self, body_name: str, distance_scale: float, radius_scale: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Scale {body_name}")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Adjust scales for {body_name}", self))

        form = QFormLayout()
        self.distance_spin = QDoubleSpinBox(self)
        self.distance_spin.setDecimals(6)
        self.distance_spin.setRange(0.001, 1000.0)
        self.distance_spin.setSingleStep(0.05)
        self.distance_spin.setValue(distance_scale)

        self.radius_spin = QDoubleSpinBox(self)
        self.radius_spin.setDecimals(6)
        self.radius_spin.setRange(0.001, 1000.0)
        self.radius_spin.setSingleStep(0.05)
        self.radius_spin.setValue(radius_scale)

        form.addRow("Distance scale (x)", self.distance_spin)
        form.addRow("Size scale (x)", self.radius_spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[float, float]:
        return (self.distance_spin.value(), self.radius_spin.value())


class TimeProgressionOverlay(QFrame):
    speedChanged = Signal(float)
    playToggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._speed = 0.0
        self._running = False
        self._min_speed = 0.01
        self._max_speed = 100.0 * 365.25 * 86400.0
        self._default_speed = 86400.0

        self.setObjectName("TimeProgressionOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title = QLabel("Time Warp", self)
        title.setObjectName("TimeWarpTitle")
        layout.addWidget(title)

        self.speed_label = QLabel("", self)
        self.speed_label.setObjectName("TimeWarpSpeed")
        layout.addWidget(self.speed_label)

        hint = QLabel("Scroll to change speed", self)
        hint.setObjectName("TimeWarpHint")
        layout.addWidget(hint)

        button_row = QHBoxLayout()
        self.play_button = QPushButton("Play", self)
        self.play_button.setObjectName("TimeWarpPlay")
        self.play_button.clicked.connect(self._toggle_play)
        button_row.addWidget(self.play_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.setStyleSheet(
            """
            QFrame#TimeProgressionOverlay {
                background-color: rgba(20, 24, 36, 180);
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 10px;
            }
            QLabel#TimeWarpTitle {
                color: #f0f4ff;
                font-weight: 600;
            }
            QLabel#TimeWarpSpeed {
                color: #d7f6ff;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#TimeWarpHint {
                color: rgba(230, 240, 255, 160);
                font-size: 10px;
            }
            QPushButton#TimeWarpPlay {
                background-color: rgba(255, 255, 255, 28);
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 6px;
                color: #eef6ff;
                padding: 4px 10px;
            }
            QPushButton#TimeWarpPlay:hover {
                background-color: rgba(255, 255, 255, 44);
            }
            """
        )

        self._update_speed_label()

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.0, min(self._max_speed, float(speed)))
        self._update_speed_label()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.play_button.setText("Pause" if self._running else "Play")

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        if self._speed <= 0.0:
            candidate = self._default_speed
        else:
            candidate = self._speed
        factor = 1.2 ** steps
        next_speed = candidate * factor
        next_speed = max(self._min_speed, min(self._max_speed, next_speed))
        self.speedChanged.emit(next_speed)
        event.accept()

    def _toggle_play(self) -> None:
        self._running = not self._running
        self.play_button.setText("Pause" if self._running else "Play")
        self.playToggled.emit(self._running)

    def _update_speed_label(self) -> None:
        if self._speed <= 0.0:
            self.speed_label.setText("Paused")
            return

        unit, unit_seconds = self._pick_unit(self._speed)
        value = self._speed / unit_seconds
        formatted = f"{value:.3g}"
        unit_label = unit if abs(value - 1.0) < 0.001 else f"{unit}s"
        self.speed_label.setText(f"{formatted} {unit_label}/sec")

    def _pick_unit(self, speed: float) -> tuple[str, float]:
        units = (
            ("sec", 1.0),
            ("min", 60.0),
            ("hour", 3600.0),
            ("day", 86400.0),
            ("year", 365.25 * 86400.0),
        )
        best = None
        best_score = None
        for name, seconds in units:
            value = speed / seconds
            if value <= 0:
                continue
            score = abs(math.log10(value))
            if best_score is None or score < best_score:
                best_score = score
                best = (name, seconds)
        return best if best is not None else ("sec", 1.0)
