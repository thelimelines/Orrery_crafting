from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .controllers.diagnostics import DiagnosticsMixin
from .controllers.kernels import KernelControlMixin
from .controllers.loading import LoadingMixin
from .controllers.mechanical_layout import MechanicalLayoutMixin
from .controllers.orbits import OrbitPathMixin
from .controllers.positions import PositionMixin
from .controllers.time_controls import TimeControlMixin
from .models import BODY_CATALOG, BodySettings, BodyState, OrbitStats
from .opengl_view import OrreryGLWidget
from .spice_service import SpiceEphemeris
from .ui.panels.gear_panel import GearPanelMixin, GearTrainEntry
from .ui.panels.orrery_panel import OrreryPanelMixin
from .widgets import TimeProgressionOverlay


class MainWindow(
    QMainWindow,
    OrreryPanelMixin,
    GearPanelMixin,
    LoadingMixin,
    KernelControlMixin,
    TimeControlMixin,
    PositionMixin,
    OrbitPathMixin,
    MechanicalLayoutMixin,
    DiagnosticsMixin,
):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Orrery Crafting - SPICE Viewer")
        self.resize(1300, 800)

        self._settings: dict[str, BodySettings] = {
            body.name: BodySettings() for body in BODY_CATALOG
        }
        self._positions: dict[str, BodyState] = {}
        self._rotations: dict[str, np.ndarray] = {}
        self._spin_dirs: dict[str, float] = {}
        self._orbit_stats: dict[str, OrbitStats] = {}
        self._orbit_stats_key: tuple[int, int] | None = None
        self._moon_bands_by_parent: dict[str, tuple[tuple[float, float], ...]] = {}
        self._focused_system: str | None = None
        self._did_initial_load = False
        self._kernels_loaded = False
        self._textures_loaded = False
        self._gl_ready = False
        self._textures_started = False
        self._focused_body: str | None = None
        self._solar_plane_normal: tuple[float, float, float] | None = None
        self._moon_bands_by_parent: dict[str, tuple[tuple[float, float], ...]] = {}
        self._focused_system: str | None = None
        self._orbit_paths: dict[str, np.ndarray] = {}
        self._orbit_paths_key: tuple[int, int, object, bool, str | None, float, float, float, float] | None = None
        self._orbit_path_centers: dict[str, str | None] = {}
        self._planet_bounds: dict[str, tuple[float, float]] = {}
        self._planet_scale = 1.0
        self._moon_scaled_bounds: dict[str, tuple[float, float]] = {}
        self._moon_scale_by_parent: dict[str, float] = {}

        self.spice = SpiceEphemeris()

        self.gl_widget = OrreryGLWidget(self)
        self.gl_widget.set_body_settings(self._settings)
        self.gl_widget.bodyDoubleClicked.connect(self._focus_body_from_view)
        self.gl_widget.glInitialized.connect(self._mark_gl_ready)
        self.gl_widget.glInitialized.connect(self._maybe_start_initial_load)
        self.gl_widget.texturesLoaded.connect(self._mark_textures_loaded)
        self.gl_widget.gearClicked.connect(self._set_selected_gear)

        self._time_speed = 0.0
        self._last_nonzero_speed = 86400.0
        self._time_last_tick = None
        self._time_timer = QTimer(self)
        self._time_timer.setInterval(33)
        self._time_timer.timeout.connect(self._advance_time)

        self._gear_trains: list[GearTrainEntry] = []
        self._gear_speed_by_id: dict[str, float] = {}
        self._gear_shaft_by_id: dict[str, int] = {}
        self._gear_label_by_id: dict[str, str] = {}
        self._gear_arms: dict[str, object] = {}
        self._arm_angles: dict[str, float] = {}
        self._gear_angles: dict[str, float] = {}
        self._selected_input_gear_id = "drive"

        self.time_overlay = TimeProgressionOverlay(self.gl_widget)
        self.time_overlay.speedChanged.connect(self._set_time_speed)
        self.time_overlay.playToggled.connect(self._toggle_time_playback)

        overlay_layout = QVBoxLayout(self.gl_widget)
        overlay_layout.setContentsMargins(12, 12, 12, 12)
        overlay_layout.setSpacing(0)
        overlay_layout.addWidget(self.time_overlay, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        overlay_layout.addStretch(1)

        left_panel = self._build_left_panel()
        splitter = QSplitter(self)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.gl_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 980])

        root = QWidget(self)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)
        self._loading_overlay = self._build_loading_overlay(root)
        self._kernel_thread: QThread | None = None
        self._kernel_worker = None

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._maybe_start_initial_load()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_loading_overlay") and self._loading_overlay is not None:
            self._loading_overlay.setGeometry(self.centralWidget().rect())


def _configure_opengl() -> None:
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)


def main() -> None:
    _configure_opengl()
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
