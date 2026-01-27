from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QDateTime, QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QToolButton,
    QDateTimeEdit,
    QFileDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QMessageBox,
    QSlider,
    QComboBox,
)
from PySide6.QtGui import QSurfaceFormat

from Gear_calc import compute_gear_train
from .gear_models import ArmRenderSpec, GearRenderSpec
from .models import BODY_BY_NAME, BODY_CATALOG, BodySettings, OrbitStats, BodyState
from .opengl_view import OrreryGLWidget
from .spice_service import SpiceEphemeris
from .widgets import TimeProgressionOverlay


class KernelLoadWorker(QObject):
    finished = Signal(str, object, object)

    def __init__(self, spice: SpiceEphemeris, kernel_dir: Path) -> None:
        super().__init__()
        self._spice = spice
        self._kernel_dir = kernel_dir

    def run(self) -> None:
        try:
            result = self._spice.load_kernels(self._kernel_dir)
            status = self._spice.status_message
            error = None
        except Exception as exc:  # noqa: BLE001
            result = None
            status = "Kernel load failed"
            error = exc
        self.finished.emit(status, result, error)


@dataclass
class GearTrainEntry:
    train_id: str
    target_body: str
    gear_pairs: List[Tuple[int, int]]
    input_gear_id: str
    ratio_target: float
    ratio_achieved: float
    gear_string: str


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Orrery Crafting - SPICE Viewer")
        self.resize(1300, 800)

        self._settings: Dict[str, BodySettings] = {
            body.name: BodySettings() for body in BODY_CATALOG
        }
        self._positions = {}
        self._rotations: Dict[str, np.ndarray] = {}
        self._spin_dirs: Dict[str, float] = {}
        self._orbit_stats: Dict[str, OrbitStats] = {}
        self._orbit_stats_key: Optional[Tuple[str, int, int]] = None
        self._moon_bands_by_parent: Dict[str, Tuple[Tuple[float, float], ...]] = {}
        self._focused_system: Optional[str] = None
        self._did_initial_load = False
        self._kernels_loaded = False
        self._textures_loaded = False
        self._gl_ready = False
        self._textures_started = False
        self._focused_body: Optional[str] = None
        self._solar_plane_normal: Optional[Tuple[float, float, float]] = None
        self._moon_bands_by_parent: Dict[str, Tuple[Tuple[float, float], ...]] = {}
        self._focused_system: Optional[str] = None
        self._orbit_paths: Dict[str, np.ndarray] = {}
        self._orbit_paths_key: Optional[Tuple[int, int, datetime, bool, Optional[str], float, float, float, float]] = None
        self._orbit_path_centers: Dict[str, Optional[str]] = {}
        self._planet_bounds: Dict[str, Tuple[float, float]] = {}
        self._planet_scale = 1.0
        self._moon_scaled_bounds: Dict[str, Tuple[float, float]] = {}
        self._moon_scale_by_parent: Dict[str, float] = {}

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

        self._gear_trains: List[GearTrainEntry] = []
        self._gear_speed_by_id: Dict[str, float] = {}
        self._gear_shaft_by_id: Dict[str, int] = {}
        self._gear_label_by_id: Dict[str, str] = {}
        self._gear_arms: Dict[str, ArmRenderSpec] = {}
        self._arm_angles: Dict[str, float] = {}
        self._gear_angles: Dict[str, float] = {}
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
        self._kernel_thread: Optional[QThread] = None
        self._kernel_worker: Optional[KernelLoadWorker] = None

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._maybe_start_initial_load()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_loading_overlay") and self._loading_overlay is not None:
            self._loading_overlay.setGeometry(self.centralWidget().rect())

    def _maybe_start_initial_load(self) -> None:
        if self._did_initial_load:
            return
        self._did_initial_load = True
        self._show_loading_overlay(True)
        self._start_textures_if_ready()
        QTimer.singleShot(0, self._start_kernel_load)

    def _mark_gl_ready(self) -> None:
        self._gl_ready = True
        self._start_textures_if_ready()

    def _start_textures_if_ready(self) -> None:
        if not self._gl_ready or self._textures_started:
            return
        self._textures_started = True
        QTimer.singleShot(0, self.gl_widget.load_textures)

    def _start_kernel_load(self) -> None:
        self._set_loading_step("Loading SPICE kernels…")
        self._kernel_thread = QThread(self)
        self._kernel_worker = KernelLoadWorker(self.spice, self.spice.kernel_dir)
        self._kernel_worker.moveToThread(self._kernel_thread)
        self._kernel_thread.started.connect(self._kernel_worker.run)
        self._kernel_worker.finished.connect(self._finish_kernel_load)
        self._kernel_worker.finished.connect(self._kernel_thread.quit)
        self._kernel_worker.finished.connect(self._kernel_worker.deleteLater)
        self._kernel_thread.finished.connect(self._kernel_thread.deleteLater)
        self._kernel_thread.start()

    def _finish_kernel_load(self, status_message: str, result, error) -> None:
        self.kernel_status.setText(f"Status: {status_message}")
        if error is not None:
            QMessageBox.critical(
                self,
                "Kernel load failed",
                f"Kernel load failed:\n{error}",
            )
        elif result and getattr(result, "errors", None):
            msg = "Some kernels failed to load:\n" + "\n".join(result.errors[:6])
            if len(result.errors) > 6:
                msg += f"\n...and {len(result.errors) - 6} more"
            QMessageBox.warning(self, "Kernel load issues", msg)
        self._set_loading_step("Preparing layout…")
        self._apply_mechanical_settings()
        self._kernels_loaded = True
        self._finalize_loading_if_ready()

    def _build_left_panel(self) -> QWidget:
        panel = QFrame(self)
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setMinimumWidth(280)

        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setSpacing(12)

        title = QLabel("Orrery Controls", content)
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        kernel_group = QGroupBox("SPICE Kernels", content)
        kernel_layout = QVBoxLayout(kernel_group)
        self.kernel_status = QLabel(f"Status: {self.spice.status_message}", kernel_group)
        kernel_layout.addWidget(self.kernel_status)
        self.kernel_path = QLineEdit(str(self.spice.kernel_dir), kernel_group)
        kernel_layout.addWidget(self.kernel_path)
        kernel_buttons = QHBoxLayout()
        browse = QPushButton("Browse", kernel_group)
        browse.clicked.connect(self._browse_kernel_dir)
        load = QPushButton("Load", kernel_group)
        load.clicked.connect(self._load_kernels)
        kernel_buttons.addWidget(browse)
        kernel_buttons.addWidget(load)
        kernel_layout.addLayout(kernel_buttons)

        time_group = QGroupBox("Time (UTC)", content)
        time_layout = QVBoxLayout(time_group)
        self.time_edit = QDateTimeEdit(QDateTime.currentDateTimeUtc(), time_group)
        self.time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss 'UTC'")
        self.time_edit.setCalendarPopup(True)
        time_layout.addWidget(self.time_edit)
        time_buttons = QHBoxLayout()
        now = QPushButton("Now", time_group)
        now.clicked.connect(self._set_now)
        refresh = QPushButton("Refresh", time_group)
        refresh.clicked.connect(self._refresh_positions)
        time_buttons.addWidget(now)
        time_buttons.addWidget(refresh)
        time_layout.addLayout(time_buttons)
        layout.addWidget(time_group)

        locate_group = QGroupBox("Quick Locate / Edit", content)
        locate_layout = QVBoxLayout(locate_group)
        self.body_list = QListWidget(locate_group)
        for body in BODY_CATALOG:
            QListWidgetItem(body.name, self.body_list)
        self.body_list.itemDoubleClicked.connect(self._focus_selected_body)
        locate_layout.addWidget(self.body_list)
        focus_btn = QPushButton("Focus Selected", locate_group)
        focus_btn.clicked.connect(self._focus_selected_body)
        locate_layout.addWidget(focus_btn)
        self.missing_label = QLabel("Missing bodies: 0", locate_group)
        locate_layout.addWidget(self.missing_label)
        layout.addWidget(locate_group)

        diagnostics_group = QGroupBox("Diagnostics", content)
        diagnostics_layout = QVBoxLayout(diagnostics_group)
        self.diagnostics = QPlainTextEdit(diagnostics_group)
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setMaximumBlockCount(200)
        diagnostics_layout.addWidget(self.diagnostics)

        mechanical_group = QGroupBox("Mechanical Layout", content)
        mechanical_layout = QVBoxLayout(mechanical_group)
        self.mechanical_check = QCheckBox("Enable mechanical layout", mechanical_group)
        self.mechanical_check.setChecked(True)
        self.mechanical_check.toggled.connect(self._apply_mechanical_settings)
        mechanical_layout.addWidget(self.mechanical_check)
        self.earth_size_spin = QDoubleSpinBox(mechanical_group)
        self.earth_size_spin.setDecimals(2)
        self.earth_size_spin.setRange(1.0, 100.0)
        self.earth_size_spin.setSingleStep(0.5)
        self.earth_size_spin.setValue(self.gl_widget.earth_diameter_mm)
        self.earth_size_spin.valueChanged.connect(self._apply_mechanical_settings)
        mechanical_layout.addWidget(QLabel("Earth diameter (mm)", mechanical_group))
        mechanical_layout.addWidget(self.earth_size_spin)
        self.log_base_spin = QDoubleSpinBox(mechanical_group)
        self.log_base_spin.setDecimals(2)
        self.log_base_spin.setRange(2.0, 20.0)
        self.log_base_spin.setSingleStep(0.5)
        self.log_base_spin.setValue(self.gl_widget.size_log_base)
        self.log_base_spin.valueChanged.connect(self._apply_mechanical_settings)
        mechanical_layout.addWidget(QLabel("Size log base", mechanical_group))
        mechanical_layout.addWidget(self.log_base_spin)
        self.planet_clearance_spin = QDoubleSpinBox(mechanical_group)
        self.planet_clearance_spin.setDecimals(2)
        self.planet_clearance_spin.setRange(0.0, 50.0)
        self.planet_clearance_spin.setSingleStep(0.5)
        self.planet_clearance_spin.setValue(5.0)
        self.planet_clearance_spin.valueChanged.connect(self._apply_mechanical_settings)
        mechanical_layout.addWidget(QLabel("Planet-planet clearance (mm)", mechanical_group))
        mechanical_layout.addWidget(self.planet_clearance_spin)
        self.moon_clearance_spin = QDoubleSpinBox(mechanical_group)
        self.moon_clearance_spin.setDecimals(2)
        self.moon_clearance_spin.setRange(0.0, 50.0)
        self.moon_clearance_spin.setSingleStep(0.5)
        self.moon_clearance_spin.setValue(5.0)
        self.moon_clearance_spin.valueChanged.connect(self._apply_mechanical_settings)
        mechanical_layout.addWidget(QLabel("Planet-moon clearance (mm)", mechanical_group))
        mechanical_layout.addWidget(self.moon_clearance_spin)
        self.sample_spin = QSpinBox(mechanical_group)
        self.sample_spin.setRange(60, 1000)
        self.sample_spin.setSingleStep(20)
        self.sample_spin.setValue(180)
        self.sample_spin.valueChanged.connect(self._apply_mechanical_settings)
        mechanical_layout.addWidget(QLabel("Orbit samples (per body)", mechanical_group))
        mechanical_layout.addWidget(self.sample_spin)
        self.span_orbits_spin = QSpinBox(mechanical_group)
        self.span_orbits_spin.setRange(1, 500)
        self.span_orbits_spin.setSingleStep(10)
        self.span_orbits_spin.setValue(100)
        self.span_orbits_spin.valueChanged.connect(self._apply_mechanical_settings)
        mechanical_layout.addWidget(QLabel("Orbit span (orbits)", mechanical_group))
        mechanical_layout.addWidget(self.span_orbits_spin)
        self.bands_check = QCheckBox("Show orbit bands", mechanical_group)
        self.bands_check.setChecked(True)
        self.bands_check.toggled.connect(self._apply_mechanical_settings)
        mechanical_layout.addWidget(self.bands_check)
        self.orbit_paths_check = QCheckBox("Show orbital paths (3D spline)", mechanical_group)
        self.orbit_paths_check.setChecked(False)
        self.orbit_paths_check.setToolTip("Shows spline orbital paths in the OpenGL view.")
        self.orbit_paths_check.toggled.connect(self._toggle_orbit_paths)
        mechanical_layout.addWidget(self.orbit_paths_check)
        self.orbit_paths_focus_check = QCheckBox("Orbit paths: focused body only", mechanical_group)
        self.orbit_paths_focus_check.setChecked(False)
        self.orbit_paths_focus_check.toggled.connect(self._toggle_orbit_paths_focus)
        mechanical_layout.addWidget(self.orbit_paths_focus_check)
        self.band_opacity_label = QLabel("Orbit band opacity", mechanical_group)
        self.band_opacity_slider = QSlider(Qt.Orientation.Horizontal, mechanical_group)
        self.band_opacity_slider.setRange(0, 100)
        initial_opacity = int(round(self.gl_widget.orbit_band_opacity * 100))
        self.band_opacity_slider.setValue(initial_opacity)
        self.band_opacity_slider.valueChanged.connect(self._apply_band_opacity)
        mechanical_layout.addWidget(self.band_opacity_label)
        mechanical_layout.addWidget(self.band_opacity_slider)
        self.real_scale_btn = QPushButton("Real scale (sizes + distances)", mechanical_group)
        self.real_scale_btn.clicked.connect(self._apply_real_scale)
        mechanical_layout.addWidget(self.real_scale_btn)
        layout.addWidget(mechanical_group)

        layout.addStretch(1)
        debug_container = QWidget(content)
        debug_layout = QVBoxLayout(debug_container)
        debug_layout.setContentsMargins(0, 0, 0, 0)
        debug_toggle = QToolButton(debug_container)
        debug_toggle.setText("Debug")
        debug_toggle.setCheckable(True)
        debug_toggle.setChecked(False)
        debug_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        debug_toggle.setArrowType(Qt.ArrowType.RightArrow)
        debug_layout.addWidget(debug_toggle)

        debug_body = QWidget(debug_container)
        debug_body_layout = QVBoxLayout(debug_body)
        debug_body_layout.addWidget(kernel_group)
        debug_body_layout.addWidget(diagnostics_group)
        debug_body.setVisible(False)
        debug_layout.addWidget(debug_body)

        def _toggle_debug(checked: bool) -> None:
            debug_body.setVisible(checked)
            debug_toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

        debug_toggle.toggled.connect(_toggle_debug)
        layout.addWidget(debug_container)
        scroll.setWidget(content)

        tabs = QTabWidget(panel)
        tabs.addTab(scroll, "Orrery")
        tabs.addTab(self._build_gear_panel(), "Gears")

        frame_layout = QVBoxLayout(panel)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(tabs)
        return panel

    def _build_gear_panel(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setSpacing(12)

        title = QLabel("Gear Modeling", content)
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        drive_group = QGroupBox("Drive Gear", content)
        drive_layout = QFormLayout(drive_group)
        self.drive_speed_spin = QDoubleSpinBox(drive_group)
        self.drive_speed_spin.setDecimals(4)
        self.drive_speed_spin.setRange(0.0, 1_000_000.0)
        self.drive_speed_spin.setValue(1.0)
        self.drive_speed_spin.setSuffix(" rev/day")
        self.drive_speed_spin.valueChanged.connect(self._rebuild_gear_layout)
        drive_layout.addRow("Drive speed", self.drive_speed_spin)

        self.drive_teeth_spin = QSpinBox(drive_group)
        self.drive_teeth_spin.setRange(6, 400)
        self.drive_teeth_spin.setValue(60)
        self.drive_teeth_spin.valueChanged.connect(self._rebuild_gear_layout)
        drive_layout.addRow("Drive teeth", self.drive_teeth_spin)

        self.drive_reverse_check = QCheckBox("Revert drive rotation", drive_group)
        self.drive_reverse_check.setChecked(False)
        self.drive_reverse_check.toggled.connect(self._rebuild_gear_layout)
        drive_layout.addRow(self.drive_reverse_check)
        layout.addWidget(drive_group)

        input_group = QGroupBox("Input Gear", content)
        input_layout = QVBoxLayout(input_group)
        self.selected_gear_label = QLabel("Input: Drive", input_group)
        input_layout.addWidget(self.selected_gear_label)
        self.gear_list = QListWidget(input_group)
        self.gear_list.itemClicked.connect(self._select_gear_from_list)
        input_layout.addWidget(self.gear_list)
        layout.addWidget(input_group)

        train_group = QGroupBox("Gear Train", content)
        train_layout = QFormLayout(train_group)
        self.target_body_combo = QComboBox(train_group)
        for body in BODY_CATALOG:
            if body.parent.upper() == "SUN" and body.name != "Sun":
                self.target_body_combo.addItem(body.name)
        train_layout.addRow("Target body", self.target_body_combo)

        self.stages_spin = QSpinBox(train_group)
        self.stages_spin.setRange(1, 12)
        self.stages_spin.setValue(4)
        train_layout.addRow("Stages", self.stages_spin)

        self.min_teeth_spin = QSpinBox(train_group)
        self.min_teeth_spin.setRange(3, 400)
        self.min_teeth_spin.setValue(10)
        train_layout.addRow("Min teeth", self.min_teeth_spin)

        self.max_teeth_spin = QSpinBox(train_group)
        self.max_teeth_spin.setRange(4, 400)
        self.max_teeth_spin.setValue(120)
        train_layout.addRow("Max teeth", self.max_teeth_spin)

        self.tolerance_spin = QDoubleSpinBox(train_group)
        self.tolerance_spin.setDecimals(9)
        self.tolerance_spin.setRange(0.0, 1.0)
        self.tolerance_spin.setSingleStep(1e-6)
        self.tolerance_spin.setValue(1e-6)
        train_layout.addRow("Tolerance", self.tolerance_spin)

        self.use_numba_check = QCheckBox("Use Numba (if available)", train_group)
        self.use_numba_check.setChecked(True)
        train_layout.addRow(self.use_numba_check)

        train_buttons = QHBoxLayout()
        self.add_train_btn = QPushButton("Generate Train", train_group)
        self.add_train_btn.clicked.connect(self._add_gear_train)
        self.clear_trains_btn = QPushButton("Clear Trains", train_group)
        self.clear_trains_btn.clicked.connect(self._clear_gear_trains)
        train_buttons.addWidget(self.add_train_btn)
        train_buttons.addWidget(self.clear_trains_btn)
        train_layout.addRow(train_buttons)
        layout.addWidget(train_group)

        layout_group = QGroupBox("Layout", content)
        layout_form = QFormLayout(layout_group)
        self.gear_scale_spin = QDoubleSpinBox(layout_group)
        self.gear_scale_spin.setDecimals(3)
        self.gear_scale_spin.setRange(0.1, 50.0)
        self.gear_scale_spin.setSingleStep(0.1)
        self.gear_scale_spin.setValue(2.0)
        self.gear_scale_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Gear scale (mm)", self.gear_scale_spin)

        self.gear_thickness_spin = QDoubleSpinBox(layout_group)
        self.gear_thickness_spin.setDecimals(2)
        self.gear_thickness_spin.setRange(0.5, 50.0)
        self.gear_thickness_spin.setSingleStep(0.5)
        self.gear_thickness_spin.setValue(4.0)
        self.gear_thickness_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Gear thickness (mm)", self.gear_thickness_spin)

        self.shaft_clearance_spin = QDoubleSpinBox(layout_group)
        self.shaft_clearance_spin.setDecimals(2)
        self.shaft_clearance_spin.setRange(0.0, 50.0)
        self.shaft_clearance_spin.setSingleStep(0.5)
        self.shaft_clearance_spin.setValue(0.0)
        self.shaft_clearance_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Shaft clearance (mm)", self.shaft_clearance_spin)

        self.vertical_clearance_spin = QDoubleSpinBox(layout_group)
        self.vertical_clearance_spin.setDecimals(2)
        self.vertical_clearance_spin.setRange(0.0, 50.0)
        self.vertical_clearance_spin.setSingleStep(0.5)
        self.vertical_clearance_spin.setValue(1.0)
        self.vertical_clearance_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Vertical clearance (mm)", self.vertical_clearance_spin)

        self.arm_clearance_spin = QDoubleSpinBox(layout_group)
        self.arm_clearance_spin.setDecimals(2)
        self.arm_clearance_spin.setRange(0.0, 200.0)
        self.arm_clearance_spin.setSingleStep(1.0)
        self.arm_clearance_spin.setValue(0.0)
        self.arm_clearance_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Arm lift (mm)", self.arm_clearance_spin)

        self.arm_vertical_clearance_spin = QDoubleSpinBox(layout_group)
        self.arm_vertical_clearance_spin.setDecimals(2)
        self.arm_vertical_clearance_spin.setRange(0.0, 200.0)
        self.arm_vertical_clearance_spin.setSingleStep(1.0)
        self.arm_vertical_clearance_spin.setValue(6.0)
        self.arm_vertical_clearance_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Arm vertical clearance (mm)", self.arm_vertical_clearance_spin)

        layout.addWidget(layout_group)

        self.gear_log = QPlainTextEdit(content)
        self.gear_log.setReadOnly(True)
        self.gear_log.setMaximumBlockCount(200)
        layout.addWidget(self.gear_log)

        layout.addStretch(1)
        scroll.setWidget(content)

        self._rebuild_gear_layout()
        return scroll

    def _build_loading_overlay(self, parent: QWidget) -> QWidget:
        overlay = QFrame(parent)
        overlay.setObjectName("loadingOverlay")
        overlay.setVisible(False)
        overlay.setStyleSheet(
            "QFrame#loadingOverlay {"
            "  background-color: rgba(12, 14, 20, 220);"
            "}"
            "QLabel {"
            "  color: #F2F4F7;"
            "}"
        )
        layout = QVBoxLayout(overlay)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        title = QLabel("Loading Orrery Crafting…", overlay)
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._loading_step_label = QLabel("Initializing…", overlay)
        self._loading_step_label.setFont(QFont("Segoe UI", 10))
        self._loading_step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._loading_step_label)

        tips = QLabel(
            "What is loading:\n"
            "• SPICE kernels and ephemerides\n"
            "• Orbit stats and mechanical layout\n"
            "• Textures and render setup",
            overlay,
        )
        tips.setFont(QFont("Segoe UI", 9))
        tips.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tips)

        return overlay

    def _drive_speed_rps(self) -> float:
        if not hasattr(self, "drive_speed_spin"):
            return 0.0
        speed = -float(self.drive_speed_spin.value()) / 86400.0
        if hasattr(self, "drive_reverse_check") and self.drive_reverse_check.isChecked():
            speed *= -1.0
        return speed

    def _parse_gear_string(self, gear_string: str) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for token in gear_string.split(","):
            part = token.strip()
            if not part:
                continue
            if part.startswith("R"):
                part = part[1:]
            if ":" not in part:
                continue
            left, right = part.split(":", 1)
            try:
                driven = int(left)
                driver = int(right)
                pairs.append((driver, driven))
            except ValueError:
                continue
        return pairs

    def _select_gear_from_list(self, item: QListWidgetItem) -> None:
        gear_id = item.data(Qt.ItemDataRole.UserRole)
        if gear_id:
            self._set_selected_gear(gear_id)

    def _set_selected_gear(self, gear_id: str) -> None:
        if gear_id not in self._gear_label_by_id:
            return
        self._selected_input_gear_id = gear_id
        label = self._gear_label_by_id.get(gear_id, gear_id)
        self.selected_gear_label.setText(f"Input: {label}")
        for idx in range(self.gear_list.count()):
            item = self.gear_list.item(idx)
            if item.data(Qt.ItemDataRole.UserRole) == gear_id:
                self.gear_list.setCurrentItem(item)
                break

    def _update_gear_list(self) -> None:
        self.gear_list.clear()
        for gear_id, label in self._gear_label_by_id.items():
            speed = self._gear_speed_by_id.get(gear_id, 0.0) * 86400.0
            item = QListWidgetItem(f"{label}  •  {speed:.4g} rev/day", self.gear_list)
            item.setData(Qt.ItemDataRole.UserRole, gear_id)
            if gear_id == self._selected_input_gear_id:
                item.setSelected(True)

    def _add_gear_train(self) -> None:
        target_name = self.target_body_combo.currentText().strip()
        body = BODY_BY_NAME.get(target_name)
        if not body or body.parent.upper() != "SUN" or body.name == "Sun":
            QMessageBox.warning(self, "Gear train", "Select a valid planet target.")
            return
        if body.orbital_period_days is None or body.orbital_period_days <= 0:
            QMessageBox.warning(self, "Gear train", "Target body has no orbital period data.")
            return

        input_speed_rps = self._gear_speed_by_id.get(self._selected_input_gear_id, 0.0)
        if input_speed_rps == 0.0:
            QMessageBox.warning(self, "Gear train", "Input gear speed must be non-zero.")
            return

        input_speed_day = input_speed_rps * 86400.0
        target_speed_day = 1.0 / float(body.orbital_period_days)
        ratio = abs(target_speed_day / input_speed_day) if input_speed_day != 0 else 0.0
        if ratio <= 0.0:
            QMessageBox.warning(self, "Gear train", "Computed target ratio is invalid.")
            return

        min_teeth = int(self.min_teeth_spin.value())
        max_teeth = int(self.max_teeth_spin.value())
        if max_teeth <= min_teeth:
            QMessageBox.warning(self, "Gear train", "Max teeth must be greater than min teeth.")
            return

        stages = int(self.stages_spin.value())
        tol = float(self.tolerance_spin.value())
        use_numba = self.use_numba_check.isChecked()

        result = compute_gear_train(
            ratio=ratio,
            tol=tol,
            stages=stages,
            minw=min_teeth,
            maxw=max_teeth,
            use_numba=use_numba,
            progress_callback=lambda _state: None,
        )
        if result.get("status") != "ok":
            reason = result.get("reason", "No solution found.")
            QMessageBox.warning(self, "Gear train", f"Gear train search failed: {reason}")
            return

        gear_string = result.get("gear_string", "")
        pairs = self._parse_gear_string(gear_string)
        if not pairs:
            QMessageBox.warning(self, "Gear train", "No gear pairs were returned.")
            return

        train_id = f"T{len(self._gear_trains) + 1}"
        self._gear_trains.append(
            GearTrainEntry(
                train_id=train_id,
                target_body=target_name,
                gear_pairs=pairs,
                input_gear_id=self._selected_input_gear_id,
                ratio_target=ratio,
                ratio_achieved=float(result.get("ratio_achieved", 0.0)),
                gear_string=gear_string,
            )
        )

        arm_id = f"arm_{train_id}"
        self._arm_angles[arm_id] = self._current_planet_angle(target_name) or 0.0

        achieved = float(result.get("ratio_achieved", 0.0))
        err = float(result.get("error", 0.0))
        self.gear_log.appendPlainText(
            f"[{train_id}] {target_name} ratio {ratio:.6g} -> {achieved:.6g} "
            f"(err {err:.3g}) | {gear_string}"
        )
        self._rebuild_gear_layout()

    def _clear_gear_trains(self) -> None:
        self._gear_trains.clear()
        self._arm_angles.clear()
        self._selected_input_gear_id = "drive"
        if hasattr(self, "gear_log"):
            self.gear_log.setPlainText("")
        self._rebuild_gear_layout()

    def _scaled_planet_mean_distance(self, name: str) -> Optional[float]:
        body = BODY_BY_NAME.get(name)
        if not body or body.mean_distance_km is None:
            return None
        actual = float(body.mean_distance_km)
        if self.mechanical_check.isChecked():
            stats = self._orbit_stats.get(name)
            bounds = self._planet_bounds.get(name)
            if stats and bounds and stats.max_km > stats.min_km:
                t = (actual - stats.min_km) / (stats.max_km - stats.min_km)
                t = max(0.0, min(1.0, t))
                return bounds[0] + t * (bounds[1] - bounds[0])
            if self._planet_scale:
                return actual * self._planet_scale
        settings = self._settings.get(name, BodySettings())
        return actual * self.gl_widget.global_distance_scale * settings.distance_scale

    def _scaled_planet_position(self, name: str) -> Optional[np.ndarray]:
        state = self._positions.get(name)
        if not state or not state.available:
            return None
        vec = np.array(state.position_km, dtype=float)
        if self.mechanical_check.isChecked():
            stats = self._orbit_stats.get(name)
            bounds = self._planet_bounds.get(name)
            r_actual = float(np.linalg.norm(vec))
            if stats and bounds and stats.max_km > stats.min_km and r_actual > 0:
                direction = vec / r_actual
                t = (r_actual - stats.min_km) / (stats.max_km - stats.min_km)
                t = max(0.0, min(1.0, t))
                scaled_r = bounds[0] + t * (bounds[1] - bounds[0])
                scaled_vec = direction * scaled_r
            elif self._planet_scale:
                scaled_vec = vec * self._planet_scale
            else:
                scaled_vec = vec
        else:
            settings = self._settings.get(name, BodySettings())
            scaled_vec = vec * self.gl_widget.global_distance_scale * settings.distance_scale

        if self.gl_widget.align_to_ecliptic:
            rot = self.gl_widget.orbital_plane_rotation_matrix()
            scaled_vec = rot @ scaled_vec
        return scaled_vec

    def _current_planet_angle(self, name: str) -> Optional[float]:
        pos = self._scaled_planet_position(name)
        if pos is None:
            return None
        return math.atan2(float(pos[2]), float(pos[0]))

    def _sun_radius_mm(self) -> float:
        sun = BODY_BY_NAME.get("Sun")
        if not sun:
            return 0.0
        earth_radius_km = 6371.0
        earth_diameter_mm = self.earth_size_spin.value()
        log_base = self.log_base_spin.value()
        ratio = sun.radius_km / earth_radius_km if earth_radius_km > 0 else 0.0
        if ratio <= 0:
            return 0.0
        diameter = earth_diameter_mm * (1.0 + math.log(ratio, log_base))
        diameter = max(diameter, self.gl_widget.min_body_diameter_mm)
        return diameter / 2.0

    def _rebuild_gear_layout(self) -> None:
        if not hasattr(self, "drive_speed_spin"):
            return

        gear_scale = float(self.gear_scale_spin.value())
        thickness = float(self.gear_thickness_spin.value())
        v_clear = float(self.vertical_clearance_spin.value())
        shaft_clear = float(self.shaft_clearance_spin.value())
        arm_lift = float(self.arm_clearance_spin.value())
        arm_v_clear = float(self.arm_vertical_clearance_spin.value())
        layer_height = max(thickness + v_clear, 0.1)

        def radius_for(teeth: int) -> float:
            return max(0.5, (float(teeth) * gear_scale) / (2.0 * math.pi))

        gears: Dict[str, GearRenderSpec] = {}
        speed_by_id: Dict[str, float] = {}
        shaft_by_id: Dict[str, int] = {}
        label_by_id: Dict[str, str] = {}
        train_outputs: Dict[str, str] = {}
        train_max_y: Dict[str, float] = {}
        mesh_links: List[Tuple[np.ndarray, np.ndarray]] = []

        drive_speed = self._drive_speed_rps()
        drive_id = "drive"
        drive_radius = radius_for(int(self.drive_teeth_spin.value()))
        drive_center = (0.0, 0.0, 0.0)
        gears[drive_id] = GearRenderSpec(
            gear_id=drive_id,
            teeth=int(self.drive_teeth_spin.value()),
            radius_mm=drive_radius,
            thickness_mm=thickness,
            center_mm=drive_center,
            color=(0.85, 0.2, 0.2),
            label=f"Drive ({int(self.drive_teeth_spin.value())}T)",
            shaft_index=0,
            speed_rps=drive_speed,
            angle_rad=self._gear_angles.get(drive_id, 0.0),
            is_drive=True,
        )
        speed_by_id[drive_id] = drive_speed
        shaft_by_id[drive_id] = 0
        label_by_id[drive_id] = gears[drive_id].label

        for train in self._gear_trains:
            input_id = train.input_gear_id
            if input_id not in gears:
                continue
            input_spec = gears[input_id]
            input_speed = speed_by_id.get(input_id, 0.0)
            input_shaft = shaft_by_id.get(input_id, 0)

            max_pair_sum = 0.0
            for driver_teeth, driven_teeth in train.gear_pairs:
                max_pair_sum = max(
                    max_pair_sum,
                    radius_for(driver_teeth) + radius_for(driven_teeth),
                )
            shaft_distance = max_pair_sum + shaft_clear
            if shaft_distance <= 0:
                shaft_distance = shaft_clear

            if input_shaft == 0:
                shaft0_x = input_spec.center_mm[0]
                shaft1_x = shaft0_x + shaft_distance
            else:
                shaft1_x = input_spec.center_mm[0]
                shaft0_x = shaft1_x - shaft_distance

            speed_shaft: List[Optional[float]] = [None, None]
            speed_shaft[input_shaft] = input_speed

            driver_shaft = input_shaft
            last_driven_id = None
            max_y = input_spec.center_mm[1] + thickness * 0.5

            for stage_idx, (driver_teeth, driven_teeth) in enumerate(train.gear_pairs):
                driven_shaft = 1 - driver_shaft
                y = input_spec.center_mm[1] + (stage_idx + 1) * layer_height
                driver_speed = speed_shaft[driver_shaft] or 0.0
                ratio = float(driver_teeth) / float(driven_teeth)
                driven_speed = -driver_speed * ratio
                speed_shaft[driven_shaft] = driven_speed

                driver_id = f"{train.train_id}_S{stage_idx + 1}_driver"
                driven_id = f"{train.train_id}_S{stage_idx + 1}_driven"
                driver_x = shaft0_x if driver_shaft == 0 else shaft1_x
                driven_x = shaft0_x if driven_shaft == 0 else shaft1_x

                gears[driver_id] = GearRenderSpec(
                    gear_id=driver_id,
                    teeth=driver_teeth,
                    radius_mm=radius_for(driver_teeth),
                    thickness_mm=thickness,
                    center_mm=(driver_x, y, 0.0),
                    color=(0.75, 0.75, 0.78),
                    label=f"{train.train_id} S{stage_idx + 1} Driver ({driver_teeth}T)",
                    shaft_index=driver_shaft,
                    speed_rps=driver_speed,
                    angle_rad=self._gear_angles.get(driver_id, 0.0),
                )
                gears[driven_id] = GearRenderSpec(
                    gear_id=driven_id,
                    teeth=driven_teeth,
                    radius_mm=radius_for(driven_teeth),
                    thickness_mm=thickness,
                    center_mm=(driven_x, y, 0.0),
                    color=(0.75, 0.75, 0.78),
                    label=f"{train.train_id} S{stage_idx + 1} Driven ({driven_teeth}T)",
                    shaft_index=driven_shaft,
                    speed_rps=driven_speed,
                    angle_rad=self._gear_angles.get(driven_id, 0.0),
                )
                speed_by_id[driver_id] = driver_speed
                speed_by_id[driven_id] = driven_speed
                shaft_by_id[driver_id] = driver_shaft
                shaft_by_id[driven_id] = driven_shaft
                label_by_id[driver_id] = gears[driver_id].label
                label_by_id[driven_id] = gears[driven_id].label
                last_driven_id = driven_id
                driver_shaft = driven_shaft
                max_y = max(max_y, y + thickness * 0.5)
                mesh_links.append(
                    (
                        np.array([driver_x, y, 0.0], dtype=float),
                        np.array([driven_x, y, 0.0], dtype=float),
                    )
                )

            if last_driven_id:
                train_outputs[train.train_id] = last_driven_id
                train_max_y[train.train_id] = max_y

        self._gear_speed_by_id = speed_by_id
        self._gear_shaft_by_id = shaft_by_id
        self._gear_label_by_id = label_by_id
        for gear_id in list(self._gear_angles.keys()):
            if gear_id not in gears:
                self._gear_angles.pop(gear_id, None)

        if self._selected_input_gear_id not in self._gear_label_by_id:
            self._selected_input_gear_id = drive_id

        arms: Dict[str, ArmRenderSpec] = {}
        global_max_y = 0.0
        for gear in gears.values():
            global_max_y = max(global_max_y, gear.center_mm[1] + thickness * 0.5)
        for idx, train in enumerate(self._gear_trains):
            output_id = train_outputs.get(train.train_id)
            if not output_id:
                continue
            length = self._scaled_planet_mean_distance(train.target_body)
            if length is None:
                continue
            arm_id = f"arm_{train.train_id}"
            angle = self._arm_angles.get(arm_id)
            if angle is None:
                angle = self._current_planet_angle(train.target_body) or 0.0
            output_speed = speed_by_id.get(output_id, 0.0)
            body = BODY_BY_NAME.get(train.target_body)
            color = body.color if body else (0.8, 0.8, 0.8)
            output_gear = gears[output_id]
            base_height = output_gear.center_mm[1] + thickness * 0.5
            arm_height = global_max_y + arm_lift + idx * arm_v_clear
            arms[arm_id] = ArmRenderSpec(
                arm_id=arm_id,
                length_mm=length,
                angle_rad=angle,
                origin_mm=(output_gear.center_mm[0], arm_height, output_gear.center_mm[2]),
                tube_base_y_mm=base_height,
                speed_rps=output_speed,
                color=color,
                label=train.target_body,
            )

        sun_radius_mm = self._sun_radius_mm()
        if sun_radius_mm > 0.0:
            max_entity_y = 0.0
            for gear in gears.values():
                max_entity_y = max(max_entity_y, gear.center_mm[1] + thickness * 0.5)
            for arm in arms.values():
                max_entity_y = max(max_entity_y, arm.origin_mm[1])
            if max_entity_y > -sun_radius_mm:
                shift = max_entity_y + sun_radius_mm
                for gear in gears.values():
                    x, y, z = gear.center_mm
                    gear.center_mm = (x, y - shift, z)
                for arm in arms.values():
                    ox, oy, oz = arm.origin_mm
                    arm.origin_mm = (ox, oy - shift, oz)
                    arm.tube_base_y_mm -= shift
                mesh_links = [
                    (
                        np.array([a[0], a[1] - shift, a[2]], dtype=float),
                        np.array([b[0], b[1] - shift, b[2]], dtype=float),
                    )
                    for a, b in mesh_links
                ]

        for arm_id in list(self._arm_angles.keys()):
            if arm_id not in arms:
                self._arm_angles.pop(arm_id, None)
        self._gear_arms = arms
        self.gl_widget.set_gears(gears)
        self.gl_widget.set_arms(arms)
        self.gl_widget.set_mesh_links(tuple(mesh_links))
        self._update_gear_list()
        self._set_selected_gear(self._selected_input_gear_id)

    def _advance_gear_animation(self, delta: float) -> None:
        if not self._gear_arms:
            return
        time_scale = max(self._time_speed, 0.0)
        for gear_id, speed in self._gear_speed_by_id.items():
            angle = self._gear_angles.get(gear_id, 0.0)
            angle = (angle + float(delta) * speed * time_scale * 2.0 * math.pi) % (2.0 * math.pi)
            self._gear_angles[gear_id] = angle
        for arm_id, arm in self._gear_arms.items():
            angle = self._arm_angles.get(arm_id, arm.angle_rad)
            angle = (angle + float(delta) * arm.speed_rps * time_scale * 2.0 * math.pi) % (2.0 * math.pi)
            self._arm_angles[arm_id] = angle
            arm.angle_rad = angle
        self.gl_widget.set_gear_angles(self._gear_angles)
        self.gl_widget.update()

    def _show_loading_overlay(self, visible: bool) -> None:
        self._loading_overlay.setVisible(visible)
        if visible:
            self._loading_overlay.raise_()
            self._loading_overlay.setGeometry(self.centralWidget().rect())

    def _set_loading_step(self, text: str) -> None:
        if hasattr(self, "_loading_step_label"):
            self._loading_step_label.setText(text)
            QApplication.processEvents()

    def _mark_textures_loaded(self) -> None:
        self._textures_loaded = True
        self._finalize_loading_if_ready()

    def _finalize_loading_if_ready(self) -> None:
        if self._kernels_loaded and self._textures_loaded:
            self._show_loading_overlay(False)

    def _browse_kernel_dir(self) -> None:
        start_dir = self.kernel_path.text().strip() or str(self.spice.kernel_dir)
        selected = QFileDialog.getExistingDirectory(self, "Select kernel folder", start_dir)
        if selected:
            self.kernel_path.setText(selected)

    def _load_kernels(self) -> None:
        kernel_dir = Path(self.kernel_path.text().strip())
        result = self.spice.load_kernels(kernel_dir)
        self.kernel_status.setText(f"Status: {self.spice.status_message}")
        if result.errors:
            msg = "Some kernels failed to load:\n" + "\n".join(result.errors[:6])
            if len(result.errors) > 6:
                msg += f"\n...and {len(result.errors) - 6} more"
            QMessageBox.warning(self, "Kernel load issues", msg)
        self._refresh_positions()
        self._update_diagnostics()

    def _set_now(self) -> None:
        self.time_edit.setDateTime(QDateTime.currentDateTimeUtc())
        self._refresh_positions()

    def _refresh_positions(self, *, refresh_orbit_stats: bool = True, update_diagnostics: bool = True) -> None:
        when = self._current_datetime()
        self._positions = self.spice.get_positions(when)
        self._rotations = self.spice.get_body_rotations(when)
        self._spin_dirs = self.spice.get_body_spin_directions(when)
        plane_normal = self.spice.get_orbital_plane_normal(when)
        self._solar_plane_normal = plane_normal
        if self.mechanical_check.isChecked():
            stats_key = (self.sample_spin.value(), self.span_orbits_spin.value())
            if refresh_orbit_stats and stats_key != self._orbit_stats_key:
                self._orbit_stats = self.spice.get_orbit_stats(
                    when,
                    BODY_CATALOG,
                    samples=self.sample_spin.value(),
                    span_orbits=self.span_orbits_spin.value(),
                    plane_normal=plane_normal,
                )
                self._orbit_stats_key = stats_key
            scaled_positions, system_bounds, orbit_sweeps, moon_bands = self._mechanical_layout(self._positions)
            self._moon_bands_by_parent = moon_bands
            self.gl_widget.update_positions(scaled_positions, already_scaled=True)
            self.gl_widget.set_body_rotations(self._rotations)
            self.gl_widget.set_body_spin_directions(self._spin_dirs)
            self.gl_widget.set_system_bounds(system_bounds, orbit_sweeps)
        else:
            self.gl_widget.update_positions(self._positions, already_scaled=False)
            self.gl_widget.set_body_rotations(self._rotations)
            self.gl_widget.set_body_spin_directions(self._spin_dirs)
            self.gl_widget.set_system_bounds({}, {})
            self._moon_bands_by_parent = {}
        missing = sum(1 for state in self._positions.values() if not state.available)
        self.missing_label.setText(f"Missing bodies: {missing}")
        if update_diagnostics:
            self._update_diagnostics()
        self._update_orbit_paths(when, refresh_orbit_stats)
        self._apply_focus_system(self._focused_system, update_camera=False)
        if self._focused_body:
            self.gl_widget.focus_on_body(self._focused_body)

    def _current_datetime(self) -> datetime:
        dt = self.time_edit.dateTime().toUTC()
        value = dt.toPython()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value
        return datetime.fromisoformat(dt.toString(Qt.DateFormat.ISODate)).replace(tzinfo=timezone.utc)

    def _focus_selected_body(self, _item=None) -> None:
        item = self.body_list.currentItem()
        if not item:
            return
        name = item.text()
        self._focused_body = name
        self._apply_focus_system(name, update_camera=False)
        self.gl_widget.focus_on_body(name)
        if self.orbit_paths_focus_check.isChecked() and self.orbit_paths_check.isChecked():
            self._update_orbit_paths(self._current_datetime(), refresh_orbit_stats=False)

    def _focus_body_from_view(self, name: str) -> None:
        self._focused_body = name
        self._apply_focus_system(name, update_camera=False)
        self.gl_widget.focus_on_body(name)
        if self.orbit_paths_focus_check.isChecked() and self.orbit_paths_check.isChecked():
            self._update_orbit_paths(self._current_datetime(), refresh_orbit_stats=False)

    def _apply_mechanical_settings(self, _value: Optional[int] = None) -> None:
        enabled = self.mechanical_check.isChecked()
        self.gl_widget.distance_mode = "mechanical" if enabled else "astronomical"
        self.gl_widget.size_mode = "log"
        self.gl_widget.earth_diameter_mm = self.earth_size_spin.value()
        self.gl_widget.size_log_base = self.log_base_spin.value()
        self.gl_widget.show_orbit_bands = self.bands_check.isChecked()
        self.gl_widget.auto_distance_scale = False
        self.gl_widget.min_visible_radius = 0.2
        self.gl_widget.highlight_threshold_earth_units = 0.5
        self.gl_widget.real_scale_mode = False

        self.earth_size_spin.setEnabled(enabled)
        self.log_base_spin.setEnabled(enabled)
        self.planet_clearance_spin.setEnabled(enabled)
        self.moon_clearance_spin.setEnabled(enabled)
        self.sample_spin.setEnabled(enabled)
        self.span_orbits_spin.setEnabled(enabled)
        self.bands_check.setEnabled(enabled)
        self.orbit_paths_check.setEnabled(True)
        self.real_scale_btn.setEnabled(True)
        show_bands = self.bands_check.isChecked()
        self.band_opacity_label.setVisible(show_bands)
        self.band_opacity_slider.setVisible(show_bands)
        self.band_opacity_slider.setEnabled(show_bands)
        self._apply_band_opacity(self.band_opacity_slider.value())

        self._refresh_positions()
        self._rebuild_gear_layout()

    def _set_time_speed(self, speed: float) -> None:
        clamped = max(0.0, float(speed))
        if clamped <= 0.0:
            self._pause_time()
            return

        self._time_speed = clamped
        self._last_nonzero_speed = clamped
        self._time_last_tick = perf_counter()
        if not self._time_timer.isActive():
            self._time_timer.start()
        self.time_overlay.set_running(True)
        self.time_overlay.set_speed(clamped)

    def _toggle_time_playback(self, running: bool) -> None:
        if running:
            speed = self._last_nonzero_speed if self._last_nonzero_speed > 0 else 86400.0
            self._set_time_speed(speed)
        else:
            self._pause_time()

    def _pause_time(self) -> None:
        self._time_speed = 0.0
        self._time_last_tick = None
        if self._time_timer.isActive():
            self._time_timer.stop()
        self.time_overlay.set_running(False)
        self.time_overlay.set_speed(0.0)

    def _advance_time(self) -> None:
        if self._time_speed <= 0.0:
            return
        now = perf_counter()
        if self._time_last_tick is None:
            self._time_last_tick = now
            return
        delta = now - self._time_last_tick
        self._time_last_tick = now
        if delta <= 0:
            return
        self._advance_gear_animation(delta)
        current = self._current_datetime()
        next_time = current + timedelta(seconds=delta * self._time_speed)
        ms = int(next_time.timestamp() * 1000)
        self.time_edit.setDateTime(QDateTime.fromMSecsSinceEpoch(ms, Qt.TimeSpec.UTC))
        self._refresh_positions(refresh_orbit_stats=False, update_diagnostics=False)

    def _apply_real_scale(self) -> None:
        self.mechanical_check.setChecked(False)
        self.gl_widget.size_mode = "linear"
        self.gl_widget.min_visible_radius = 0.0
        self.gl_widget.highlight_threshold_earth_units = 0.0
        self.gl_widget.real_scale_mode = True

        for settings in self._settings.values():
            settings.distance_scale = 1.0
            settings.radius_scale = 1.0

        positions = self._positions
        max_dist = 0.0
        for state in positions.values():
            if not state.available:
                continue
            vec = np.array(state.position_km, dtype=float)
            max_dist = max(max_dist, float(np.linalg.norm(vec)))
        if max_dist <= 0:
            return

        scale = self.gl_widget.target_scene_radius / max_dist
        self.gl_widget.global_distance_scale = scale
        self.gl_widget.global_radius_scale = scale
        self.gl_widget.update()

    def _apply_band_opacity(self, value: int) -> None:
        clamped = max(0, min(100, int(value)))
        self.gl_widget.orbit_band_opacity = clamped / 100.0
        self.band_opacity_label.setText(f"Orbit band opacity ({clamped}%)")
        self.gl_widget.update()

    def _toggle_orbit_paths(self, checked: bool) -> None:
        self.gl_widget.show_orbit_paths = checked
        if checked:
            self._refresh_positions()
        else:
            self._orbit_paths = {}
            self._orbit_paths_key = None
            self._orbit_path_centers = {}
            self.gl_widget.set_orbit_paths({})

    def _toggle_orbit_paths_focus(self, _checked: bool) -> None:
        if self.orbit_paths_check.isChecked():
            self._update_orbit_paths(self._current_datetime(), refresh_orbit_stats=False)

    def _update_orbit_paths(self, when: datetime, refresh_orbit_stats: bool) -> None:
        if not self.orbit_paths_check.isChecked():
            return
        if not self.spice.loaded:
            self._orbit_paths = {}
            self._orbit_paths_key = None
            self._orbit_path_centers = {}
            self.gl_widget.set_orbit_paths({})
            return

        orbits = max(1, int(self.span_orbits_spin.value()))
        samples_per_orbit = max(2, int(self.sample_spin.value()))
        focus_only = self.orbit_paths_focus_check.isChecked()
        focus_name = self._focused_body if focus_only else None
        key_time = when.replace(hour=0, minute=0, second=0, microsecond=0)
        key = (
            samples_per_orbit,
            orbits,
            key_time,
            bool(self.mechanical_check.isChecked()),
            focus_name,
            float(self.earth_size_spin.value()),
            float(self.log_base_spin.value()),
            float(self.planet_clearance_spin.value()),
            float(self.moon_clearance_spin.value()),
        )
        if not refresh_orbit_stats and self._orbit_paths_key == key and self._orbit_paths:
            self.gl_widget.set_orbit_paths(
                self._orbit_paths,
                centers=self._orbit_path_centers,
                already_scaled=self.mechanical_check.isChecked(),
            )
            return

        bodies = self._orbit_path_bodies(focus_only)
        if not bodies:
            self._orbit_paths = {}
            self._orbit_paths_key = key
            self._orbit_path_centers = {}
            self.gl_widget.set_orbit_paths({})
            return

        paths = self.spice.get_orbit_paths(
            when,
            bodies,
            samples_per_orbit=samples_per_orbit,
            span_orbits=orbits,
            frame="parent",
        )

        smoothed: Dict[str, np.ndarray] = {}
        centers: Dict[str, Optional[str]] = {}
        for body in bodies:
            points = paths.get(body.name)
            if points is None or points.size == 0:
                continue
            if self.mechanical_check.isChecked():
                if body.parent.upper() == "SUN":
                    points = self._scale_planet_path_mechanical(body.name, points)
                else:
                    points = self._scale_moon_path_mechanical(body.name, points)
            smoothed[body.name] = self._smooth_path(points, samples_per_segment=2)
            centers[body.name] = body.parent.title() if body.parent.upper() != "SUN" else None

        self._orbit_paths = smoothed
        self._orbit_paths_key = key
        self._orbit_path_centers = centers
        self.gl_widget.set_orbit_paths(
            self._orbit_paths,
            centers=self._orbit_path_centers,
            already_scaled=self.mechanical_check.isChecked(),
        )

    @staticmethod
    def _smooth_path(points: np.ndarray, samples_per_segment: int = 2) -> np.ndarray:
        if points.shape[0] < 4 or samples_per_segment <= 1:
            return points
        pts = np.asarray(points, dtype=float)
        extended = np.vstack([pts[0], pts, pts[-1]])
        curve = []
        for i in range(1, len(extended) - 2):
            p0, p1, p2, p3 = extended[i - 1], extended[i], extended[i + 1], extended[i + 2]
            for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                point = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                curve.append(point)
        curve.append(pts[-1])
        return np.asarray(curve, dtype=float)

    def _scale_planet_path_mechanical(self, name: str, points: np.ndarray) -> np.ndarray:
        stats = self._orbit_stats.get(name)
        bounds = self._planet_bounds.get(name)
        if stats is None or bounds is None:
            return points
        min_km = stats.min_km
        max_km = stats.max_km
        if max_km <= min_km:
            return points
        scaled_min, scaled_max = bounds
        scaled = np.zeros_like(points, dtype=float)
        for idx, vec in enumerate(points):
            r_actual = float(np.linalg.norm(vec))
            if r_actual > 0:
                direction = vec / r_actual
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=float)
            t = (r_actual - min_km) / (max_km - min_km)
            t = max(0.0, min(1.0, t))
            scaled_r = scaled_min + t * (scaled_max - scaled_min)
            scaled[idx] = direction * scaled_r
        return scaled

    def _scale_moon_path_mechanical(self, name: str, points: np.ndarray) -> np.ndarray:
        stats = self._orbit_stats.get(name)
        bounds = self._moon_scaled_bounds.get(name)
        if stats is None or bounds is None:
            return points
        min_km = stats.min_km
        max_km = stats.max_km
        if max_km <= min_km:
            return points
        scaled_min, scaled_max = bounds
        scaled = np.zeros_like(points, dtype=float)
        for idx, vec in enumerate(points):
            r_actual = float(np.linalg.norm(vec))
            if r_actual > 0:
                direction = vec / r_actual
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=float)
            t = (r_actual - min_km) / (max_km - min_km)
            t = max(0.0, min(1.0, t))
            scaled_r = scaled_min + t * (scaled_max - scaled_min)
            scaled[idx] = direction * scaled_r
        return scaled

    def _orbit_path_bodies(self, focus_only: bool) -> list:
        if focus_only:
            focus_name = self._focused_body
            if not focus_name:
                return []
            info = BODY_BY_NAME.get(focus_name)
            if not info or info.spice_name.upper() == "SUN":
                return []
            return [info]
        return [
            body
            for body in BODY_CATALOG
            if body.parent.upper() == "SUN" and body.name != "Sun"
        ]

    def _apply_focus_system(self, name: Optional[str], update_camera: bool = True) -> None:
        if not name:
            self._focused_system = None
        else:
            self._focused_system = name

        if not self.mechanical_check.isChecked():
            self.gl_widget.show_moon_bands = False
            self.gl_widget.set_moon_bands(None, {})
            self.gl_widget.show_orbit_bands = self.bands_check.isChecked()
            self.gl_widget.set_orbital_plane_normal(self._solar_plane_normal)
            if update_camera:
                self.gl_widget.update()
            return

        moons = self._moon_bands_by_parent.get(name or "", ())
        if moons:
            # Focused on a planet system with moons
            self.gl_widget.show_orbit_bands = False
            self.gl_widget.show_moon_bands = True
            self.gl_widget.set_moon_bands(name, self._moon_bands_by_parent)
            system_normal = self._system_plane_normal(name) if name else None
            self.gl_widget.set_orbital_plane_normal(system_normal or self._solar_plane_normal)
        else:
            # Default solar system view
            self.gl_widget.show_orbit_bands = self.bands_check.isChecked()
            self.gl_widget.show_moon_bands = False
            self.gl_widget.set_moon_bands(None, {})
            self.gl_widget.set_orbital_plane_normal(self._solar_plane_normal)

        if update_camera:
            self.gl_widget.update()

    def _system_plane_normal(self, name: str) -> Optional[Tuple[float, float, float]]:
        info = BODY_BY_NAME.get(name)
        if not info:
            return None
        when = self._current_datetime()

        # If system has moons, use the innermost moon's orbital plane as the system plane.
        moons = [body for body in BODY_CATALOG if body.parent.title() == name]
        if moons:
            moons.sort(key=lambda b: b.mean_distance_km or float("inf"))
            moon = moons[0]
            return self.spice.get_orbital_plane_normal(when, moon.spice_name, info.spice_name)

        # Otherwise use this body's own orbital plane around its parent.
        parent = info.parent or "SUN"
        return self.spice.get_orbital_plane_normal(when, info.spice_name, parent)

    def _mechanical_layout(
        self,
        positions: Dict[str, "BodyState"],
    ) -> Tuple[
        Dict[str, "BodyState"],
        Dict[str, float],
        Dict[str, Tuple[float, float, float]],
        Dict[str, Tuple[Tuple[float, float], ...]],
    ]:
        earth_radius_km = 6371.0
        earth_diameter_mm = self.earth_size_spin.value()
        log_base = self.log_base_spin.value()
        planet_clearance = self.planet_clearance_spin.value()
        moon_clearance = self.moon_clearance_spin.value()
        def size_radius_mm(radius_km: float, scale: float) -> float:
            ratio = radius_km / earth_radius_km
            if ratio <= 0:
                return 0.5
            diameter = earth_diameter_mm * (1.0 + math.log(ratio, log_base))
            diameter = max(diameter, self.gl_widget.min_body_diameter_mm)
            return (diameter / 2.0) * scale

        body_radius_mm: Dict[str, float] = {}
        for body in BODY_CATALOG:
            settings = self._settings.get(body.name, BodySettings())
            body_radius_mm[body.name] = size_radius_mm(body.radius_km, settings.radius_scale)

        moons_by_parent: Dict[str, list] = {}
        for body in BODY_CATALOG:
            if body.parent and body.parent.upper() != "SUN":
                parent_name = body.parent.title()
                moons_by_parent.setdefault(parent_name, []).append(body)

        moon_scaled_bounds: Dict[str, Tuple[float, float]] = {}
        moon_bands_by_parent: Dict[str, list] = {}
        moon_scale_by_parent: Dict[str, float] = {}
        system_radius: Dict[str, float] = {}

        for parent, moons in moons_by_parent.items():
            parent_radius = body_radius_mm.get(parent, body_radius_mm.get(parent.title(), 1.0))
            entries = []
            min_km = None
            for moon in moons:
                stats = self._orbit_stats.get(moon.name)
                if stats:
                    moon_min = stats.min_km
                    moon_max = stats.max_km
                else:
                    current = positions.get(moon.name)
                    if current:
                        moon_min = moon_max = float(np.linalg.norm(np.array(current.position_km, dtype=float)))
                    else:
                        moon_min = moon_max = moon.mean_distance_km or 0.0
                if min_km is None or moon_min < min_km:
                    min_km = moon_min
                entries.append((moon, moon_min, moon_max))

            if not entries:
                continue

            moon_scale = 1.0
            if min_km and min_km > 0:
                moon_scale = (parent_radius + moon_clearance) / min_km
            moon_scale_by_parent[parent] = moon_scale

            entries.sort(key=lambda item: item[1])
            prev_outer = parent_radius
            for moon, moon_min, moon_max in entries:
                raw_min = moon_min * moon_scale
                raw_max = moon_max * moon_scale
                moon_radius = body_radius_mm.get(moon.name, 0.5)
                required_min = prev_outer + moon_clearance + moon_radius
                scaled_min = max(required_min, parent_radius + moon_clearance + moon_radius)
                shift = scaled_min - raw_min
                scaled_max = raw_max + shift
                outer_surface = scaled_max + moon_radius
                inner_surface = max(scaled_min - moon_radius, 0.0)
                moon_scaled_bounds[moon.name] = (scaled_min, scaled_max)
                moon_bands_by_parent.setdefault(parent, []).append((inner_surface, outer_surface))
                prev_outer = outer_surface
            system_radius[parent] = max(prev_outer, parent_radius)

        planets = [body for body in BODY_CATALOG if body.parent.upper() == "SUN" and body.name != "Sun"]
        planet_bounds: Dict[str, Tuple[float, float]] = {}
        prev_max_center = 0.0
        prev_system_radius = body_radius_mm.get("Sun", 0.0)

        min_planet_km = None
        for body in planets:
            stats = self._orbit_stats.get(body.name)
            if stats:
                candidate = stats.min_km
            else:
                candidate = body.mean_distance_km or None
            if candidate is None:
                continue
            if min_planet_km is None or candidate < min_planet_km:
                min_planet_km = candidate

        planet_scale = 1.0
        if min_planet_km and min_planet_km > 0:
            inner = sorted(planets, key=lambda b: b.mean_distance_km or float("inf"))[0]
            inner_radius = system_radius.get(inner.name, body_radius_mm.get(inner.name, 1.0))
            sun_radius = body_radius_mm.get("Sun", 0.0)
            target_min = sun_radius + inner_radius + planet_clearance
            planet_scale = target_min / min_planet_km

        for body in sorted(planets, key=lambda b: b.mean_distance_km or float("inf")):
            stats = self._orbit_stats.get(body.name)
            if stats:
                min_km = stats.min_km
                max_km = stats.max_km
            else:
                current = positions.get(body.name)
                if current:
                    min_km = max_km = float(np.linalg.norm(np.array(current.position_km, dtype=float)))
                else:
                    min_km = max_km = body.mean_distance_km or 0.0

            raw_min = min_km * planet_scale
            raw_max = max_km * planet_scale
            sys_radius = system_radius.get(body.name, body_radius_mm.get(body.name, 1.0))
            required_min = prev_max_center + prev_system_radius + sys_radius + planet_clearance
            scaled_min = max(required_min, 0.0)
            shift = scaled_min - raw_min
            scaled_max = raw_max + shift
            planet_bounds[body.name] = (scaled_min, scaled_max)

            prev_max_center = scaled_max
            prev_system_radius = sys_radius

        scaled_positions: Dict[str, "BodyState"] = {}

        sun_state = positions.get("Sun")
        if sun_state:
            scaled_positions["Sun"] = BodyState((0.0, 0.0, 0.0), sun_state.available, sun_state.error)

        for body in planets:
            state = positions.get(body.name)
            if not state:
                continue
            vec = np.array(state.position_km, dtype=float)
            r_actual = float(np.linalg.norm(vec))
            stats = self._orbit_stats.get(body.name)
            if stats and stats.max_km > stats.min_km:
                t = (r_actual - stats.min_km) / (stats.max_km - stats.min_km)
                t = max(0.0, min(1.0, t))
            else:
                t = 0.5
            scaled_min, scaled_max = planet_bounds.get(body.name, (r_actual * planet_scale, r_actual * planet_scale))
            scaled_r = scaled_min + t * (scaled_max - scaled_min)
            if r_actual > 0:
                direction = vec / r_actual
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=float)
            scaled_vec = direction * scaled_r
            scaled_positions[body.name] = BodyState(
                (float(scaled_vec[0]), float(scaled_vec[1]), float(scaled_vec[2])),
                state.available,
                state.error,
            )

        for parent, moons in moons_by_parent.items():
            parent_scaled = scaled_positions.get(parent)
            parent_actual = positions.get(parent)
            if not parent_scaled or not parent_actual:
                continue
            parent_scaled_vec = np.array(parent_scaled.position_km, dtype=float)
            parent_actual_vec = np.array(parent_actual.position_km, dtype=float)
            for moon in moons:
                state = positions.get(moon.name)
                if not state:
                    continue
                moon_vec = np.array(state.position_km, dtype=float)
                rel_vec = moon_vec - parent_actual_vec
                r_actual = float(np.linalg.norm(rel_vec))
                stats = self._orbit_stats.get(moon.name)
                if stats and stats.max_km > stats.min_km:
                    t = (r_actual - stats.min_km) / (stats.max_km - stats.min_km)
                    t = max(0.0, min(1.0, t))
                else:
                    t = 0.5
                moon_scale = moon_scale_by_parent.get(parent, 1.0)
                scaled_min, scaled_max = moon_scaled_bounds.get(
                    moon.name,
                    (r_actual * moon_scale, r_actual * moon_scale),
                )
                scaled_r = scaled_min + t * (scaled_max - scaled_min)
                if r_actual > 0:
                    direction = rel_vec / r_actual
                else:
                    direction = np.array([1.0, 0.0, 0.0], dtype=float)
                scaled_rel = direction * scaled_r
                scaled_pos = parent_scaled_vec + scaled_rel
                scaled_positions[moon.name] = BodyState(
                    (float(scaled_pos[0]), float(scaled_pos[1]), float(scaled_pos[2])),
                    state.available,
                    state.error,
                )

        for body in BODY_CATALOG:
            if body.name in scaled_positions:
                continue
            state = positions.get(body.name)
            if not state:
                continue
            scaled_positions[body.name] = BodyState(state.position_km, state.available, state.error)

        system_bounds_mm: Dict[str, float] = {}
        orbit_sweeps_mm: Dict[str, Tuple[float, float, float]] = {}
        for body in planets:
            sys_radius = system_radius.get(body.name, body_radius_mm.get(body.name, 1.0))
            system_bounds_mm[body.name] = sys_radius
            bounds = planet_bounds.get(body.name)
            if bounds:
                inner = max(bounds[0] - sys_radius, 0.0)
                outer = bounds[1] + sys_radius
                thickness = 0.0
                stats = self._orbit_stats.get(body.name)
                if stats and stats.max_normal_km > 0:
                    thickness = stats.max_normal_km * planet_scale
                orbit_sweeps_mm[body.name] = (inner, outer, thickness)

        moon_bands_final: Dict[str, Tuple[Tuple[float, float], ...]] = {}
        for parent, bands in moon_bands_by_parent.items():
            moon_bands_final[parent] = tuple(bands)

        self._planet_bounds = dict(planet_bounds)
        self._planet_scale = float(planet_scale)
        self._moon_scaled_bounds = dict(moon_scaled_bounds)
        self._moon_scale_by_parent = dict(moon_scale_by_parent)

        return scaled_positions, system_bounds_mm, orbit_sweeps_mm, moon_bands_final

    def _update_diagnostics(self) -> None:
        lines = []
        lines.append(f"SPICE: {self.spice.status_message}")
        lines.append(f"Kernel dir: {self.kernel_path.text().strip()}")

        if self.spice.load_errors:
            lines.append("Kernel load errors:")
            for err in self.spice.load_errors[:6]:
                lines.append(f"  - {err}")
            if len(self.spice.load_errors) > 6:
                lines.append(f"  ...and {len(self.spice.load_errors) - 6} more")

        missing_names = [name for name, state in self._positions.items() if not state.available]
        if missing_names:
            lines.append(f"Missing bodies ({len(missing_names)}): {', '.join(missing_names[:12])}")
            if len(missing_names) > 12:
                lines.append(f"...and {len(missing_names) - 12} more")
            error_counts = {}
            for name in missing_names:
                state = self._positions.get(name)
                err = state.error if state else None
                if not err:
                    continue
                error_counts[err] = error_counts.get(err, 0) + 1
            if error_counts:
                lines.append("Sample SPICE errors:")
                for err, count in list(error_counts.items())[:6]:
                    lines.append(f"  - ({count}x) {err}")
        else:
            lines.append("Missing bodies: 0")

        try:
            kernel_dir = Path(self.kernel_path.text().strip())
            tmp_files = list(kernel_dir.rglob("BIT*.tmp"))
            if tmp_files:
                lines.append("Warning: kernel downloads still in progress (.tmp files present).")
        except Exception:
            pass

        self.diagnostics.setPlainText("\n".join(lines))


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
