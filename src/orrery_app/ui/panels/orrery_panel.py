from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QDateTime, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QDateTimeEdit,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...models import BODY_CATALOG

if TYPE_CHECKING:
    _MainWindowBase = Any
else:
    _MainWindowBase = object


class OrreryPanelMixin(_MainWindowBase):
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

        title = QLabel("Loading Orrery Crafting...", overlay)
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._loading_step_label = QLabel("Initializing...", overlay)
        self._loading_step_label.setFont(QFont("Segoe UI", 10))
        self._loading_step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._loading_step_label)

        tips = QLabel(
            "What is loading:\n"
            "- SPICE kernels and ephemerides\n"
            "- Orbit stats and mechanical layout\n"
            "- Textures and render setup",
            overlay,
        )
        tips.setFont(QFont("Segoe UI", 9))
        tips.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tips)

        return overlay
