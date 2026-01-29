from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from Gear_calc import compute_gear_train

from ...gear_models import ArmRenderSpec, GearRenderSpec
from ...models import BODY_BY_NAME, BODY_CATALOG, BodySettings

if TYPE_CHECKING:
    _MainWindowBase = Any
else:
    _MainWindowBase = object


@dataclass
class GearTrainEntry:
    train_id: str
    target_body: str
    gear_pairs: list[tuple[int, int]]
    input_gear_id: str
    ratio_target: float
    ratio_achieved: float
    gear_string: str


class GearPanelMixin(_MainWindowBase):
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
        self.drive_teeth_spin.setEnabled(False)
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

        target_group = QGroupBox("Target Body", content)
        target_layout = QVBoxLayout(target_group)
        self.target_body_combo = QComboBox(target_group)
        for body in BODY_CATALOG:
            if body.parent.upper() == "SUN" and body.name != "Sun":
                self.target_body_combo.addItem(body.name)
        target_layout.addWidget(self.target_body_combo)
        layout.addWidget(target_group)

        search_group = QGroupBox("Gear Search", content)
        search_layout = QFormLayout(search_group)
        self.stages_spin = QSpinBox(search_group)
        self.stages_spin.setRange(1, 6)
        self.stages_spin.setValue(2)
        search_layout.addRow("Stages", self.stages_spin)
        self.min_teeth_spin = QSpinBox(search_group)
        self.min_teeth_spin.setRange(4, 60)
        self.min_teeth_spin.setValue(12)
        search_layout.addRow("Min teeth", self.min_teeth_spin)
        self.max_teeth_spin = QSpinBox(search_group)
        self.max_teeth_spin.setRange(6, 600)
        self.max_teeth_spin.setValue(80)
        search_layout.addRow("Max teeth", self.max_teeth_spin)
        self.tolerance_spin = QDoubleSpinBox(search_group)
        self.tolerance_spin.setDecimals(6)
        self.tolerance_spin.setRange(0.0, 0.05)
        self.tolerance_spin.setSingleStep(0.0001)
        self.tolerance_spin.setValue(0.001)
        search_layout.addRow("Tolerance", self.tolerance_spin)
        self.use_numba_check = QCheckBox("Use numba speedup", search_group)
        self.use_numba_check.setChecked(True)
        search_layout.addRow(self.use_numba_check)
        add_train_btn = QPushButton("Add Gear Train", search_group)
        add_train_btn.clicked.connect(self._add_gear_train)
        clear_trains_btn = QPushButton("Clear Trains", search_group)
        clear_trains_btn.clicked.connect(self._clear_gear_trains)
        search_layout.addRow(add_train_btn, clear_trains_btn)
        layout.addWidget(search_group)

        layout_group = QGroupBox("Layout", content)
        layout_form = QFormLayout(layout_group)
        self.gear_scale_spin = QDoubleSpinBox(layout_group)
        self.gear_scale_spin.setDecimals(3)
        self.gear_scale_spin.setRange(0.1, 100.0)
        self.gear_scale_spin.setSingleStep(0.1)
        self.gear_scale_spin.setValue(1.0)
        self.gear_scale_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Gear size multiplier", self.gear_scale_spin)

        self.gear_thickness_spin = QDoubleSpinBox(layout_group)
        self.gear_thickness_spin.setDecimals(2)
        self.gear_thickness_spin.setRange(0.2, 50.0)
        self.gear_thickness_spin.setSingleStep(0.1)
        self.gear_thickness_spin.setValue(2.5)
        self.gear_thickness_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Gear thickness (mm)", self.gear_thickness_spin)

        self.shaft_clearance_spin = QDoubleSpinBox(layout_group)
        self.shaft_clearance_spin.setDecimals(2)
        self.shaft_clearance_spin.setRange(0.0, 50.0)
        self.shaft_clearance_spin.setSingleStep(0.5)
        self.shaft_clearance_spin.setValue(10.0)
        self.shaft_clearance_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Shaft spacing (mm)", self.shaft_clearance_spin)

        self.gear_clearance_spin = QDoubleSpinBox(layout_group)
        self.gear_clearance_spin.setDecimals(2)
        self.gear_clearance_spin.setRange(0.0, 20.0)
        self.gear_clearance_spin.setSingleStep(0.1)
        self.gear_clearance_spin.setValue(0.4)
        self.gear_clearance_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Gear clearance (mm)", self.gear_clearance_spin)

        self.drive_offset_spin = QDoubleSpinBox(layout_group)
        self.drive_offset_spin.setDecimals(2)
        self.drive_offset_spin.setRange(-500.0, 500.0)
        self.drive_offset_spin.setSingleStep(1.0)
        self.drive_offset_spin.setValue(0.0)
        self.drive_offset_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Drive stack X offset (mm)", self.drive_offset_spin)

        self.vertical_clearance_spin = QDoubleSpinBox(layout_group)
        self.vertical_clearance_spin.setDecimals(2)
        self.vertical_clearance_spin.setRange(0.0, 50.0)
        self.vertical_clearance_spin.setSingleStep(0.5)
        self.vertical_clearance_spin.setValue(1.0)
        self.vertical_clearance_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Vertical clearance (mm)", self.vertical_clearance_spin)

        self.shaft_wall_spin = QDoubleSpinBox(layout_group)
        self.shaft_wall_spin.setDecimals(2)
        self.shaft_wall_spin.setRange(0.2, 50.0)
        self.shaft_wall_spin.setSingleStep(0.2)
        self.shaft_wall_spin.setValue(1.0)
        self.shaft_wall_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Min annulus wall (mm)", self.shaft_wall_spin)

        self.shaft_bore_clear_spin = QDoubleSpinBox(layout_group)
        self.shaft_bore_clear_spin.setDecimals(2)
        self.shaft_bore_clear_spin.setRange(0.0, 50.0)
        self.shaft_bore_clear_spin.setSingleStep(0.2)
        self.shaft_bore_clear_spin.setValue(0.4)
        self.shaft_bore_clear_spin.valueChanged.connect(self._rebuild_gear_layout)
        layout_form.addRow("Min shaft clearance (mm)", self.shaft_bore_clear_spin)

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

    def _drive_speed_rps(self) -> float:
        if not hasattr(self, "drive_speed_spin"):
            return 0.0
        speed = -float(self.drive_speed_spin.value()) / 86400.0
        if hasattr(self, "drive_reverse_check") and self.drive_reverse_check.isChecked():
            speed *= -1.0
        return speed

    def _parse_gear_string(self, gear_string: str) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
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
            item = QListWidgetItem(f"{label} - {speed:.4g} rev/day", self.gear_list)
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

    def _scaled_planet_mean_distance(self, name: str) -> float | None:
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

    def _scaled_planet_position(self, name: str) -> np.ndarray | None:
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

    def _current_planet_angle(self, name: str) -> float | None:
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
        gear_clear = float(self.gear_clearance_spin.value()) if hasattr(self, "gear_clearance_spin") else 0.0
        drive_offset_x = float(self.drive_offset_spin.value()) if hasattr(self, "drive_offset_spin") else 0.0
        min_wall = float(self.shaft_wall_spin.value()) if hasattr(self, "shaft_wall_spin") else 1.0
        min_bore_clear = (
            float(self.shaft_bore_clear_spin.value()) if hasattr(self, "shaft_bore_clear_spin") else 0.0
        )
        arm_lift = float(self.arm_clearance_spin.value())
        arm_v_clear = float(self.arm_vertical_clearance_spin.value())
        layer_height = max(thickness + v_clear, 0.1)

        def base_radius(teeth: int) -> float:
            return max(0.5, (float(teeth) * gear_scale) / (2.0 * math.pi))

        def pair_radii(driver_teeth: int, driven_teeth: int, shaft_distance: float) -> tuple[float, float, float]:
            total = float(driver_teeth + driven_teeth)
            if total <= 0 or shaft_distance <= 0:
                r_driver = base_radius(driver_teeth)
                r_driven = base_radius(driven_teeth)
                module = (r_driver + r_driven) * 2.0 / max(total, 1.0)
                return r_driver, r_driven, module
            module = (2.0 * shaft_distance) / total
            return (module * driver_teeth) / 2.0, (module * driven_teeth) / 2.0, module

        def overlaps_existing(x: float, y: float, z: float, radius: float, placed: list[GearRenderSpec]) -> bool:
            for other in placed:
                dx = x - other.center_mm[0]
                dz = z - other.center_mm[2]
                lateral_limit = radius + other.radius_mm + gear_clear
                if math.hypot(dx, dz) >= lateral_limit:
                    continue
                vertical_limit = (thickness + other.thickness_mm) * 0.5 + v_clear
                if abs(y - other.center_mm[1]) < vertical_limit:
                    return True
            return False

        def find_vertical_offset(
            base_y: float,
            base_z: float,
            shaft0_x: float,
            shaft1_x: float,
            gear_pairs: list[tuple[int, int]],
            placed: list[GearRenderSpec],
        ) -> tuple[float, int]:
            max_steps = 80
            offset = 0.0
            last_collisions = 0
            shaft_distance = abs(shaft1_x - shaft0_x)
            for _ in range(max_steps):
                collisions = 0
                driver_shaft = 0
                for stage_idx, (driver_teeth, driven_teeth) in enumerate(gear_pairs):
                    driven_shaft = 1 - driver_shaft
                    y = base_y + offset + (stage_idx + 1) * layer_height
                    driver_x = shaft0_x if driver_shaft == 0 else shaft1_x
                    driven_x = shaft0_x if driven_shaft == 0 else shaft1_x
                    r_driver, r_driven, _module = pair_radii(
                        driver_teeth,
                        driven_teeth,
                        shaft_distance,
                    )
                    if overlaps_existing(driver_x, y, base_z, r_driver, placed):
                        collisions += 1
                    if overlaps_existing(driven_x, y, base_z, r_driven, placed):
                        collisions += 1
                    if collisions:
                        break
                    driver_shaft = driven_shaft
                if collisions == 0:
                    return offset, 0
                last_collisions = collisions
                offset += layer_height
            return offset, last_collisions

        shaft_registry: dict[tuple[float, float], int] = {}

        def shaft_id_for(x: float, z: float) -> int:
            key = (round(x, 3), round(z, 3))
            if key not in shaft_registry:
                shaft_registry[key] = len(shaft_registry)
            return shaft_registry[key]

        def choose_layout(
            base_spec: GearRenderSpec,
            gear_pairs: list[tuple[int, int]],
            placed: list[GearRenderSpec],
        ) -> tuple[float, float, float, float]:
            max_pair_sum = 0.0
            for driver_teeth, driven_teeth in gear_pairs:
                max_pair_sum = max(
                    max_pair_sum,
                    base_radius(driver_teeth) + base_radius(driven_teeth),
                )
            shaft_distance = max(max_pair_sum + shaft_clear, shaft_clear, 0.1)
            loop_step = max(shaft_clear, max_pair_sum * 0.5, 1.0)

            input_x = base_spec.center_mm[0]
            input_y = base_spec.center_mm[1]
            input_z = base_spec.center_mm[2]

            candidates: list[tuple[int, float, float, float, float]] = []
            for loop_idx in range(3):
                distance = shaft_distance + loop_idx * loop_step
                for sign in (1.0, -1.0):
                    shaft0_x = input_x
                    shaft1_x = input_x + sign * distance
                    offset, collisions = find_vertical_offset(
                        input_y,
                        input_z,
                        shaft0_x,
                        shaft1_x,
                        gear_pairs,
                        placed,
                    )
                    candidates.append((collisions, offset, distance, shaft0_x, shaft1_x))

            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            _collisions, y_offset, used_distance, shaft0_x, shaft1_x = candidates[0]
            return y_offset, used_distance, shaft0_x, shaft1_x

        gears: dict[str, GearRenderSpec] = {}
        speed_by_id: dict[str, float] = {}
        shaft_by_id: dict[str, int] = {}
        label_by_id: dict[str, str] = {}
        train_outputs: dict[str, str] = {}
        train_max_y: dict[str, float] = {}
        mesh_links: list[tuple[np.ndarray, np.ndarray]] = []
        shaft_links: list[tuple[np.ndarray, np.ndarray]] = []

        drive_speed = self._drive_speed_rps()
        drive_id = "drive"
        auto_drive_teeth = None
        for train in self._gear_trains:
            if train.input_gear_id != drive_id:
                continue
            if train.gear_pairs:
                auto_drive_teeth = int(train.gear_pairs[0][0])
                break
        drive_teeth = auto_drive_teeth or int(self.drive_teeth_spin.value())
        if auto_drive_teeth is not None and hasattr(self, "drive_teeth_spin"):
            blocker = self.drive_teeth_spin.blockSignals(True)
            self.drive_teeth_spin.setValue(drive_teeth)
            self.drive_teeth_spin.blockSignals(blocker)
        drive_radius = base_radius(drive_teeth)
        drive_center = (drive_offset_x, 0.0, 0.0)
        center_x = 0.0
        drive_shaft = shaft_id_for(drive_center[0], drive_center[2])
        gears[drive_id] = GearRenderSpec(
            gear_id=drive_id,
            teeth=drive_teeth,
            radius_mm=drive_radius,
            thickness_mm=thickness,
            center_mm=drive_center,
            color=(0.85, 0.2, 0.2),
            label=f"Drive ({drive_teeth}T)",
            shaft_index=drive_shaft,
            speed_rps=drive_speed,
            angle_rad=self._gear_angles.get(drive_id, 0.0),
            is_drive=True,
        )
        speed_by_id[drive_id] = drive_speed
        shaft_by_id[drive_id] = drive_shaft
        label_by_id[drive_id] = gears[drive_id].label

        placed_gears: list[GearRenderSpec] = [gears[drive_id]]

        for train in self._gear_trains:
            input_id = train.input_gear_id
            if input_id not in gears:
                continue
            input_spec = gears[input_id]
            input_speed = speed_by_id.get(input_id, 0.0)
            input_teeth = int(input_spec.teeth)

            gear_pairs = list(train.gear_pairs)

            if abs(input_spec.center_mm[0] - center_x) > 1.0e-3:
                transfer_teeth = max(8, input_teeth)
                input_x = input_spec.center_mm[0]
                input_z = input_spec.center_mm[2]
                mid_x = (input_x + center_x) * 0.5
                dist_a = max(abs(mid_x - input_x), 0.1)
                dist_b = max(abs(center_x - mid_x), 0.1)
                r_a, r_b, _module = pair_radii(transfer_teeth, transfer_teeth, dist_a)
                r_c, r_d, _module = pair_radii(transfer_teeth, transfer_teeth, dist_b)
                transfer_y = input_spec.center_mm[1] + layer_height
                for _ in range(60):
                    if not overlaps_existing(input_x, transfer_y, input_z, r_a, placed_gears) and not overlaps_existing(
                        mid_x,
                        transfer_y,
                        input_z,
                        r_b,
                        placed_gears,
                    ):
                        if not overlaps_existing(
                            mid_x,
                            transfer_y + layer_height,
                            input_z,
                            r_c,
                            placed_gears,
                        ) and not overlaps_existing(
                            center_x,
                            transfer_y + layer_height,
                            input_z,
                            r_d,
                            placed_gears,
                        ):
                            break
                    transfer_y += layer_height

                transfer_a_id = f"{train.train_id}_center_a"
                transfer_b_id = f"{train.train_id}_center_b"
                transfer_c_id = f"{train.train_id}_center_c"
                transfer_d_id = f"{train.train_id}_center_d"

                shaft_a = shaft_id_for(input_x, input_z)
                shaft_b = shaft_id_for(mid_x, input_z)
                shaft_c = shaft_id_for(center_x, input_z)

                gears[transfer_a_id] = GearRenderSpec(
                    gear_id=transfer_a_id,
                    teeth=transfer_teeth,
                    radius_mm=r_a,
                    thickness_mm=thickness,
                    center_mm=(input_x, transfer_y, input_z),
                    color=(0.78, 0.78, 0.82),
                    label=f"{train.train_id} Center A ({transfer_teeth}T)",
                    shaft_index=shaft_a,
                    speed_rps=input_speed,
                    angle_rad=self._gear_angles.get(transfer_a_id, 0.0),
                )
                gears[transfer_b_id] = GearRenderSpec(
                    gear_id=transfer_b_id,
                    teeth=transfer_teeth,
                    radius_mm=r_b,
                    thickness_mm=thickness,
                    center_mm=(mid_x, transfer_y, input_z),
                    color=(0.78, 0.78, 0.82),
                    label=f"{train.train_id} Center B ({transfer_teeth}T)",
                    shaft_index=shaft_b,
                    speed_rps=-input_speed,
                    angle_rad=self._gear_angles.get(transfer_b_id, 0.0),
                )
                gears[transfer_c_id] = GearRenderSpec(
                    gear_id=transfer_c_id,
                    teeth=transfer_teeth,
                    radius_mm=r_c,
                    thickness_mm=thickness,
                    center_mm=(mid_x, transfer_y + layer_height, input_z),
                    color=(0.78, 0.78, 0.82),
                    label=f"{train.train_id} Center C ({transfer_teeth}T)",
                    shaft_index=shaft_b,
                    speed_rps=-input_speed,
                    angle_rad=self._gear_angles.get(transfer_c_id, 0.0),
                )
                gears[transfer_d_id] = GearRenderSpec(
                    gear_id=transfer_d_id,
                    teeth=transfer_teeth,
                    radius_mm=r_d,
                    thickness_mm=thickness,
                    center_mm=(center_x, transfer_y + layer_height, input_z),
                    color=(0.78, 0.78, 0.82),
                    label=f"{train.train_id} Center D ({transfer_teeth}T)",
                    shaft_index=shaft_c,
                    speed_rps=input_speed,
                    angle_rad=self._gear_angles.get(transfer_d_id, 0.0),
                )

                for gear in (transfer_a_id, transfer_b_id, transfer_c_id, transfer_d_id):
                    spec = gears[gear]
                    speed_by_id[gear] = spec.speed_rps
                    shaft_by_id[gear] = spec.shaft_index
                    label_by_id[gear] = spec.label
                    placed_gears.append(spec)

                mesh_links.append(
                    (
                        np.array([input_x, transfer_y, input_z], dtype=float),
                        np.array([mid_x, transfer_y, input_z], dtype=float),
                    )
                )
                mesh_links.append(
                    (
                        np.array([mid_x, transfer_y + layer_height, input_z], dtype=float),
                        np.array([center_x, transfer_y + layer_height, input_z], dtype=float),
                    )
                )

                input_spec = gears[transfer_d_id]
                input_speed = input_spec.speed_rps

            if len(gear_pairs) % 2 != 0:
                idler_teeth = max(8, int(input_spec.teeth))
                gear_pairs.append((idler_teeth, idler_teeth))

            y_offset, used_distance, shaft0_x, shaft1_x = choose_layout(
                input_spec,
                gear_pairs,
                placed_gears,
            )

            input_z = input_spec.center_mm[2]
            speed_shaft: list[float | None] = [None, None]
            speed_shaft[0] = input_speed

            driver_shaft = 0
            last_driven_id = None
            max_y = input_spec.center_mm[1] + thickness * 0.5

            for stage_idx, (driver_teeth, driven_teeth) in enumerate(gear_pairs):
                driven_shaft = 1 - driver_shaft
                y = input_spec.center_mm[1] + y_offset + (stage_idx + 1) * layer_height
                driver_speed = speed_shaft[driver_shaft] or 0.0
                ratio = float(driver_teeth) / float(driven_teeth)
                driven_speed = -driver_speed * ratio
                speed_shaft[driven_shaft] = driven_speed

                driver_id = f"{train.train_id}_S{stage_idx + 1}_driver"
                driven_id = f"{train.train_id}_S{stage_idx + 1}_driven"
                driver_x = shaft0_x if driver_shaft == 0 else shaft1_x
                driven_x = shaft0_x if driven_shaft == 0 else shaft1_x
                r_driver, r_driven, _module = pair_radii(driver_teeth, driven_teeth, used_distance)
                driver_shaft_id = shaft_id_for(driver_x, input_z)
                driven_shaft_id = shaft_id_for(driven_x, input_z)

                gears[driver_id] = GearRenderSpec(
                    gear_id=driver_id,
                    teeth=driver_teeth,
                    radius_mm=r_driver,
                    thickness_mm=thickness,
                    center_mm=(driver_x, y, input_z),
                    color=(0.75, 0.75, 0.78),
                    label=f"{train.train_id} S{stage_idx + 1} Driver ({driver_teeth}T)",
                    shaft_index=driver_shaft_id,
                    speed_rps=driver_speed,
                    angle_rad=self._gear_angles.get(driver_id, 0.0),
                )
                gears[driven_id] = GearRenderSpec(
                    gear_id=driven_id,
                    teeth=driven_teeth,
                    radius_mm=r_driven,
                    thickness_mm=thickness,
                    center_mm=(driven_x, y, input_z),
                    color=(0.75, 0.75, 0.78),
                    label=f"{train.train_id} S{stage_idx + 1} Driven ({driven_teeth}T)",
                    shaft_index=driven_shaft_id,
                    speed_rps=driven_speed,
                    angle_rad=self._gear_angles.get(driven_id, 0.0),
                )
                speed_by_id[driver_id] = driver_speed
                speed_by_id[driven_id] = driven_speed
                shaft_by_id[driver_id] = driver_shaft_id
                shaft_by_id[driven_id] = driven_shaft_id
                label_by_id[driver_id] = gears[driver_id].label
                label_by_id[driven_id] = gears[driven_id].label
                last_driven_id = driven_id
                driver_shaft = driven_shaft
                max_y = max(max_y, y + thickness * 0.5)
                mesh_links.append(
                    (
                        np.array([driver_x, y, input_z], dtype=float),
                        np.array([driven_x, y, input_z], dtype=float),
                    )
                )

            for stage_idx, _pair in enumerate(gear_pairs):
                driver_id = f"{train.train_id}_S{stage_idx + 1}_driver"
                driven_id = f"{train.train_id}_S{stage_idx + 1}_driven"
                placed_gears.append(gears[driver_id])
                placed_gears.append(gears[driven_id])

            if last_driven_id:
                train_outputs[train.train_id] = last_driven_id
                train_max_y[train.train_id] = max_y

        arms: dict[str, ArmRenderSpec] = {}
        global_max_y = 0.0
        for gear in gears.values():
            global_max_y = max(global_max_y, gear.center_mm[1] + thickness * 0.5)
        arm_entries: list[tuple[float, GearTrainEntry, str]] = []
        for train in self._gear_trains:
            output_id = train_outputs.get(train.train_id)
            if not output_id:
                continue
            length = self._scaled_planet_mean_distance(train.target_body)
            if length is None:
                continue
            arm_entries.append((length, train, output_id))

        arm_entries.sort(key=lambda item: item[0], reverse=True)

        for idx, (length, train, output_id) in enumerate(arm_entries):
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

        arm_entries_by_length = sorted(arm_entries, key=lambda item: item[0])
        sleeve_requirements: list[tuple[int, float, float]] = []
        sleeve_outer_by_shaft: dict[int, float] = {}

        for _length, train, output_id in arm_entries_by_length:
            output_gear = gears.get(output_id)
            if output_gear is None:
                continue
            arm_id = f"arm_{train.train_id}"
            arm = arms.get(arm_id)
            if arm is None:
                continue
            base_height = output_gear.center_mm[1] + thickness * 0.5
            sleeve_height = arm.origin_mm[1] - base_height
            if sleeve_height <= 0.05:
                continue
            prev_outer = sleeve_outer_by_shaft.get(output_gear.shaft_index, 0.0)
            sleeve_inner = 0.0 if prev_outer <= 0.0 else prev_outer + min_bore_clear
            sleeve_outer = sleeve_inner + max(min_wall, 0.0)
            sleeve_outer_by_shaft[output_gear.shaft_index] = sleeve_outer
            sleeve_id = f"{train.train_id}_sleeve"
            sleeve = GearRenderSpec(
                gear_id=sleeve_id,
                teeth=0,
                radius_mm=sleeve_outer,
                bore_radius_mm=sleeve_inner,
                thickness_mm=sleeve_height,
                center_mm=(
                    output_gear.center_mm[0],
                    base_height + sleeve_height * 0.5,
                    output_gear.center_mm[2],
                ),
                color=(0.7, 0.7, 0.72),
                label="",
                shaft_index=output_gear.shaft_index,
                speed_rps=output_gear.speed_rps,
                angle_rad=self._gear_angles.get(sleeve_id, 0.0),
            )
            gears[sleeve_id] = sleeve
            speed_by_id[sleeve_id] = sleeve.speed_rps
            shaft_by_id[sleeve_id] = sleeve.shaft_index
            sleeve_requirements.append((output_gear.shaft_index, base_height, sleeve_outer))

        speed_palette = [
            (0.86, 0.42, 0.32),
            (0.36, 0.62, 0.86),
            (0.45, 0.78, 0.52),
            (0.86, 0.74, 0.3),
            (0.74, 0.46, 0.86),
            (0.32, 0.82, 0.78),
            (0.82, 0.52, 0.25),
            (0.62, 0.62, 0.92),
        ]

        def speed_key(speed: float) -> float:
            if abs(speed) < 1.0e-10:
                return 0.0
            return round(speed, 9)

        speed_colors: dict[float, Color] = {}
        for gear in gears.values():
            key = speed_key(gear.speed_rps)
            if key not in speed_colors:
                speed_colors[key] = speed_palette[len(speed_colors) % len(speed_palette)]
            gear.color = speed_colors[key]

        gears_by_shaft: dict[int, list[GearRenderSpec]] = {}
        for gear in gears.values():
            gears_by_shaft.setdefault(gear.shaft_index, []).append(gear)

        for shaft_gears in gears_by_shaft.values():
            for gear in shaft_gears:
                if gear.teeth == 0:
                    continue
                gear_top = gear.center_mm[1] + gear.thickness_mm * 0.5
                required = 0.0
                for shaft_idx, base_height, sleeve_outer in sleeve_requirements:
                    if gear.shaft_index != shaft_idx:
                        continue
                    if gear_top <= base_height + 1.0e-6:
                        continue
                    required = max(required, sleeve_outer + min_bore_clear)
                if required <= 0.0:
                    gear.bore_radius_mm = 0.0
                    continue
                max_bore = max(0.0, gear.radius_mm - max(min_wall, 0.0))
                if required <= max_bore:
                    gear.bore_radius_mm = required
                else:
                    gear.bore_radius_mm = 0.0

        shaft_ranges: dict[int, tuple[float, float, float, float]] = {}
        for gear in gears.values():
            min_y = gear.center_mm[1] - gear.thickness_mm * 0.5
            max_y = gear.center_mm[1] + gear.thickness_mm * 0.5
            if gear.shaft_index not in shaft_ranges:
                shaft_ranges[gear.shaft_index] = (min_y, max_y, gear.center_mm[0], gear.center_mm[2])
            else:
                prev_min, prev_max, x, z = shaft_ranges[gear.shaft_index]
                shaft_ranges[gear.shaft_index] = (min(prev_min, min_y), max(prev_max, max_y), x, z)

        for min_y, max_y, x, z in shaft_ranges.values():
            if max_y > min_y + 0.01:
                shaft_links.append(
                    (
                        np.array([x, min_y, z], dtype=float),
                        np.array([x, max_y, z], dtype=float),
                    )
                )

        self._gear_speed_by_id = speed_by_id
        self._gear_shaft_by_id = shaft_by_id
        self._gear_label_by_id = label_by_id
        for gear_id in list(self._gear_angles.keys()):
            if gear_id not in gears:
                self._gear_angles.pop(gear_id, None)

        if self._selected_input_gear_id not in self._gear_label_by_id:
            self._selected_input_gear_id = drive_id

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
                shaft_links = [
                    (
                        np.array([a[0], a[1] - shift, a[2]], dtype=float),
                        np.array([b[0], b[1] - shift, b[2]], dtype=float),
                    )
                    for a, b in shaft_links
                ]

        for arm_id in list(self._arm_angles.keys()):
            if arm_id not in arms:
                self._arm_angles.pop(arm_id, None)
        self._gear_arms = arms
        self.gl_widget.set_gears(gears)
        self.gl_widget.set_arms(arms)
        self.gl_widget.set_mesh_links(tuple(mesh_links))
        self.gl_widget.set_shaft_links(tuple(shaft_links))
        self._update_gear_list()
        self._set_selected_gear(self._selected_input_gear_id)

    def _advance_gear_animation(self, delta: float) -> None:
        if not self._gear_arms:
            return
        time_scale = max(self._time_speed, 0.0)
        for gear_id, speed in self._gear_speed_by_id.items():
            angle = self._gear_angles.get(gear_id, 0.0)
            angle = (angle - float(delta) * speed * time_scale * 2.0 * math.pi) % (2.0 * math.pi)
            self._gear_angles[gear_id] = angle
        for arm_id, arm in self._gear_arms.items():
            angle = self._arm_angles.get(arm_id, arm.angle_rad)
            angle = (angle + float(delta) * arm.speed_rps * time_scale * 2.0 * math.pi) % (2.0 * math.pi)
            self._arm_angles[arm_id] = angle
            arm.angle_rad = angle
        self.gl_widget.set_gear_angles(self._gear_angles)
        self.gl_widget.update()
