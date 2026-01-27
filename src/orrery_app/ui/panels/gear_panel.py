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
        arm_lift = float(self.arm_clearance_spin.value())
        arm_v_clear = float(self.arm_vertical_clearance_spin.value())
        layer_height = max(thickness + v_clear, 0.1)

        def radius_for(teeth: int) -> float:
            return max(0.5, (float(teeth) * gear_scale) / (2.0 * math.pi))

        gears: dict[str, GearRenderSpec] = {}
        speed_by_id: dict[str, float] = {}
        shaft_by_id: dict[str, int] = {}
        label_by_id: dict[str, str] = {}
        train_outputs: dict[str, str] = {}
        train_max_y: dict[str, float] = {}
        mesh_links: list[tuple[np.ndarray, np.ndarray]] = []

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

            speed_shaft: list[float | None] = [None, None]
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

        arms: dict[str, ArmRenderSpec] = {}
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
