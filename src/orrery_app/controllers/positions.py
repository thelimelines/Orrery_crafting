from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt

from ..models import BODY_CATALOG

if TYPE_CHECKING:
    _MainWindowBase = Any
else:
    _MainWindowBase = object


class PositionMixin(_MainWindowBase):
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
                value = value.replace(tzinfo=UTC)
            return value
        return datetime.fromisoformat(dt.toString(Qt.DateFormat.ISODate)).replace(tzinfo=UTC)

    def _focus_selected_body(self, _item=None) -> None:
        item = self.body_list.currentItem()
        if not item:
            return
        name = item.text()
        self._focused_body = name
        self._apply_focus_system(name, update_camera=False)
        self.gl_widget.focus_on_body(name)

    def _focus_body_from_view(self, name: str) -> None:
        self._focused_body = name
        self._apply_focus_system(name, update_camera=False)
        self.gl_widget.focus_on_body(name)
