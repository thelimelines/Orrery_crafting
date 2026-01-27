from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from ..models import BODY_BY_NAME, BODY_CATALOG

if TYPE_CHECKING:
    _MainWindowBase = Any
else:
    _MainWindowBase = object


class OrbitPathMixin(_MainWindowBase):
    def _toggle_orbit_paths(self, checked: bool) -> None:
        if checked:
            self._update_orbit_paths(self._current_datetime(), refresh_orbit_stats=True)
        else:
            self._orbit_paths = {}
            self.gl_widget.set_orbit_paths({}, centers={})

    def _toggle_orbit_paths_focus(self, _checked: bool) -> None:
        if not self.orbit_paths_check.isChecked():
            return
        self._update_orbit_paths(self._current_datetime(), refresh_orbit_stats=True)

    def _update_orbit_paths(self, when: datetime, refresh_orbit_stats: bool) -> None:
        if not self.orbit_paths_check.isChecked():
            self._orbit_paths = {}
            self.gl_widget.set_orbit_paths({}, centers={})
            return

        key = (
            self.sample_spin.value(),
            self.span_orbits_spin.value(),
            when,
            self.mechanical_check.isChecked(),
            self._focused_body if self.orbit_paths_focus_check.isChecked() else None,
            self.gl_widget.global_distance_scale,
            self.gl_widget.global_radius_scale,
            self.gl_widget.min_visible_radius,
            self.gl_widget.size_log_base,
        )
        if key == self._orbit_paths_key:
            return

        if refresh_orbit_stats:
            self._orbit_stats = self.spice.get_orbit_stats(
                when,
                BODY_CATALOG,
                samples=self.sample_spin.value(),
                span_orbits=self.span_orbits_spin.value(),
                plane_normal=self._solar_plane_normal,
            )
            self._orbit_stats_key = (self.sample_spin.value(), self.span_orbits_spin.value())

        focus_only = self.orbit_paths_focus_check.isChecked()
        bodies = self._orbit_path_bodies(focus_only)
        if not bodies:
            self._orbit_paths = {}
            self.gl_widget.set_orbit_paths({}, centers={})
            return

        orbits = self.span_orbits_spin.value()
        samples_per_orbit = self.sample_spin.value()
        paths = self.spice.get_orbit_paths(
            when,
            bodies,
            samples_per_orbit=samples_per_orbit,
            span_orbits=orbits,
            frame="parent",
        )

        smoothed: dict[str, np.ndarray] = {}
        centers: dict[str, str | None] = {}
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
