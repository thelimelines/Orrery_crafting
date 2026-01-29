from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from ..models import (
    BODY_BY_NAME,
    BODY_CATALOG,
    SATURN_RING_INNER_KM,
    SATURN_RING_OUTER_KM,
    BodySettings,
    BodyState,
)

if TYPE_CHECKING:
    _MainWindowBase = Any
else:
    _MainWindowBase = object


class MechanicalLayoutMixin(_MainWindowBase):
    def _apply_mechanical_settings(self, _value: int | None = None) -> None:
        enabled = self.mechanical_check.isChecked()
        self.gl_widget.mechanical_mode = enabled
        self.gl_widget.orbit_band_opacity = self.band_opacity_slider.value() / 100.0
        self.gl_widget.show_orbit_bands = self.bands_check.isChecked() if enabled else False
        self.gl_widget.show_moon_bands = False
        self.gl_widget.real_scale_mode = False
        self._planet_scale = 1.0
        self._moon_scale_by_parent = {}
        self._moon_scaled_bounds = {}

        if enabled:
            self.gl_widget.min_visible_radius = 0.2
            self.gl_widget.highlight_threshold_earth_units = 1.0
            self.gl_widget.size_mode = "log"
        else:
            self.gl_widget.min_visible_radius = 0.0
            self.gl_widget.highlight_threshold_earth_units = 0.0
            self.gl_widget.size_mode = "linear"

        self.orbit_paths_check.setEnabled(enabled)
        self.orbit_paths_focus_check.setEnabled(enabled)
        if not enabled:
            self.orbit_paths_check.setChecked(False)
            self.orbit_paths_focus_check.setChecked(False)
            self._orbit_paths = {}
            self.gl_widget.set_orbit_paths({}, centers={})

        self.real_scale_btn.setEnabled(True)
        show_bands = self.bands_check.isChecked()
        self.band_opacity_label.setVisible(show_bands)
        self.band_opacity_slider.setVisible(show_bands)
        self.band_opacity_slider.setEnabled(show_bands)
        self._apply_band_opacity(self.band_opacity_slider.value())

        self._refresh_positions()
        self._rebuild_gear_layout()

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
        for body in BODY_CATALOG:
            state = positions.get(body.name)
            if state and state.available:
                dist = float(np.linalg.norm(np.array(state.position_km, dtype=float)))
                max_dist = max(max_dist, dist)
        if max_dist > 0:
            scale = self.gl_widget.real_scale_max_distance_km / max_dist
            self.gl_widget.global_distance_scale = scale
            self.gl_widget.global_radius_scale = scale
        self.gl_widget.update()
        self._refresh_positions(refresh_orbit_stats=False, update_diagnostics=False)

    def _apply_band_opacity(self, value: int) -> None:
        self.gl_widget.orbit_band_opacity = float(value) / 100.0
        self.gl_widget.update()

    def _apply_focus_system(self, name: str | None, update_camera: bool = True) -> None:
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

    def _system_plane_normal(self, name: str) -> tuple[float, float, float] | None:
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
        positions: dict[str, BodyState],
    ) -> tuple[
        dict[str, BodyState],
        dict[str, float],
        dict[str, tuple[float, float, float]],
        dict[str, tuple[tuple[float, float], ...]],
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

        body_radius_mm: dict[str, float] = {}
        for body in BODY_CATALOG:
            settings = self._settings.get(body.name, BodySettings())
            body_radius_mm[body.name] = size_radius_mm(body.radius_km, settings.radius_scale)

        moons_by_parent: dict[str, list] = {}
        for body in BODY_CATALOG:
            if body.parent and body.parent.upper() != "SUN":
                parent_name = body.parent.title()
                moons_by_parent.setdefault(parent_name, []).append(body)
        saturn_moons = moons_by_parent.get("Saturn")
        if saturn_moons is not None:
            saturn_moons = list(saturn_moons)
            saturn_moons.append(SimpleNamespace(name="SaturnRings"))
            moons_by_parent["Saturn"] = saturn_moons

        moon_scaled_bounds: dict[str, tuple[float, float]] = {}
        moon_bands_by_parent: dict[str, list] = {}
        moon_scale_by_parent: dict[str, float] = {}
        system_radius: dict[str, float] = {}

        for parent, moons in moons_by_parent.items():
            parent_radius = body_radius_mm.get(parent, body_radius_mm.get(parent.title(), 1.0))
            entries: list[tuple[Any, float, float, bool]] = []
            min_km = None
            for moon in moons:
                if moon.name == "SaturnRings":
                    moon_min = SATURN_RING_INNER_KM
                    moon_max = SATURN_RING_OUTER_KM
                    is_ring = True
                else:
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
                    is_ring = False
                if not is_ring and (min_km is None or moon_min < min_km):
                    min_km = moon_min
                entries.append((moon, moon_min, moon_max, is_ring))

            if not entries:
                continue

            moon_scale = 1.0
            if min_km and min_km > 0:
                moon_scale = (parent_radius + moon_clearance) / min_km
            moon_scale_by_parent[parent] = moon_scale

            entries.sort(key=lambda item: item[1])
            prev_outer = parent_radius
            for moon, moon_min, moon_max, is_ring in entries:
                raw_min = moon_min * moon_scale
                raw_max = moon_max * moon_scale
                if is_ring:
                    moon_radius = 0.0
                    scaled_min = raw_min
                    scaled_max = raw_max
                    ring_thickness = max(scaled_max - scaled_min, 0.0)
                    outer_surface = scaled_max + ring_thickness
                    inner_surface = max(scaled_min, 0.0)
                else:
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
        planet_bounds: dict[str, tuple[float, float]] = {}
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

        scaled_positions: dict[str, BodyState] = {}

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

        system_bounds_mm: dict[str, float] = {}
        orbit_sweeps_mm: dict[str, tuple[float, float, float]] = {}
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

        moon_bands_final: dict[str, tuple[tuple[float, float], ...]] = {}
        for parent, bands in moon_bands_by_parent.items():
            moon_bands_final[parent] = tuple(bands)

        self._planet_bounds = dict(planet_bounds)
        self._planet_scale = float(planet_scale)
        self._moon_scaled_bounds = dict(moon_scaled_bounds)
        self._moon_scale_by_parent = dict(moon_scale_by_parent)

        return scaled_positions, system_bounds_mm, orbit_sweeps_mm, moon_bands_final
