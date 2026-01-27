from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .models import BODY_CATALOG, BodyInfo, BodyState, OrbitStats

try:
    import spiceypy as spice
except ImportError:  # pragma: no cover - optional dependency until installed
    spice = None


@dataclass
class KernelLoadResult:
    loaded: bool
    kernels: list[Path]
    errors: list[str]


def _default_kernel_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "kernels"


def _vec3_tuple(values: Iterable[float]) -> tuple[float, float, float]:
    arr = np.asarray(tuple(values), dtype=float)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


class SpiceEphemeris:
    def __init__(self, kernel_dir: Path | None = None) -> None:
        self.kernel_dir = kernel_dir or _default_kernel_dir()
        self._kernels_loaded: list[Path] = []
        self._load_errors: list[str] = []
        self._loaded = False

    @property
    def available(self) -> bool:
        return spice is not None

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def status_message(self) -> str:
        if not self.available:
            return "spiceypy not installed"
        if not self._loaded:
            return "no kernels loaded"
        return f"{len(self._kernels_loaded)} kernels loaded"

    @property
    def load_errors(self) -> list[str]:
        return list(self._load_errors)

    def load_kernels(self, kernel_dir: Path | None = None) -> KernelLoadResult:
        spice_mod = spice
        if spice_mod is None:
            return KernelLoadResult(False, [], ["spiceypy not installed"])

        target_dir = kernel_dir or self.kernel_dir
        if not target_dir.exists():
            self._loaded = False
            self._kernels_loaded = []
            self._load_errors = [f"kernel dir not found: {target_dir}"]
            return KernelLoadResult(False, [], list(self._load_errors))

        spice_mod.kclear()
        self._kernels_loaded = []
        self._load_errors = []

        kernel_paths = sorted(
            p
            for p in target_dir.rglob("*")
            if p.suffix.lower() in {".bsp", ".tls", ".tpc", ".bpc", ".tf"}
        )

        for path in kernel_paths:
            try:
                spice_mod.furnsh(str(path))
                self._kernels_loaded.append(path)
            except Exception as exc:
                self._load_errors.append(f"{path.name}: {exc}")

        self._loaded = len(self._kernels_loaded) > 0
        return KernelLoadResult(self._loaded, list(self._kernels_loaded), list(self._load_errors))

    def get_positions(self, when: datetime) -> dict[str, BodyState]:
        spice_mod = spice
        if spice_mod is None or not self._loaded:
            return self._approximate_positions(when, reason="kernels unavailable")

        try:
            when_utc = when.astimezone(UTC)
            et = spice_mod.utc2et(when_utc.strftime("%Y-%m-%dT%H:%M:%S"))
        except Exception as exc:
            return self._approximate_positions(when, reason=str(exc))

        positions: dict[str, BodyState] = {}
        parent_positions: dict[str, np.ndarray] = {}

        for body in BODY_CATALOG:
            pos = self._position_for_body(body, et, parent_positions)
            positions[body.name] = pos

        return positions

    def get_body_rotations(self, when: datetime) -> dict[str, np.ndarray]:
        spice_mod = spice
        if spice_mod is None or not self._loaded:
            return {}

        try:
            when_utc = when.astimezone(UTC)
            et = spice_mod.utc2et(when_utc.strftime("%Y-%m-%dT%H:%M:%S"))
        except Exception:
            return {}

        rotations: dict[str, np.ndarray] = {}
        for body in BODY_CATALOG:
            frame = f"IAU_{body.spice_name.upper()}"
            try:
                rot = spice_mod.pxform(frame, "J2000", et)
            except Exception:
                continue
            rotations[body.name] = np.asarray(rot, dtype=np.float32)

        return rotations

    def get_body_spin_directions(
        self,
        when: datetime,
        dt_seconds: float = 60.0,
    ) -> dict[str, float]:
        spice_mod = spice
        if spice_mod is None or not self._loaded:
            return {}
        if dt_seconds <= 0:
            dt_seconds = 60.0

        try:
            when_utc = when.astimezone(UTC)
            et0 = spice_mod.utc2et(when_utc.strftime("%Y-%m-%dT%H:%M:%S"))
        except Exception:
            return {}

        rotations: dict[str, float] = {}
        et1 = et0 + dt_seconds
        for body in BODY_CATALOG:
            frame = f"IAU_{body.spice_name.upper()}"
            try:
                rot0 = np.asarray(spice_mod.pxform(frame, "J2000", et0), dtype=np.float64)
                rot1 = np.asarray(spice_mod.pxform(frame, "J2000", et1), dtype=np.float64)
            except Exception:
                continue

            delta = rot1 @ rot0.T
            skew = 0.5 * (delta - delta.T)
            w_inertial = np.array(
                [skew[2, 1], skew[0, 2], skew[1, 0]],
                dtype=np.float64,
            ) / dt_seconds
            w_body = rot0.T @ w_inertial
            z = float(w_body[2])
            if abs(z) < 1.0e-12:
                continue
            rotations[body.name] = 1.0 if z >= 0.0 else -1.0

        return rotations

    def get_orbital_plane_normal(
        self,
        when: datetime,
        body: str = "EARTH",
        center: str = "SUN",
    ) -> tuple[float, float, float] | None:
        spice_mod = spice
        if spice_mod is None or not self._loaded:
            return None

        try:
            when_utc = when.astimezone(UTC)
            et = spice_mod.utc2et(when_utc.strftime("%Y-%m-%dT%H:%M:%S"))
        except Exception:
            return None

        try:
            state, _ = spice_mod.spkezr(body, et, "J2000", "NONE", center)
        except Exception:
            return None
        state_arr = np.asarray(state, dtype=float)
        r = state_arr[0:3]
        v = state_arr[3:6]
        h = np.cross(r, v)
        norm = np.linalg.norm(h)
        if norm <= 0:
            return None
        h_unit = h / norm
        return (float(h_unit[0]), float(h_unit[1]), float(h_unit[2]))

    def get_orbit_stats(
        self,
        when: datetime,
        bodies: Iterable[BodyInfo],
        samples: int = 180,
        span_orbits: int = 100,
        plane_normal: tuple[float, float, float] | None = None,
    ) -> dict[str, OrbitStats]:
        spice_mod = spice
        if spice_mod is None or not self._loaded:
            return {}

        try:
            when_utc = when.astimezone(UTC)
            et0 = spice_mod.utc2et(when_utc.strftime("%Y-%m-%dT%H:%M:%S"))
        except Exception:
            return {}

        results: dict[str, OrbitStats] = {}
        normal_vec = None
        if plane_normal is not None:
            normal_vec = np.array(plane_normal, dtype=float)
            norm = np.linalg.norm(normal_vec)
            normal_vec = normal_vec / norm if norm > 0 else None
        for body in bodies:
            if body.spice_name.upper() == "SUN":
                continue
            parent = body.parent
            if not parent:
                continue

            period_days = body.orbital_period_days
            if period_days and period_days > 0:
                duration = period_days * 86400.0 * max(span_orbits, 1)
            else:
                duration = 365.0 * 86400.0 * max(span_orbits, 1)

            if samples < 2:
                samples = 2
            start = et0 - duration / 2.0
            step = duration / (samples - 1)

            min_km = None
            max_km = None
            max_normal = 0.0

            for idx in range(samples):
                et = start + step * idx
                try:
                    pos, _ = spice_mod.spkpos(body.spice_name, et, "J2000", "NONE", parent)
                    pos_arr = np.asarray(pos, dtype=float)
                    r = float(np.linalg.norm(pos_arr))
                    if normal_vec is not None:
                        normal_dist = abs(float(np.dot(pos_arr, normal_vec)))
                        if normal_dist > max_normal:
                            max_normal = normal_dist
                except Exception:
                    r = None
                if r is None:
                    continue
                if min_km is None or r < min_km:
                    min_km = r
                if max_km is None or r > max_km:
                    max_km = r

            if min_km is None or max_km is None:
                if body.mean_distance_km:
                    min_km = body.mean_distance_km
                    max_km = body.mean_distance_km
                else:
                    continue
            results[body.name] = OrbitStats(min_km=min_km, max_km=max_km, max_normal_km=max_normal)

        return results

    def get_orbit_paths(
        self,
        when: datetime,
        bodies: Iterable[BodyInfo],
        samples_per_orbit: int = 180,
        span_orbits: int = 10,
        frame: str = "heliocentric",
    ) -> dict[str, np.ndarray]:
        spice_mod = spice
        if spice_mod is None or not self._loaded:
            return {}

        try:
            when_utc = when.astimezone(UTC)
            et0 = spice_mod.utc2et(when_utc.strftime("%Y-%m-%dT%H:%M:%S"))
        except Exception:
            return {}

        results: dict[str, np.ndarray] = {}
        for body in bodies:
            if body.spice_name.upper() == "SUN":
                continue
            parent = body.parent or "SUN"
            center = "SUN" if frame == "heliocentric" else parent
            period_days = body.orbital_period_days
            if period_days is None or period_days <= 0:
                continue
            orbit_count = max(span_orbits, 1)
            total_samples = max(2, int(samples_per_orbit) * orbit_count)
            duration = period_days * 86400.0 * orbit_count
            times = np.linspace(et0, et0 + duration, total_samples, dtype=float)
            positions: list[np.ndarray] = []
            for et in times:
                try:
                    pos, _ = spice_mod.spkpos(body.spice_name, float(et), "J2000", "NONE", center)
                except Exception:
                    break
                positions.append(np.asarray(pos, dtype=float))
            if len(positions) < 2:
                continue
            results[body.name] = np.asarray(positions, dtype=float)

        return results

    def _position_for_body(
        self,
        body: BodyInfo,
        et: float,
        parent_positions: dict[str, np.ndarray],
    ) -> BodyState:
        spice_mod = spice
        if spice_mod is None:
            return BodyState((0.0, 0.0, 0.0), False, "spiceypy not installed")
        try:
            if body.spice_name.upper() == "SUN" and body.parent.upper() == "SUN":
                return BodyState((0.0, 0.0, 0.0), True)
            if body.parent.upper() == "SUN":
                position, _ = spice_mod.spkpos(body.spice_name, et, "J2000", "NONE", "SUN")
                return BodyState(_vec3_tuple(position), True)

            parent_position = parent_positions.get(body.parent)
            if parent_position is None:
                parent_position, _ = spice_mod.spkpos(body.parent, et, "J2000", "NONE", "SUN")
                parent_positions[body.parent] = np.array(parent_position, dtype=float)

            relative_position, _ = spice_mod.spkpos(body.spice_name, et, "J2000", "NONE", body.parent)
            absolute_position = parent_position + np.array(relative_position, dtype=float)
            return BodyState(_vec3_tuple(absolute_position), True)
        except Exception as exc:
            return BodyState((0.0, 0.0, 0.0), False, str(exc))

    def _approximate_positions(self, when: datetime, reason: str) -> dict[str, BodyState]:
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = when.timestamp()

        positions: dict[str, BodyState] = {}
        for body in BODY_CATALOG:
            if body.spice_name.upper() == "SUN":
                positions[body.name] = BodyState((0.0, 0.0, 0.0), True)
                continue
            position = self._approximate_orbit(body, seconds)
            positions[body.name] = BodyState(position, False, reason)
        return positions

    def _approximate_orbit(self, body: BodyInfo, seconds: float) -> tuple[float, float, float]:
        if body.mean_distance_km is None or body.orbital_period_days is None:
            return (0.0, 0.0, 0.0)
        period_seconds = body.orbital_period_days * 86400.0
        angle = (seconds % period_seconds) / period_seconds * (2.0 * np.pi)
        radius = body.mean_distance_km
        return (float(radius * np.cos(angle)), 0.0, float(radius * np.sin(angle)))
