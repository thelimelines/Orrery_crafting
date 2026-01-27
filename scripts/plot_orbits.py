from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from orrery_app.models import BODY_CATALOG, BodyInfo  # noqa: E402
from orrery_app.spice_service import SpiceEphemeris  # noqa: E402

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

try:
    from scipy.interpolate import splprep, splev
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required for spline interpolation. Install with: pip install scipy") from exc


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _body_lookup() -> Dict[str, BodyInfo]:
    return {body.name.lower(): body for body in BODY_CATALOG}


def _select_bodies(selection: str) -> List[BodyInfo]:
    selection = selection.strip().lower()
    lookup = _body_lookup()
    if selection in {"planets", "planet"}:
        return [body for body in BODY_CATALOG if body.parent.upper() == "SUN" and body.spice_name.upper() != "SUN"]
    if selection in {"moons", "moon"}:
        return [body for body in BODY_CATALOG if body.parent.upper() != "SUN"]
    if selection in {"all", "everything"}:
        return [body for body in BODY_CATALOG if body.spice_name.upper() != "SUN"]

    names = [name.strip().lower() for name in selection.split(",") if name.strip()]
    result: List[BodyInfo] = []
    for name in names:
        body = lookup.get(name)
        if body is None:
            raise SystemExit(f"Unknown body '{name}'. Use --bodies planets|moons|all or a comma list of names.")
        if body.spice_name.upper() == "SUN":
            continue
        result.append(body)
    return result


def _sample_positions(
    spice: SpiceEphemeris,
    body: BodyInfo,
    start: datetime,
    orbits: int,
    samples_per_orbit: int,
    frame: str,
) -> Tuple[np.ndarray, np.ndarray]:
    import spiceypy as spice_api  # local import after kernels load

    period_days = body.orbital_period_days
    if period_days is None or period_days <= 0:
        raise ValueError(f"Body '{body.name}' has no orbital period in the catalog.")

    total_samples = max(2, orbits * samples_per_orbit)
    duration = period_days * 86400.0 * max(orbits, 1)

    start_utc = start.astimezone(timezone.utc)
    et0 = spice_api.utc2et(start_utc.strftime("%Y-%m-%dT%H:%M:%S"))
    times = np.linspace(et0, et0 + duration, total_samples, dtype=float)

    center = "SUN" if frame == "heliocentric" else body.parent

    positions = np.zeros((total_samples, 3), dtype=float)
    for idx, et in enumerate(times):
        pos, _ = spice_api.spkpos(body.spice_name, float(et), "J2000", "NONE", center)
        positions[idx, :] = np.asarray(pos, dtype=float)

    return times, positions


def _spline_curve(positions: np.ndarray, smoothing: float, resolution: int) -> np.ndarray:
    data = positions.T
    if positions.shape[0] < 4:
        return positions
    tck, _ = splprep(data, s=smoothing, k=3)
    u_fine = np.linspace(0.0, 1.0, resolution, dtype=float)
    x_s, y_s, z_s = splev(u_fine, tck)
    return np.vstack((x_s, y_s, z_s)).T


def _set_axes_equal(ax) -> None:
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    spans = limits[:, 1] - limits[:, 0]
    centers = limits.mean(axis=1)
    radius = 0.5 * spans.max()
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)


def _rgb(color: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (float(color[0]), float(color[1]), float(color[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 3D orbital paths with spline smoothing.")
    parser.add_argument("--bodies", default="planets", help="planets|moons|all or comma list of names")
    parser.add_argument("--orbits", type=int, default=10, help="Number of orbits to span")
    parser.add_argument("--samples-per-orbit", type=int, default=180, help="Samples per orbit")
    parser.add_argument("--frame", choices=["parent", "heliocentric"], default="parent")
    parser.add_argument("--smoothing", type=float, default=0.0, help="Spline smoothing factor")
    parser.add_argument("--resolution", type=int, default=2000, help="Spline resolution")
    parser.add_argument("--start", default=None, help="UTC start time, e.g. 2026-01-26T00:00:00")
    parser.add_argument("--no-spline", action="store_true", help="Plot raw samples without spline")
    args = parser.parse_args()

    bodies = _select_bodies(args.bodies)
    if not bodies:
        raise SystemExit("No bodies selected.")

    if args.start is None:
        start = _now_utc()
    else:
        start = datetime.fromisoformat(args.start)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)

    spice = SpiceEphemeris()
    load_result = spice.load_kernels()
    if not load_result.loaded:
        errors = "\n".join(load_result.errors) if load_result.errors else "unknown error"
        raise SystemExit(f"Failed to load SPICE kernels:\n{errors}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for body in bodies:
        try:
            _times, positions = _sample_positions(
                spice,
                body,
                start,
                args.orbits,
                args.samples_per_orbit,
                args.frame,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {body.name}: {exc}")
            continue

        if args.no_spline:
            curve = positions
        else:
            curve = _spline_curve(positions, args.smoothing, args.resolution)

        color = _rgb(body.color)
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, label=body.name, linewidth=1.5)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color=color, s=12)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color=color, s=12, marker="x")

    ax.set_title("Orbital Paths (Spline Smoothed)")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend(loc="upper right", fontsize="small")
    _set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
