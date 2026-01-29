from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

TEXTURE_ROOT = Path(__file__).resolve().parent / "data" / "textures"

TEXTURE_CANDIDATES: dict[str, Iterable[str]] = {
    "Sun": ("8k/sun.jpg", "2k/sun.jpg"),
    "Mercury": ("8k/mercury.jpg", "2k/mercury.jpg"),
    "Venus": ("8k/venus_surface.jpg", "2k/venus_surface.jpg"),
    "Earth": ("8k/earth_day.jpg", "2k/earth_day.jpg"),
    "Moon": ("8k/moon.jpg", "2k/moon.jpg"),
    "Mars": ("8k/mars.jpg", "2k/mars.jpg"),
    "Phobos": ("low/phobos.jpg",),
    "Deimos": ("low/deimos.jpg",),
    "Jupiter": ("8k/jupiter.jpg", "2k/jupiter.jpg"),
    "Io": ("low/io.jpg",),
    "Europa": ("low/europa.jpg",),
    "Ganymede": ("low/ganymede.jpg",),
    "Callisto": ("low/callisto.jpg",),
    "Saturn": ("8k/saturn.jpg", "2k/saturn.jpg"),
    "Titan": ("low/titan.jpg",),
    "Enceladus": ("low/enceladus.jpg",),
    "Rhea": ("low/rhea.jpg",),
    "Iapetus": ("low/iapetus.jpg",),
    "Dione": ("low/dione.jpg",),
    "Tethys": ("low/tethys.jpg",),
    "Mimas": ("low/mimas.jpg",),
    "Uranus": ("2k/uranus.jpg",),
    "Miranda": ("low/miranda.jpg",),
    "Ariel": ("low/ariel.jpg",),
    "Umbriel": ("low/umbriel.jpg",),
    "Titania": ("low/titania.jpg",),
    "Oberon": ("low/oberon.jpg",),
    "Neptune": ("2k/neptune.jpg",),
    "Triton": ("low/triton.jpg",),
    "Pluto": ("low/pluto.jpg",),
    "Charon": ("low/charon.jpg",),
    "Ceres": ("4k/ceres.jpg", "2k/ceres.jpg"),
    "Eris": ("4k/eris.jpg", "2k/eris.jpg"),
    "Haumea": ("4k/haumea.jpg", "2k/haumea.jpg"),
    "Makemake": ("4k/makemake.jpg", "2k/makemake.jpg"),
    "SaturnRings": ("extras/saturn_rings.png",),
}


def resolve_texture_paths() -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, candidates in TEXTURE_CANDIDATES.items():
        for candidate in candidates:
            path = TEXTURE_ROOT / candidate
            if path.exists():
                resolved[name] = path
                break
    return resolved
