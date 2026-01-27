# Orrery_crafting
Tools for developing mechanical planetariums (orreries).

## Install (Windows PowerShell)
Prerequisite: Python 3.12.x on PATH.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

This installs the runtime dependencies declared in `pyproject.toml`:
numba, numpy, PyOpenGL, PySide6, scipy, spiceypy, and pyright.

## SPICE-powered viewer (PySide6 + OpenGL)
Run the interactive solar-system viewer:

```bash
pip install -e .
python -m orrery_app
```

### SPICE kernels
Place kernels under `src/orrery_app/data/kernels` (or point the UI to another folder). At minimum:
- leap seconds (`.tls`)
- planetary constants (`.tpc`)
- an SPK ephemeris (example: JPL DE441) for planets and dwarf planets

Moons require additional satellite SPK kernels beyond the planetary ephemeris.

### Textures & attribution
Textures live under `src/orrery_app/data/textures`. Download them with:

```powershell
./scripts/download_textures.ps1
```

The viewer will use the highest-resolution texture it finds for each body and fall back to lower-resolution maps (including 2k versions for dithering).
The 21600×10800 Earth base map is included as a source asset but is not loaded at runtime.

Sources and credits:
- Solar System Scope textures for the Sun and planetary base maps (plus Earth clouds and dwarf planet artist maps). License: CC BY 4.0. Source and license: https://www.solarsystemscope.com/textures/ and https://creativecommons.org/licenses/by/4.0/
- NASA Earth Observatory / Blue Marble Next Generation base map (January 2004) stored as `src/orrery_app/data/textures/hires/earth_21600.jpg`. Source: https://science.nasa.gov/resource/blue-marble-next-generation/ (NASA imagery is generally public domain; do not imply NASA endorsement)
- NASA/JPL Solar System Simulator texture maps for moons and Pluto/Charon (Voyager mosaics and composites). Source: https://maps.jpl.nasa.gov/tmaps/

Notes on realism:
- Venus surface uses Magellan radar-based imagery (not true color).
- Gas/ice giant maps are representative composites; Neptune in particular is not true-color ground truth.
- Pluto/Charon and Titan maps from JPL are marked “fictitious.”
- Dwarf planet maps from Solar System Scope are artistic impressions where no global imaging exists.
