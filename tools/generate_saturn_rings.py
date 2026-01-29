from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtGui import QColor, QImage


@dataclass(frozen=True)
class RingBand:
    name: str
    inner_km: float
    outer_km: float
    optical_depth: float
    albedo: float


RING_BANDS = [
    RingBand("D", 66900.0, 74491.0, 3.0e-4, 0.08),
    RingBand("C", 74491.0, 91975.0, 0.18, 0.20),
    RingBand("B", 91975.0, 117570.0, 2.0, 0.50),
    RingBand("Cassini", 117500.0, 122050.0, 0.05, 0.25),
    RingBand("A", 122050.0, 136770.0, 0.60, 0.50),
    RingBand("Roche", 136770.0, 139380.0, 1.0e-4, 0.10),
    RingBand("F", 139826.0, 140612.0, 0.08, 0.45),
    RingBand("G", 166000.0, 173200.0, 1.0e-6, 0.10),
]

GAPS = [
    ("Encke", 133423.0, 133745.0),
    ("Keeler", 136487.0, 136522.0),
]

RING_MIN_KM = RING_BANDS[0].inner_km
RING_MAX_KM = RING_BANDS[-1].outer_km


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if x <= edge0:
        return 0.0
    if x >= edge1:
        return 1.0
    t = (x - edge0) / (edge1 - edge0)
    return t * t * (3.0 - 2.0 * t)


def _ring_profile(radius_km: float) -> tuple[float, float]:
    edge_soft_km = 120.0
    for band in RING_BANDS:
        if band.inner_km <= radius_km <= band.outer_km:
            fade_in = _smoothstep(band.inner_km, band.inner_km + edge_soft_km, radius_km)
            fade_out = 1.0 - _smoothstep(band.outer_km - edge_soft_km, band.outer_km, radius_km)
            edge = min(fade_in, fade_out)
            tau = band.optical_depth * edge
            albedo = band.albedo
            for _name, g0, g1 in GAPS:
                if g0 <= radius_km <= g1:
                    tau *= 0.02
            return tau, albedo
    return 0.0, 0.0


def generate_texture(path: str, width: int = 2048, height: int = 512) -> None:
    image = QImage(width, height, QImage.Format.Format_RGBA8888)

    for y in range(height):
        t = y / (height - 1)
        radius_km = RING_MIN_KM + t * (RING_MAX_KM - RING_MIN_KM)
        tau, albedo = _ring_profile(radius_km)

        alpha = 1.0 - math.exp(-tau)
        alpha = min(1.0, max(0.0, alpha * 0.95))

        base = 0.55 + 0.55 * albedo
        base = max(0.15, min(0.95, base))
        r_base = 210 * base
        g_base = 195 * base
        b_base = 175 * base

        for x in range(width):
            az = (x / width) * 2.0 * math.pi
            ripple = 1.0 + 0.035 * math.sin(az * 8.0 + t * 14.0)
            rr = max(0, min(255, int(r_base * ripple)))
            gg = max(0, min(255, int(g_base * ripple)))
            bb = max(0, min(255, int(b_base * ripple)))
            aa = max(0, min(255, int(255 * alpha)))
            image.setPixelColor(x, y, QColor(rr, gg, bb, aa))

    image.save(path)


if __name__ == "__main__":
    output = (
        "c:\\Users\\brodi\\Desktop\\Cloned Repos\\Orrery_crafting\\"
        "src\\orrery_app\\data\\textures\\extras\\saturn_rings.png"
    )
    generate_texture(output)
