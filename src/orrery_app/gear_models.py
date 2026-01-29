from __future__ import annotations

from dataclasses import dataclass

Color = tuple[float, float, float]


@dataclass
class GearRenderSpec:
    gear_id: str
    teeth: int
    radius_mm: float
    thickness_mm: float
    center_mm: tuple[float, float, float]
    color: Color
    label: str
    shaft_index: int
    speed_rps: float
    bore_radius_mm: float = 0.0
    angle_rad: float = 0.0
    is_drive: bool = False


@dataclass
class ArmRenderSpec:
    arm_id: str
    length_mm: float
    angle_rad: float
    origin_mm: tuple[float, float, float]
    tube_base_y_mm: float
    speed_rps: float
    color: Color
    label: str = ""
