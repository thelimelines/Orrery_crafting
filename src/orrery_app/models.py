from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

Color = tuple[float, float, float]


@dataclass(frozen=True)
class BodyInfo:
    name: str
    spice_name: str
    parent: str
    color: Color
    radius_km: float
    mean_distance_km: float | None = None
    orbital_period_days: float | None = None


@dataclass
class BodySettings:
    distance_scale: float = 1.0
    radius_scale: float = 1.0


@dataclass
class BodyState:
    position_km: tuple[float, float, float]
    available: bool
    error: str | None = None


@dataclass(frozen=True)
class OrbitStats:
    min_km: float
    max_km: float
    max_normal_km: float = 0.0


BODY_CATALOG: tuple[BodyInfo, ...] = (
    BodyInfo(
        name="Sun",
        spice_name="SUN",
        parent="SUN",
        color=(0.98, 0.86, 0.35),
        radius_km=695_700.0,
        mean_distance_km=None,
        orbital_period_days=None,
    ),
    BodyInfo(
        name="Mercury",
        spice_name="MERCURY",
        parent="SUN",
        color=(0.75, 0.75, 0.75),
        radius_km=2439.7,
        mean_distance_km=57_909_227.0,
        orbital_period_days=87.969,
    ),
    BodyInfo(
        name="Venus",
        spice_name="VENUS",
        parent="SUN",
        color=(0.9, 0.8, 0.6),
        radius_km=6051.8,
        mean_distance_km=108_209_475.0,
        orbital_period_days=224.701,
    ),
    BodyInfo(
        name="Earth",
        spice_name="EARTH",
        parent="SUN",
        color=(0.3, 0.5, 0.95),
        radius_km=6371.0,
        mean_distance_km=149_598_262.0,
        orbital_period_days=365.256,
    ),
    BodyInfo(
        name="Moon",
        spice_name="MOON",
        parent="EARTH",
        color=(0.8, 0.8, 0.85),
        radius_km=1737.4,
        mean_distance_km=384_400.0,
        orbital_period_days=27.321,
    ),
    BodyInfo(
        name="Mars",
        spice_name="MARS",
        parent="SUN",
        color=(0.9, 0.4, 0.3),
        radius_km=3389.5,
        mean_distance_km=227_943_824.0,
        orbital_period_days=686.980,
    ),
    BodyInfo(
        name="Phobos",
        spice_name="PHOBOS",
        parent="MARS",
        color=(0.65, 0.6, 0.55),
        radius_km=11.1,
        mean_distance_km=9_376.0,
        orbital_period_days=0.319,
    ),
    BodyInfo(
        name="Deimos",
        spice_name="DEIMOS",
        parent="MARS",
        color=(0.7, 0.65, 0.6),
        radius_km=6.2,
        mean_distance_km=23_463.0,
        orbital_period_days=1.263,
    ),
    BodyInfo(
        name="Jupiter",
        spice_name="JUPITER",
        parent="SUN",
        color=(0.9, 0.7, 0.5),
        radius_km=69_911.0,
        mean_distance_km=778_340_821.0,
        orbital_period_days=4332.59,
    ),
    BodyInfo(
        name="Io",
        spice_name="IO",
        parent="JUPITER",
        color=(0.9, 0.8, 0.4),
        radius_km=1821.6,
        mean_distance_km=421_700.0,
        orbital_period_days=1.769,
    ),
    BodyInfo(
        name="Europa",
        spice_name="EUROPA",
        parent="JUPITER",
        color=(0.85, 0.85, 0.9),
        radius_km=1560.8,
        mean_distance_km=671_034.0,
        orbital_period_days=3.551,
    ),
    BodyInfo(
        name="Ganymede",
        spice_name="GANYMEDE",
        parent="JUPITER",
        color=(0.7, 0.7, 0.75),
        radius_km=2634.1,
        mean_distance_km=1_070_412.0,
        orbital_period_days=7.155,
    ),
    BodyInfo(
        name="Callisto",
        spice_name="CALLISTO",
        parent="JUPITER",
        color=(0.6, 0.55, 0.5),
        radius_km=2410.3,
        mean_distance_km=1_882_709.0,
        orbital_period_days=16.689,
    ),
    BodyInfo(
        name="Saturn",
        spice_name="SATURN",
        parent="SUN",
        color=(0.95, 0.85, 0.6),
        radius_km=58_232.0,
        mean_distance_km=1_426_666_422.0,
        orbital_period_days=10_759.22,
    ),
    BodyInfo(
        name="Titan",
        spice_name="TITAN",
        parent="SATURN",
        color=(0.9, 0.7, 0.5),
        radius_km=2574.7,
        mean_distance_km=1_221_870.0,
        orbital_period_days=15.945,
    ),
    BodyInfo(
        name="Enceladus",
        spice_name="ENCELADUS",
        parent="SATURN",
        color=(0.85, 0.9, 0.95),
        radius_km=252.1,
        mean_distance_km=237_948.0,
        orbital_period_days=1.370,
    ),
    BodyInfo(
        name="Rhea",
        spice_name="RHEA",
        parent="SATURN",
        color=(0.75, 0.75, 0.8),
        radius_km=763.8,
        mean_distance_km=527_108.0,
        orbital_period_days=4.518,
    ),
    BodyInfo(
        name="Iapetus",
        spice_name="IAPETUS",
        parent="SATURN",
        color=(0.6, 0.6, 0.65),
        radius_km=734.5,
        mean_distance_km=3_560_820.0,
        orbital_period_days=79.321,
    ),
    BodyInfo(
        name="Dione",
        spice_name="DIONE",
        parent="SATURN",
        color=(0.7, 0.7, 0.75),
        radius_km=561.4,
        mean_distance_km=377_396.0,
        orbital_period_days=2.737,
    ),
    BodyInfo(
        name="Tethys",
        spice_name="TETHYS",
        parent="SATURN",
        color=(0.8, 0.8, 0.85),
        radius_km=531.1,
        mean_distance_km=294_619.0,
        orbital_period_days=1.888,
    ),
    BodyInfo(
        name="Mimas",
        spice_name="MIMAS",
        parent="SATURN",
        color=(0.75, 0.75, 0.8),
        radius_km=198.2,
        mean_distance_km=185_539.0,
        orbital_period_days=0.942,
    ),
    BodyInfo(
        name="Uranus",
        spice_name="URANUS",
        parent="SUN",
        color=(0.65, 0.85, 0.9),
        radius_km=25_362.0,
        mean_distance_km=2_870_658_186.0,
        orbital_period_days=30_688.5,
    ),
    BodyInfo(
        name="Miranda",
        spice_name="MIRANDA",
        parent="URANUS",
        color=(0.7, 0.7, 0.75),
        radius_km=235.8,
        mean_distance_km=129_390.0,
        orbital_period_days=1.414,
    ),
    BodyInfo(
        name="Ariel",
        spice_name="ARIEL",
        parent="URANUS",
        color=(0.75, 0.75, 0.8),
        radius_km=578.9,
        mean_distance_km=190_900.0,
        orbital_period_days=2.520,
    ),
    BodyInfo(
        name="Umbriel",
        spice_name="UMBRIEL",
        parent="URANUS",
        color=(0.6, 0.6, 0.65),
        radius_km=584.7,
        mean_distance_km=266_000.0,
        orbital_period_days=4.144,
    ),
    BodyInfo(
        name="Titania",
        spice_name="TITANIA",
        parent="URANUS",
        color=(0.65, 0.65, 0.7),
        radius_km=788.9,
        mean_distance_km=436_300.0,
        orbital_period_days=8.706,
    ),
    BodyInfo(
        name="Oberon",
        spice_name="OBERON",
        parent="URANUS",
        color=(0.6, 0.6, 0.65),
        radius_km=761.4,
        mean_distance_km=583_500.0,
        orbital_period_days=13.463,
    ),
    BodyInfo(
        name="Neptune",
        spice_name="NEPTUNE",
        parent="SUN",
        color=(0.3, 0.45, 0.9),
        radius_km=24_622.0,
        mean_distance_km=4_498_396_441.0,
        orbital_period_days=60_195.0,
    ),
    BodyInfo(
        name="Triton",
        spice_name="TRITON",
        parent="NEPTUNE",
        color=(0.75, 0.8, 0.9),
        radius_km=1353.4,
        mean_distance_km=354_759.0,
        orbital_period_days=5.877,
    ),
    BodyInfo(
        name="Pluto",
        spice_name="PLUTO",
        parent="SUN",
        color=(0.7, 0.65, 0.6),
        radius_km=1188.3,
        mean_distance_km=5_906_376_272.0,
        orbital_period_days=90_560.0,
    ),
    BodyInfo(
        name="Charon",
        spice_name="CHARON",
        parent="PLUTO",
        color=(0.7, 0.7, 0.75),
        radius_km=606.0,
        mean_distance_km=19_596.0,
        orbital_period_days=6.387,
    ),
    BodyInfo(
        name="Ceres",
        spice_name="20000001",
        parent="SUN",
        color=(0.7, 0.7, 0.7),
        radius_km=473.0,
        mean_distance_km=413_700_000.0,
        orbital_period_days=1680.0,
    ),
    BodyInfo(
        name="Eris",
        spice_name="20136199",
        parent="SUN",
        color=(0.8, 0.8, 0.85),
        radius_km=1163.0,
        mean_distance_km=None,
        orbital_period_days=None,
    ),
    BodyInfo(
        name="Haumea",
        spice_name="20136108",
        parent="SUN",
        color=(0.75, 0.75, 0.8),
        radius_km=620.0,
        mean_distance_km=None,
        orbital_period_days=None,
    ),
    BodyInfo(
        name="Makemake",
        spice_name="20136472",
        parent="SUN",
        color=(0.8, 0.75, 0.7),
        radius_km=715.0,
        mean_distance_km=None,
        orbital_period_days=None,
    ),
)


BODY_BY_NAME: dict[str, BodyInfo] = {body.name: body for body in BODY_CATALOG}


def iter_body_names() -> Iterable[str]:
    return (body.name for body in BODY_CATALOG)
