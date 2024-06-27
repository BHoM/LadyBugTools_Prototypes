"""Unit test package for masterplanenergydemand."""

# pylint: disable=E0401
from copy import deepcopy

from masterplanenergydemand.fabric import BuildingType, Fabric, Vintage

from . import EPW_OBJ

# pylint: enable=E0401


def test_init():
    """_"""
    assert isinstance(
        Fabric(
            wall_u_value=[1, 1, 1, 1, 1, 1, 1, 1],
            wall_sri=45,
            floor_u_value=1,
            roof_u_value=1,
            window_u_value=[1, 1, 1, 1, 1, 1, 1, 1],
            window_shgc=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            skylight_u_value=1,
            skylight_shgc=0.5,
            roof_sri=0.7,
        ),
        Fabric,
    )


def test_random():
    """_"""
    assert isinstance(Fabric.random(), Fabric)


def test_equal():
    """_"""
    fabric1 = Fabric.random()
    fabric2 = deepcopy(fabric1)
    assert fabric1 == fabric2


def test_roundtrip():
    """_"""
    fabric = Fabric.random()

    # default
    assert isinstance(Fabric.parse_obj(fabric.dict()), Fabric)

    # extended
    d = fabric.dict()
    for n, k in enumerate(
        [
            "wall_u_value_N",
            "wall_u_value_NE",
            "wall_u_value_E",
            "wall_u_value_SE",
            "wall_u_value_S",
            "wall_u_value_SW",
            "wall_u_value_W",
            "wall_u_value_NW",
        ]
    ):
        d[k] = d["wall_u_value"][n]
    d.pop("wall_u_value")
    assert isinstance(Fabric.parse_obj_extended(d), Fabric)

def test_from_building_type():
    """_"""
    for bt in BuildingType:
        assert isinstance(Fabric.from_building_type(building_type=bt, epw=EPW_OBJ, vintage=Vintage.ASHRAE_901_2019), Fabric)
