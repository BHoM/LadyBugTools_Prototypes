"""Unit test package for masterplanenergydemand."""

# pylint: disable=E0401
import pytest

from masterplanenergydemand.fabric import (BuildingType, ConstructionType,
                                           Fabric, Vintage)

# pylint: enable=E0401


def test_instantiation_default():
    """_"""
    assert isinstance(Fabric(), Fabric)

def test_instantiation_with_args_good():
    """_"""
    assert isinstance(
        Fabric(
            vintage=Vintage.ASHRAE_901_2019,
            construction_type=ConstructionType.MASS,
            wall_u_value=[1, 1, 1, 1, 1, 1, 1, 1],
            wall_sri=[35] * 8,
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

def test_instantiation_with_args_bad():
    """_"""
    # bad values for args (with good value at end to allow for passover)
    _vintage = ["A vintage", None]
    _construction_type = ["A construction type", None]
    _wall_u_value = [0.5, [0.5] * 7, ([0.5] * 7) + [1.5], ["0.5"] * 8, None]
    _wall_sri = [-0.1, 1000, "0.1", None]
    _floor_u_value = [-0.1, 1000, "0.1", None]
    _roof_u_value = [-0.1, 1000, "0.1", None]
    _window_u_value = [0.5, [0.5] * 7, ([0.5] * 7) + [1.5], ["0.5"] * 8, None]
    _window_shgc = [0.5, [0.5] * 7, ([0.5] * 7) + [1.5], ["0.5"] * 8, None]
    _skylight_u_value = [-0.1, 1000, "0.1", None]
    _skylight_shgc = [-0.1, 1000, "0.1", None]
    _roof_sri = [-0.1, 1000, "0.1", None]    

    for vtg, constr, wl_u, wl_sri, flr_u, rf_u, wdw_u, wdw_shgc, sk_u, sk_shgc, rf_sri in zip(*[_vintage, _construction_type, _wall_u_value, _wall_sri, _floor_u_value, _roof_u_value, _window_u_value, _window_shgc, _skylight_u_value, _skylight_shgc, _roof_sri]):
        if all(i is None for i in [vtg, constr, wl_u, wl_sri, flr_u, rf_u, wdw_u, wdw_shgc, sk_u, sk_shgc, rf_sri]):
            continue
        with pytest.raises(ValueError):
            Fabric(
                vintage=vtg,
                construction_type=constr,
                wall_u_value=wl_u,
                wall_sri=wl_sri,
                floor_u_value=flr_u,
                roof_u_value=rf_u,
                window_u_value=wdw_u,
                window_shgc=wdw_shgc,
                skylight_u_value=sk_u,
                skylight_shgc=sk_shgc,
                roof_sri=rf_sri,
            )

def test_equal():
    """_"""
    fabric1 = Fabric()
    fabric2 = Fabric()
    assert fabric1 == fabric2

def test_instantiation_from_defaults():
    """..."""
    for building_type in BuildingType:
        assert isinstance(Fabric.from_defaults(building_type), Fabric)

def test_to_from_dict():
    """_"""
    fabric = Fabric()
    fabric_dict = fabric.to_dict()
    assert isinstance(fabric_dict, dict)

    new_fabric = Fabric.from_dict(fabric_dict)
    assert isinstance(new_fabric, Fabric)

    assert fabric == new_fabric

# TODO - add methods testing here
