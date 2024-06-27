"""Unit test package for masterplanenergydemand."""

# pylint: disable=E0401
from copy import deepcopy

from masterplanenergydemand.system import (BuildingType, EconomizerType,
                                           IdealAirSystem, SHWSystem, System,
                                           Vintage)

from . import EPW_OBJ

# pylint: enable=E0401

def test_init():
    """_"""
    assert isinstance(
        System(
            economizer_type=EconomizerType.NO_ECONOMIZER,
            sensible_heat_recovery_effectiveness=0,
            latent_heat_recovery_effectiveness=0,
            demand_controlled_ventilation=False,
            daylight_dimming=False,
            heating_cop=1,
            cooling_eer=1,
            fan_power=1.8,
            pump_power=0.35,
    ), System)

def test_random():
    """_"""
    assert isinstance(System.random(), System)

def test_equal():
    """_"""
    system1 = System.random()
    system2 = deepcopy(system1)
    assert system1 == system2

def test_ideal_air():
    """_"""
    system = System.random()
    assert isinstance(system.ideal_air(), IdealAirSystem)

def test_shw():
    """_"""
    system = System.random()
    assert isinstance(system.shw(), SHWSystem)

def test_roundtrip():
    """_"""
    System.parse_obj(System.random().dict())

def test_from_building_type():
    """_"""
    for bt in BuildingType:
        assert isinstance(System.from_building_type(building_type=bt, epw=EPW_OBJ, vintage=Vintage.ASHRAE_901_2019), System)
