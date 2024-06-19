"""Unit test package for masterplanenergydemand."""

# pylint: disable=E0401
from masterplanenergydemand.enums import (BuildingType, ConstructionSet,
                                          ConstructionType, ProgramType,
                                          TerrainType, Vintage,
                                          default_construction_type,
                                          default_constructionset,
                                          default_context_shade_distance,
                                          default_floor_height,
                                          default_footprint_area, default_gfa,
                                          default_glazing_ratio,
                                          default_number_of_floors,
                                          default_program)

from . import EPW_OBJ

# pylint: enable=E0401


def test_default_construction_type():
    """_"""
    for building_type in BuildingType:
        assert isinstance(default_construction_type(building_type), ConstructionType)


def test_default_context_shade_distance():
    """_"""
    for terrain_type in TerrainType:
        assert isinstance(default_context_shade_distance(terrain_type), float)


def test_default_floor_height():
    """_"""
    for building_type in BuildingType:
        assert isinstance(default_floor_height(building_type), float)


def test_default_footprint_area():
    """_"""
    for building_type in BuildingType:
        assert isinstance(default_footprint_area(building_type), float)


def test_default_gfa():
    """_"""
    for building_type in BuildingType:
        assert isinstance(default_gfa(building_type), float)


def test_default_glazing_ratio():
    """_"""
    for building_type in BuildingType:
        assert isinstance(default_glazing_ratio(building_type), float)


def test_default_number_of_floors():
    """_"""
    for building_type in BuildingType:
        assert isinstance(default_number_of_floors(building_type), int)


def test_default_program():
    """_"""
    for building_type in BuildingType:
        assert isinstance(default_program(building_type), ProgramType)


def test_default_constructionset():
    """_"""
    for vintage in Vintage:
        for construction_type in ConstructionType:
            assert isinstance(
                default_constructionset(
                    vintage=vintage, construction_type=construction_type, epw=EPW_OBJ
                ),
                ConstructionSet,
            )
