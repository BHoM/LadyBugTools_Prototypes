"""Unit test package for masterplanenergydemand."""

# pylint: disable=E0401
from copy import deepcopy

import pytest

from masterplanenergydemand.form import (BuildingType, Form, Model, Polygon2D,
                                         TerrainType)

# pylint: enable=E0401


def test_init():
    """_"""
    assert isinstance(
        Form(
            average_footprint_area=100,
            average_num_floors=2,
            average_floor_height=3,
            rotation=4,
            terrain=TerrainType.COUNTRY,
            glazing_ratio=[0.5] * 8,
            skylight_ratio=0.1,
        ),
        Form,
    )

def test_random():
    """_"""
    assert isinstance(Form.random(), Form)

def test_equal():
    """_"""
    form1 = Form.random()
    form2 = deepcopy(form1)
    assert form1 == form2

def test_from_building_type():
    """..."""
    for building_type in BuildingType:
        assert isinstance(Form.from_building_type(building_type=building_type, rotation=0, terrain=TerrainType.URBAN), Form)

def test_building_height():
    """_"""
    form = Form.random()
    assert form.average_num_floors * form.average_floor_height == form.building_height()

def test_footprint():
    """_"""
    form = Form.random()
    assert isinstance(form.footprint(), Polygon2D)

def test_base_model():
    """_"""
    form = Form.random()
    form.average_footprint_area = 100
    form.average_num_floors = 2
    model = form.base_model()
    assert isinstance(model, Model)
    assert model.floor_area == pytest.approx(200, rel=1e-2)

def test_roundtrip():
    """_"""
    Form.parse_obj(Form.random().dict())
