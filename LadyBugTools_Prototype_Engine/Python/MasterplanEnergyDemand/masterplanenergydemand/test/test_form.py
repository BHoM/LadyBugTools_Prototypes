"""Unit test package for masterplanenergydemand."""

# pylint: disable=E0401
import pytest

from masterplanenergydemand.form import (BuildingType, Form, Model, Polygon2D,
                                         TerrainType)

# pylint: enable=E0401


def test_instantiation_default():
    """_"""
    assert isinstance(Form(), Form)

def test_instantiation_with_args_good():
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

def test_instantiation_with_args_bad():
    """_"""
    # bad values for args (with good value at end to allow for passover)
    _average_footprint_area = [-10, 0, "100", None]
    _average_num_floors = [-10, 0, "2", None]
    _average_floor_height = [-10, 0, "3", None]
    _rotation = [-1, 361, "4", None]
    _terrain = ["A terrain", None]
    _glazing_ratio = [0.5, [0.5] * 7, ([0.5] * 7) + [1.5], ["0.5"] * 8, None]
    _skylight_ratio = [-0.1, 1.2, "0.1", None]
    for avg_fp_area, avg_n_flr, avg_flr_hgt, rot, ter, gr, sr in zip(*[_average_footprint_area, _average_num_floors, _average_floor_height, _rotation, _terrain, _glazing_ratio, _skylight_ratio]):
        if all(i is None for i in [avg_fp_area, avg_n_flr, avg_flr_hgt, rot, ter, gr, sr]):
            continue
        with pytest.raises(ValueError):
            Form(
                average_footprint_area=avg_fp_area,
                average_num_floors=avg_n_flr,
                average_floor_height=avg_flr_hgt,
                rotation=rot,
                terrain=ter,
                glazing_ratio=gr,
                skylight_ratio=sr,
            )

def test_equal():
    """_"""
    form1 = Form()
    form2 = Form()
    assert form1 == form2

def test_instantiation_from_defaults():
    """..."""
    for building_type in BuildingType:
        assert isinstance(Form.from_defaults(building_type), Form)

def test_to_from_dict():
    """_"""
    form = Form()
    form_dict = form.to_dict()
    assert isinstance(form_dict, dict)

    new_form = Form.from_dict(form_dict)
    assert isinstance(new_form, Form)

    assert form == new_form

def test_building_height():
    """_"""
    form = Form(average_num_floors=2, average_floor_height=3)
    assert form.building_height == 6

def test_footprint():
    """_"""
    form = Form()
    assert isinstance(form.footprint(), Polygon2D)

def test_base_model():
    """_"""
    form = Form(average_footprint_area=100, average_num_floors=2)
    model = form.base_model()
    assert isinstance(model, Model)
    assert model.floor_area == pytest.approx(200, rel=1e-2)
