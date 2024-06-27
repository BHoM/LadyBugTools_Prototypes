"""Unit test package for masterplanenergydemand."""

# pylint: disable=E0401
import warnings
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from masterplanenergydemand.enums import TerrainType
from masterplanenergydemand.mped import Masterplan
from masterplanenergydemand.typology import (BuildingType, Fabric, Form, Model,
                                             Program, ProgramType, System,
                                             Typology, Vintage)

from . import (EPW_OBJ, EPW_PATH, EXCEL_PATH, MPED_ID, SIMULATION_DIRECTORY,
               TYPOLOGY_ID)

# pylint: enable=E0401

def test_random():
    """_"""
    assert isinstance(Masterplan.random(), Masterplan)

def test_equal():
    """_"""
    mped1 = Masterplan.random()
    mped2 = deepcopy(mped1)
    assert mped1 == mped2

def test_roundtrip():
    """_"""
    Masterplan.parse_obj(Masterplan.random().dict())

def test_from_excel():
    """_"""
    assert isinstance(Masterplan.from_excel(EXCEL_PATH, MPED_ID), Masterplan)

def test_results():
    """_"""
    mped = Masterplan.from_excel(EXCEL_PATH, MPED_ID)
    assert isinstance(mped.results(), pd.DataFrame)

def test_sensitivity():
    """_"""
    n_typologies = 3

    mped = Masterplan.random(n_typologies=n_typologies)
    mped.identifier = MPED_ID

    for i in range(n_typologies):
        mped.typologies[i].identifier = f"{TYPOLOGY_ID}_{i}"
        mped.typologies[i].building_type = BuildingType.COMMERCIAL_OFFICE_MEDIUM

    # 2 * people gain in typology 1
    mped.typologies[1].program.occupant_density = mped.typologies[0].program.occupant_density * 2

    # 2 * area in typology 2
    mped.typologies[2].total_area = mped.typologies[0].total_area * 2

    df = mped.results()
    
    p_0, p_1, _ = df.iloc[:, df.columns.get_level_values(1) == "People (kWh)"].mean().values
    assert p_0 * 2 == pytest.approx(p_1, 10)

    c_0, _, c_2 = df.iloc[:, df.columns.get_level_values(1) == "Cooling (kWh)"].mean().values
    assert c_0 * 2 == pytest.approx(c_2, 10)
