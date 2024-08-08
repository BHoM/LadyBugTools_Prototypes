"""Unit test package for masterplanenergydemand."""

# pylint: disable=E0401
import warnings
from copy import deepcopy
from pathlib import Path

import pandas as pd

from masterplanenergydemand.typology_OLD import (BuildingType, Fabric, Form,
                                                 Model, Program, ProgramType,
                                                 System, TerrainType, Typology,
                                                 Vintage)

from . import EPW_OBJ, EPW_PATH, EXCEL_PATH, SIMULATION_DIRECTORY, TYPOLOGY_ID

# pylint: enable=E0401


def test_init():
    """_"""
    assert isinstance(
        Typology(
            identifier=TYPOLOGY_ID,
            total_area=1000,
            epw=EPW_PATH,
            building_type=BuildingType.EDUCATION_SCHOOL_PRIMARY,
            vintage=Vintage.ASHRAE_901_2010,
            form=Form.random(),
            fabric=Fabric.random(),
            program=Program.random(),
            system=System.random(),
        ),
        Typology,
    )

def test_random():
    """_"""
    assert isinstance(Typology.random(), Typology)

def test_equal():
    """_"""
    typology1 = Typology.random()
    typology2 = deepcopy(typology1)
    assert typology1 == typology2

def test_roundtrip():
    """_"""
    Typology.parse_obj(Typology.random().dict())

def test_model():
    """_"""
    assert isinstance(Typology.random().model(epw=EPW_OBJ), Model)

def test_program_type():
    """_"""
    assert isinstance(Typology.random().program_type, ProgramType)

def test_profile_table():
    """_"""
    assert isinstance(Typology.random().profile_table(), pd.DataFrame)

def test_number_of_buildings():
    """_"""
    typology = Typology.random()
    typology.total_area = 1000
    typology.form.average_footprint_area = 100
    typology.form.average_num_floors = 1
    assert typology.number_of_buildings == typology.total_area / (
        typology.form.average_footprint_area * typology.form.average_num_floors
    )

def test_occupancy_schedule():
    """_"""
    assert isinstance(Typology.random().occupancy_schedule(), pd.Series)

def test_estimate_lift_energy():
    """_"""
    assert isinstance(Typology.random().estimate_lift_energy(), pd.Series)

def test_occupants():
    """_"""
    for bt in BuildingType:
        typology = Typology.random()
        typology.building_type = bt
        assert isinstance(typology.occupants, pd.Series)

def test_simulate():
    """_"""
    typology = Typology.random()

    sql_path = typology.simulate(epw=EPW_OBJ, directory=SIMULATION_DIRECTORY)
    assert Path(sql_path).exists()

def test_results():
    """_"""
    typology = Typology.random()

    results = typology.load_results(epw=EPW_OBJ, directory=SIMULATION_DIRECTORY)
    assert isinstance(results, pd.DataFrame)

def test_parse_obj_extended():
    """_"""

    # load excel file containing configuration
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        df = pd.read_excel(EXCEL_PATH, sheet_name="TestMPED", engine="openpyxl", header=None, index_col=0)

    # get columns containing typology data
    for n, (_, s) in enumerate(df.items()):
        if n == 0:
            continue
        d = s.to_dict()

        # try to convert to typology
        typology = Typology.parse_obj_extended(d)

        assert isinstance(typology, Typology)

def test_from_building_type():
    """_"""
    for bt in BuildingType:
        assert isinstance(Typology.from_building_type(building_type=bt, total_area=100, epw=EPW_OBJ), Typology)
