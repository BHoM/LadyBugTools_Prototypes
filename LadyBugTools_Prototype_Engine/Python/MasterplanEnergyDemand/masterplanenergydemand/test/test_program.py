
import pandas as pd

from masterplanenergydemand.program import BuildingType, Program, ProgramType

from . import EPW_OBJ


def test_from_building_type():
    """_"""
    for bt in BuildingType:
        assert isinstance(Program.from_building_type(building_type=bt), Program)

def test_init():
    """_"""
    prog = Program(
        occupant_density=12,
        lighting_power_density=13,
        equipment_power_density=11,
        infiltration_rate=0.0023,
        ventilation_rate=0.234,
        heating_setpoint=30,
        heating_setback=20,
        cooling_setpoint=35,
        cooling_setback=200,
        humidifying_setpoint=49,
        humidifying_setback=48,
        dehumidifying_setpoint=51,
        dehumidifying_setback=52,
    )
    assert isinstance(prog, Program)
    assert prog.occupant_density == 12
    assert prog.lighting_power_density == 13
    assert prog.equipment_power_density == 11
    assert prog.infiltration_rate == 0.0023
    assert prog.ventilation_rate == 0.234
    assert prog.heating_setpoint == 30
    assert prog.heating_setback == 20
    assert prog.cooling_setpoint == 35
    assert prog.cooling_setback == 200
    assert prog.humidifying_setpoint == 49
    assert prog.humidifying_setback == 48
    assert prog.dehumidifying_setpoint == 51
    assert prog.dehumidifying_setback == 52

def test_random():
    """_"""
    assert isinstance(Program.random(), Program)

def test_roundtrip():
    """_"""
    Program.parse_obj(Program.random().dict())

def test_program_type():
    """_"""
    for bt in BuildingType:
        assert isinstance(Program.from_building_type(bt).program_type(bt), ProgramType)

def test_profiles():
    """_"""
    prog = Program.random()
    for method in [
        prog.profile_lighting,
        prog.profile_equipment,
        prog.profile_infiltration,
        prog.profile_ventilation,
        prog.profile_people,
        prog.profile_heating,
        prog.profile_cooling,
        prog.profile_humidifying,
        prog.profile_dehumidifying,
        prog.profile_shw,
    ]:
        for bt in BuildingType:
            assert isinstance(method(bt), pd.Series)

def test_parse_obj_extended():
    """_"""

    assert isinstance(Program.parse_obj_extended({}), Program)

