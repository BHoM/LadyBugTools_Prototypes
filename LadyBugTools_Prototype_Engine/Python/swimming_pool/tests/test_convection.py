import warnings

import pytest
from ladybug.epw import EPW

from ..convection import convection_gain_bowen_ratio, convection_gain_czarnecki

EPW_FILE = r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\tests\assets\example.epw"
EPW_OBJ = EPW(EPW_FILE)


def test_convection_gain_czarnecki() -> None:
    """_"""
    warnings.warn("This might be funky - sort out before commiting!")
    assert convection_gain_czarnecki(
        surface_area=10, epw=EPW_OBJ, water_temperature=10
    ).sum() == pytest.approx(-790505.454272734, 0.001)


def test_convection_gain_bowen_ratio() -> None:
    """_"""
    warnings.warn("This might be funky - sort out before commiting!")
    assert convection_gain_bowen_ratio(
        surface_area=10, epw=EPW_OBJ, water_temperature=10
    ).sum() == pytest.approx(880108.0630333215, 0.001)
