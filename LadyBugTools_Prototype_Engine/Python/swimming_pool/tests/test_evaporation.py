import warnings

import pytest
from ladybug.epw import EPW

from ..evaporation_convection import (
    evaporation_gain_bensmallwood,
    evaporation_gain_jamesramsden,
    evaporation_gain,
    evaporation_gain_woolley,
    evaporation_rate_bensmallwood,
    evaporation_rate_jamesramsden,
    evaporation_rate,
    vapor_pressure_antoine,
)

EPW_FILE = r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\tests\assets\example.epw"
EPW_OBJ = EPW(EPW_FILE)


def test_vapor_pressure_antoine() -> None:
    """_"""
    assert vapor_pressure_antoine(0) == pytest.approx(605.5786273867391, 0.0001)


def test_evaporation_rate_jamesramsden() -> None:
    """_"""
    assert evaporation_rate_jamesramsden(EPW_OBJ).sum() == pytest.approx(
        1250.8199689576, 0.0001
    )


def test_evaporation_rate_bensmallwood() -> None:
    """_"""
    assert evaporation_rate_bensmallwood(EPW_OBJ).sum() == pytest.approx(
        1246.9686090852313, 0.0001
    )


def test_evaporation_rate_penman() -> None:
    """_"""
    assert evaporation_rate(EPW_OBJ).sum() == pytest.approx(1199.4298578206578, 0.0001)


def test_evaporation_gain_bensmallwood() -> None:
    """_"""
    warnings.warn("This might be funky - sort out before commiting!")
    assert evaporation_gain_bensmallwood(
        epw=EPW_OBJ, surface_area=10
    ).sum() == pytest.approx(-28144081.507053673, 0.0001)


def test_evaporation_gain_woolley() -> None:
    """_"""
    warnings.warn("This might be funky - sort out before commiting!")
    assert evaporation_gain_woolley(
        epw=EPW_OBJ, surface_area=10, water_temperature=10
    ).sum() == pytest.approx(-51299104912.083176, 0.0001)


def test_evaporation_gain_mancic() -> None:
    """_"""
    warnings.warn("This might be funky - sort out before commiting!")
    assert evaporation_gain(epw=EPW_OBJ, surface_area=10).sum() == pytest.approx(
        -27107114.78674687, 0.0001
    )


def test_evaporation_gain_jamesramsden() -> None:
    """_"""
    warnings.warn("This might be funky - sort out before commiting!")
    assert evaporation_gain_jamesramsden(
        epw=EPW_OBJ,
        surface_area=10,
    ).sum() == pytest.approx(-30695122.0382195, 0.0001)
