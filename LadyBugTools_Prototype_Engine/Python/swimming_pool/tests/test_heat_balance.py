import pytest
from ladybug.epw import EPW
from matplotlib import pyplot as plt

from ..heat_balance import heat_balance, plot_monthly_balance, plot_timeseries

EPW_FILE = r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\tests\assets\example.epw"
EPW_OBJ = EPW(EPW_FILE)


def test_heat_balance() -> None:
    """_"""
    assert heat_balance(
        epw=EPW_OBJ,
        water_surface_area=1,
        average_depth=1,
        target_water_temperature=20,
    ).values.sum() == pytest.approx(-2924748.028092618, 0.001)


def test_plot_timeseries() -> None:
    """_"""
    hb_df = heat_balance(
        epw=EPW_OBJ,
        water_surface_area=1,
        average_depth=1,
        target_water_temperature=20,
    )
    assert isinstance(plot_timeseries(hb_df), plt.Axes)


def test_plot_monthly_balance() -> None:
    """_"""
    hb_df = heat_balance(
        epw=EPW_OBJ,
        water_surface_area=1,
        average_depth=1,
        target_water_temperature=20,
    )
    assert isinstance(plot_monthly_balance(hb_df), plt.Axes)
