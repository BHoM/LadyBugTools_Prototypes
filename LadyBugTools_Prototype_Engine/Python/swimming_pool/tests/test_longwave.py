import pytest

from ..longwave import longwave_gain


def test_longwave_gain() -> None:
    """_"""

    assert longwave_gain(
        surface_area=1, sky_temperature=10, water_temperature=0, water_emissivity=0.95
    ) == pytest.approx(46.384495617585785, 0.001)
    assert longwave_gain(
        surface_area=1, sky_temperature=-40, water_temperature=0, water_emissivity=0.95
    ) == pytest.approx(-140.6991385829005, 0.001)
    assert longwave_gain(
        surface_area=1, sky_temperature=12, water_temperature=15, water_emissivity=0.1
    ) == pytest.approx(-1.60273128022928, 0.001)
