import pytest

from ..makeup_water import supply_water_heating, water_temperature_at_depth


def test_supply_water_heating() -> None:
    """_"""
    assert supply_water_heating(
        water_volume_m3_hour=1, supply_temperature=10, target_temperature=11
    ) == pytest.approx(1162.7777777777778, 0.001)
    assert supply_water_heating(
        water_volume_m3_hour=1, supply_temperature=10, target_temperature=9
    ) == pytest.approx(-1162.7777777777778, 0.001)


def test_water_temperature_at_depth() -> None:
    """_"""
    assert water_temperature_at_depth(10, 0) == 10
    assert water_temperature_at_depth(10, 1) == 9.6
    assert water_temperature_at_depth(10, 23) == pytest.approx(8.737718019291965, 0.001)
