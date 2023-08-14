import pytest

from ..makeup_water import supply_water_heating


def test_supply_water_heating() -> None:
    """_"""
    assert supply_water_heating(
        water_volume_m3_hour=1, supply_temperature=10, target_temperature=11
    ) == pytest.approx(1162.7777777777778, 0.001)
    assert supply_water_heating(
        water_volume_m3_hour=1, supply_temperature=10, target_temperature=9
    ) == pytest.approx(-1162.7777777777778, 0.001)
