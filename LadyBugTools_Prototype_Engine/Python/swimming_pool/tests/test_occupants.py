import pytest

from ..occupants import (
    occupant_gain,
    occupant_gain_from_density,
    occupant_gain_from_number,
)


def test_occupant_gain_from_number() -> None:
    """_"""
    assert occupant_gain_from_number(number_of_people=1, gain_per_person=100) == 100


def test_occupant_gain_from_density() -> None:
    """_"""
    assert (
        occupant_gain_from_density(m2_per_person=1, surface_area=1, gain_per_person=100)
        == 100
    )


def test_occupant_gain() -> None:
    """_"""
    assert occupant_gain(number_of_people=1, gain_per_person=100) == 100


def test_occupant_gain_errors():
    """_"""
    with pytest.raises(ValueError):
        occupant_gain(number_of_people=5, m2_per_person=2)

    with pytest.raises(ValueError):
        occupant_gain(number_of_people=5, surface_area=100, m2_per_person=2)

    with pytest.raises(ValueError):
        occupant_gain(m2_per_person=2)

    with pytest.raises(ValueError):
        occupant_gain(surface_area=100)

    with pytest.raises(ValueError):
        occupant_gain()
