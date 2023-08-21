import pytest

from ..conduction import (
    conduction_gain_interface_area,
    conduction_gain_shape_factor,
    ground_interface_area_cylinder,
    ground_interface_area_prism,
    pool_characteristic_length,
    pool_dimensionless_conduction_heat_rate,
    pool_shape_factor,
    edge_length_from_aspect_ratio,
)


def test_pool_width_length() -> None:
    """_"""

    width, length = edge_length_from_aspect_ratio(12)
    assert width == pytest.approx(2.723308256380237, 0.0001)
    assert length == pytest.approx(4.406405324070858, 0.0001)

    width, length = edge_length_from_aspect_ratio(12, 3)
    assert width == pytest.approx(2, 0.1)
    assert length == pytest.approx(6, 0.1)

    with pytest.raises(TypeError):
        edge_length_from_aspect_ratio("12")


def test_ground_interface_area_prism() -> None:
    """_"""

    assert ground_interface_area_prism(
        surface_area=10, average_depth=1
    ) == pytest.approx(23.01701652187996, 0.001)


def test_ground_interface_area_cylinder() -> None:
    """_"""

    assert ground_interface_area_cylinder(
        surface_area=10, average_depth=1
    ) == pytest.approx(21.209982432795858, 0.001)


def test_pool_shape_factor() -> None:
    """_"""

    assert pool_shape_factor(surface_area=10, average_depth=1) == pytest.approx(
        0.6254320210036949, 0.001
    )


def test_pool_characteristic_length() -> None:
    """_"""

    assert pool_characteristic_length(surface_area=10) == pytest.approx(
        0.8920620580763856, 0.001
    )


def test_pool_dimensionless_conduction_heat_rate() -> None:
    """_"""

    assert pool_dimensionless_conduction_heat_rate(
        pool_shape_factor(surface_area=10, average_depth=1)
    ) == pytest.approx(0.95156787, 0.001)


def test_conduction_gain_interface_area() -> None:
    """_"""

    assert conduction_gain_interface_area(
        surface_area=10,
        average_depth=10,
        interface_u_value=0.5,
        soil_temperature=10,
        water_temperature=12,
    ) == pytest.approx(-140.1701652187996, 0.001)


def test_conduction_gain_shape_factor() -> None:
    """_"""

    assert conduction_gain_shape_factor(
        surface_area=10,
        average_depth=1,
        soil_temperature=10,
        water_temperature=12,
        soil_conductivity=0.333,
    ) == pytest.approx(-0.5653391560239911, 0.001)
