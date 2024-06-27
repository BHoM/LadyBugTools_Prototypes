# pylint: disable=E0401
import pytest
from ladybug_geometry.geometry3d import Face3D, Plane, Vector3D

from masterplanenergydemand.util import (angle_from_north, cardinality,
                                         estimate_sri_properties,
                                         face_orientation,
                                         list_of_nums_validator,
                                         number_validator)

# pylint: enable=E0401


def test_cardinality():
    """..."""
    # Test with valid inputs
    assert cardinality(90, 4) == "E"
    assert cardinality(45, 8) == "NE"
    assert cardinality(22.5, 16) == "NNE"
    assert cardinality(11.25, 32) == "NbE"

    # Test with direction_angle out of range
    with pytest.raises(ValueError):
        cardinality(370, 4)
    with pytest.raises(ValueError):
        cardinality(-10, 8)

    # Test with invalid directions
    with pytest.raises(ValueError):
        cardinality(90, 5)
    with pytest.raises(ValueError):
        cardinality(45, 10)


def test_angle_from_north():
    """..."""
    # Test with vectors pointing to cardinal directions
    assert angle_from_north(Vector3D(0, 1, 0)) == pytest.approx(0.0, 0.01)  # North
    assert angle_from_north(Vector3D(1, 0, 0)) == pytest.approx(90.0, 0.01)  # East
    assert angle_from_north(Vector3D(0, -1, 0)) == pytest.approx(180.0, 0.01)  # South
    assert angle_from_north(Vector3D(-1, 0, 0)) == pytest.approx(270.0, 0.01)  # West

    # Test with vectors pointing to intercardinal directions
    assert angle_from_north(Vector3D(1, 1, 0)) == pytest.approx(45.0, 0.01)  # Northeast
    assert angle_from_north(Vector3D(1, -1, 0)) == pytest.approx(
        135.0, 0.01
    )  # Southeast
    assert angle_from_north(Vector3D(-1, -1, 0)) == pytest.approx(
        225.0, 0.01
    )  # Southwest
    assert angle_from_north(Vector3D(-1, 1, 0)) == pytest.approx(
        315.0, 0.01
    )  # Northwest


def test_face_orientation():
    """..."""
    # Test with faces oriented towards cardinal directions
    north_face = Face3D.from_rectangle(1, 1, Plane(Vector3D(0, 1, 0)))
    assert face_orientation(north_face) == "N"

    east_face = Face3D.from_rectangle(1, 1, Plane(Vector3D(1, 0, 0)))
    assert face_orientation(east_face) == "E"

    south_face = Face3D.from_rectangle(1, 1, Plane(Vector3D(0, -1, 0)))
    assert face_orientation(south_face) == "S"

    west_face = Face3D.from_rectangle(1, 1, Plane(Vector3D(-1, 0, 0)))
    assert face_orientation(west_face) == "W"

    with pytest.raises(ValueError):
        face_orientation(Face3D.from_rectangle(1, 1, Plane(Vector3D(0, 0, 1))))
        face_orientation(Face3D.from_rectangle(1, 1, Plane(Vector3D(0, 0, -1))))


def test_number_validator():
    """..."""
    # Test with valid inputs
    number_validator(5, "Test", gt=3)
    number_validator(5, "Test", ge=5)
    number_validator(5, "Test", lt=7)
    number_validator(5, "Test", le=5)

    # Test with invalid inputs
    with pytest.raises(ValueError):
        number_validator("5", "Test", gt=3)
    with pytest.raises(ValueError):
        number_validator(5, "Test", gt=5)
    with pytest.raises(ValueError):
        number_validator(5, "Test", ge=6)
    with pytest.raises(ValueError):
        number_validator(5, "Test", lt=5)
    with pytest.raises(ValueError):
        number_validator(5, "Test", le=4)

    # Test with conflicting constraints
    with pytest.raises(ValueError):
        number_validator(5, "Test", gt=3, ge=4)
    with pytest.raises(ValueError):
        number_validator(5, "Test", lt=7, le=6)


def test_list_of_nums_validator():
    """..."""
    # Test with valid inputs
    list_of_nums_validator([5, 6, 7], "Test", 3, ge=4, le=8)
    list_of_nums_validator((5, 6, 7), "Test", 3, gt=4, lt=8)

    # Test with invalid inputs
    with pytest.raises(ValueError):
        list_of_nums_validator("5, 6, 7", "Test", 3, ge=4, le=8)
    with pytest.raises(ValueError):
        list_of_nums_validator([5, 6, 7], "Test", 4, ge=4, le=8)
    with pytest.raises(ValueError):
        list_of_nums_validator([5, 6, 7], "Test", 3, ge=6, le=8)
    with pytest.raises(ValueError):
        list_of_nums_validator([5, 6, 7], "Test", 3, ge=4, le=6)

    # Test with conflicting constraints
    with pytest.raises(ValueError):
        list_of_nums_validator([5, 6, 7], "Test", 3, gt=4, ge=5)
    with pytest.raises(ValueError):
        list_of_nums_validator([5, 6, 7], "Test", 3, lt=8, le=7)


def test_estimate_sri_properties():
    """..."""
    # Test with valid inputs
    sa, ta = estimate_sri_properties(50, 0.85, 5)
    assert isinstance(sa, float)
    assert isinstance(ta, float)
    assert 0 <= sa <= 1
    assert 0 < ta < 1

    # Test with invalid inputs
    with pytest.raises(ValueError):
        estimate_sri_properties(-1, 0.85, 5)
    with pytest.raises(ValueError):
        estimate_sri_properties(123, 0.85, 5)
    with pytest.raises(ValueError):
        estimate_sri_properties(50, -0.1, 5)
    with pytest.raises(ValueError):
        estimate_sri_properties(50, 1.1, 5)

    # Test with edge case
    sa, ta = estimate_sri_properties(0, 0.85, 5)
    assert 0 <= sa <= 1
    assert 0 < ta < 1

    assert estimate_sri_properties(
        target_sri=1, target_emittance=0.1, tolerance=1
    ) == pytest.approx((0.77, 0.1), 0.05)
    assert estimate_sri_properties(
        target_sri=1, target_emittance=0.1, tolerance=5
    ) == pytest.approx((0.77, 0.1), 0.05)
    assert estimate_sri_properties(
        target_sri=1, target_emittance=0.1, tolerance=10
    ) == pytest.approx((0.77, 0.1), 0.05)
    assert estimate_sri_properties(
        target_sri=1, target_emittance=0.7, tolerance=1
    ) == pytest.approx((0.895, 0.7), 0.05)
    assert estimate_sri_properties(
        target_sri=1, target_emittance=0.7, tolerance=5
    ) == pytest.approx((0.895, 0.7), 0.05)
    assert estimate_sri_properties(
        target_sri=1, target_emittance=0.7, tolerance=10
    ) == pytest.approx((0.89, 0.7), 0.05)
    assert estimate_sri_properties(
        target_sri=1, target_emittance=0.9, tolerance=1
    ) == pytest.approx((0.94, 0.9), 0.05)
    assert estimate_sri_properties(
        target_sri=1, target_emittance=0.9, tolerance=5
    ) == pytest.approx((0.94, 0.9), 0.05)
    assert estimate_sri_properties(
        target_sri=1, target_emittance=0.9, tolerance=10
    ) == pytest.approx((0.93, 0.9), 0.05)
    assert estimate_sri_properties(
        target_sri=50, target_emittance=0.1, tolerance=1
    ) == pytest.approx((0.405, 0.1), 0.05)
    assert estimate_sri_properties(
        target_sri=50, target_emittance=0.1, tolerance=5
    ) == pytest.approx((0.405, 0.1), 0.05)
    assert estimate_sri_properties(
        target_sri=50, target_emittance=0.1, tolerance=10
    ) == pytest.approx((0.40, 0.1), 0.05)
    assert estimate_sri_properties(
        target_sri=50, target_emittance=0.7, tolerance=1
    ) == pytest.approx((0.53, 0.7), 0.05)
    assert estimate_sri_properties(
        target_sri=50, target_emittance=0.7, tolerance=5
    ) == pytest.approx((0.53, 0.7), 0.05)
    assert estimate_sri_properties(
        target_sri=50, target_emittance=0.7, tolerance=10
    ) == pytest.approx((0.53, 0.7), 0.05)
    assert estimate_sri_properties(
        target_sri=50, target_emittance=0.9, tolerance=1
    ) == pytest.approx((0.575, 0.9), 0.05)
    assert estimate_sri_properties(
        target_sri=50, target_emittance=0.9, tolerance=5
    ) == pytest.approx((0.575, 0.9), 0.05)
    assert estimate_sri_properties(
        target_sri=50, target_emittance=0.9, tolerance=10
    ) == pytest.approx((0.57, 0.9), 0.05)
    assert estimate_sri_properties(
        target_sri=100, target_emittance=0.1, tolerance=1
    ) == pytest.approx((0.03, 0.1), 0.05)
    assert estimate_sri_properties(
        target_sri=100, target_emittance=0.1, tolerance=5
    ) == pytest.approx((0.03, 0.1), 0.05)
    assert estimate_sri_properties(
        target_sri=100, target_emittance=0.1, tolerance=10
    ) == pytest.approx((0.05, 0.1), 0.05)
    assert estimate_sri_properties(
        target_sri=100, target_emittance=0.7, tolerance=1
    ) == pytest.approx((0.16, 0.7), 0.05)
    assert estimate_sri_properties(
        target_sri=100, target_emittance=0.7, tolerance=5
    ) == pytest.approx((0.16, 0.7), 0.05)
    assert estimate_sri_properties(
        target_sri=100, target_emittance=0.7, tolerance=10
    ) == pytest.approx((0.16, 0.7), 0.05)
    assert estimate_sri_properties(
        target_sri=100, target_emittance=0.9, tolerance=1
    ) == pytest.approx((0.2, 0.9), 0.05)
    assert estimate_sri_properties(
        target_sri=100, target_emittance=0.9, tolerance=5
    ) == pytest.approx((0.2, 0.9), 0.05)
    assert estimate_sri_properties(
        target_sri=100, target_emittance=0.9, tolerance=10
    ) == pytest.approx((0.2, 0.9), 0.05)

    # test values received
