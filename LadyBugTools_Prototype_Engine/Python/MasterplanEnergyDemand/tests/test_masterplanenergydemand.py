#!/usr/bin/env python

"""Tests for `masterplanenergydemand` package."""

import pytest
import sys
from pathlib import Path

# get current file location and determine location of the package
package_dir = Path(__file__).parents[1]
package_name = "masterplanenergydemand"
sys.path.insert(0, package_dir.as_posix())

from masterplanenergydemand.mped import MPED
from masterplanenergydemand.enum import BuildingType, typical_footprint_area

TEST_EPW_FILE = package_dir / "tests" / "test.epw"


def test_creation():
    """Test the creation of an MPED object."""

    # creation of an MPED object without arguments should raise a TypeError
    with pytest.raises(TypeError):
        MPED()

    # create an MPED object with an EPW file that doesnt exist raises a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        MPED(epw_file="nonexistent.epw", total_area=10000)

    # creation of an MPED object with an EPW file and total area, with all defaults
    assert isinstance(
        MPED(epw_file=TEST_EPW_FILE, total_area=1000, average_footprint_area=1000), MPED
    )

    # test creation of MPED with all different building types
    for building_type in BuildingType:
        assert isinstance(
            MPED(
                epw_file=TEST_EPW_FILE,
                total_area=typical_footprint_area(building_type),
                building_type=building_type,
            ),
            MPED,
        )


def test_simulation():
    """Test the simulation of an MPED object."""

    # create an MPED object with an EPW file and total area, with all defaults
    mped = MPED(
        epw_file=TEST_EPW_FILE,
        total_area=100,
        average_footprint_area=100,
        average_num_floors=1,
        building_identifier="pytest_building",
        case_identifier="pytest_case",
        project_identifier="pytest_project",
    )

    # run the simulation
    mped.run_all()
    assert mped._sql_file.exists()
    assert mped._config_json.exists()
    assert mped._energy_consumption_file_normalised.exists()
    assert mped._thermal_load_balance_file_normalised.exists()
