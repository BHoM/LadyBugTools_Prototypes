"""Methods for post-processing honeybee-radiance daylight results."""

from honeybee_radiance.sensorgrid import SensorGrid, Sensor
from pathlib import Path
from ladybugtools_toolkit.honeybee_extension.results import (
    load_ill,
    load_npy,
    load_pts,
    make_annual,
)
import pandas as pd


# BR 209
def side_lit_target_illuminance(
    illuminance_file: Path, target_illuminance: float = 300
) -> pd.Series:
    """
    Determine the percentage of hours that a target illuminance is met on a sensor grid, and the area over which that target is achieved.

    This is based on BR 209 2022 Site Layout Planning for Daylight and Sunlight: A Guide to Good Practice, Appendix C:Interior daylighting recommendations.

    From the above guide, the following is stated:
    "C5 A target illuminance (ET) should be achieved across at least half of the reference plane in a daylit space for
    at least half of the daylight hours. Another target illuminance (ETM) should also be achieved across 95% of the
    reference plane for at least half of the daylight hours; this is the minimum target illuminance to be achieved towards
    the back of the room."

    Table C1 - Target illumiannce from daylight over at least half of daylight hours
    Level of recommendation |                            Target Illuminance                             |
                            | ET (lx) for half of assessment grid | ETM (lx) for 95% of assessment grid |
    ------------------------|-------------------------------------|-------------------------------------|
    Minimum                 | 300                                 | 100                                 |
    Medium                  | 500                                 | 300                                 |
    High                    | 750                                 | 500                                 |
    ------------------------|-------------------------------------|-------------------------------------|
    """

    # load the illuminance file
    ill_values = load_ill([illuminance_file])
    n_daylit_hours = len(ill_values)

    # ill_values
