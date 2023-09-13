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
from warnings import warn
from honeybee_radiance_postprocess.en17037 import en17037_to_folder


# BR 209
def br209_illuminance_method_c1(
    illuminance_df: pd.DataFrame,
    target_illuminance: float,
    assessment_grid_proportion: float,
    n_daylight_hours: float = None,
) -> float:
    """
    Determine the percentage of hours that a target illuminance is met on a sensor grid, and the area over which that target is achieved.

    Args:
        illuminance_df (pd.DataFrame):
            A pandas DataFrame with the illuminance values for each sensor in a single sensor grid. The index of the DataFrame should be timestamps, with each column representing a sensor in the assessment grid.
        target_illuminance (float):
            The target illuminance in lux.
        assessment_grid_proportion (float):
            The proportion of the sensor grid that must achieve the target illuminance for the result to be considered a pass. Should be a value between 0 and 1.
        n_daylight_hours (int, optional):
            The number of daylight hours in the analysis period. Default: None.

    Returns:
        bool:
            True if the target illuminance is achieved for at least half of the simulated period, over at the target assessment grid proportion. False otherwise.

    Documentation:
        This is based on BR 209 2022 Site Layout Planning for Daylight and Sunlight: A Guide to Good Practice, Appendix C:Interior daylighting recommendations.

        From the above guide, the following is stated:
        "C5 A target illuminance (ET) should be achieved across at least half of the reference plane in a daylit space for
        at least half of the daylight hours. Another target illuminance (ETM) should also be achieved across 95% of the
        reference plane for at least half of the daylight hours; this is the minimum target illuminance to be achieved towards
        the back of the room."

        Table C1 - Target illuminance from daylight over at least half of daylight hours
        Level of recommendation |                            Target Illuminance                             |
                                | ET (lx) for half of assessment grid | ETM (lx) for 95% of assessment grid |
        ------------------------|-------------------------------------|-------------------------------------|
        Minimum                 | 300                                 | 100                                 |
        Medium                  | 500                                 | 300                                 |
        High                    | 750                                 | 500                                 |
        ------------------------|-------------------------------------|-------------------------------------|
    """

    if target_illuminance <= 0:
        raise ValueError("target_illuminance must be greater than zero.")

    if assessment_grid_proportion <= 0 or assessment_grid_proportion > 1:
        raise ValueError("assessment_grid_proportion must be between 0 and 1.")

    if n_daylight_hours is None:
        n_daylight_hours = len(illuminance_df)
        warn(
            f"No value given to n_daylight_hours. The illuminance_df will be assumed to be the same length as daylight-hours per Honeybee-Radiance convention ({n_daylight_hours} hours). If a custom time-period was used for simulation the result given will be incorrect."
        )

    # checks here to let user know if combinations of target_illuminance and assessment_grid_proportion are not "standard"
    if (target_illuminance, assessment_grid_proportion) not in [
        (300, 0.5),
        (500, 0.5),
        (750, 0.5),
        (100, 0.95),
        (300, 0.95),
        (500, 0.95),
    ]:
        warn(
            f"target_illuminance of {target_illuminance} and assessment_grid_proportion of {assessment_grid_proportion} are not standard combinations. See BR 209 2022 Appendix C for standard combinations."
        )

    # determine the number of sensors in the sensor grid
    n_sensors = len(illuminance_df.columns)

    # create a booean matrix indicating whether the target illuminance has been met
    achieves_target = illuminance_df > target_illuminance

    # determine whether a proportion of the analysis plane achieves the target for each timestep
    achieves_spatial = achieves_target.sum(axis=1) > (
        n_sensors * assessment_grid_proportion
    )

    # determine whether at least half of the time, the sensors achieve the target
    achieves_temporal = achieves_spatial.sum() / n_daylight_hours

    if achieves_temporal >= 0.5:
        return True

    return False


def br209_daylightfactor_method_c2(
    daylight_factor_series: pd.Series,
    target_daylight_factor: float,
    assessment_grid_proportion: float,
) -> float:
    """
    Determine whether a target daylight factor is achieved over a given proportion of an assessment grid.

    Args:
        daylight_factor_series (pd.Series):
            A pandas Series with the daylight factor values for each sensor in a single sensor grid.
        target_daylight_factor (float):
            The target daylight factor, as a value between 0-1. For example 5% would be 0.05.
        assessment_grid_proportion (float):
            The proportion of the sensor grid that must achieve the target daylight factor for the result to be considered a pass. Should be a value between 0 and 1.

    Returns:
        bool:
            True if the target daylight factor is achieved, over at the target assessment grid proportion. False otherwise.

    Documentation:
        This is based on BR 209 2022 Site Layout Planning for Daylight and Sunlight: A Guide to Good Practice, Appendix C:Interior daylighting recommendations.

    """

    if not isinstance(daylight_factor_series, pd.Series):
        raise TypeError("daylight_factor_series must be a pandas Series.")

    if target_daylight_factor <= 0 or target_daylight_factor > 1:
        raise ValueError("target_daylight_factor must be between 0 and 1.")

    if assessment_grid_proportion <= 0 or assessment_grid_proportion > 1:
        raise ValueError("assessment_grid_proportion must be between 0 and 1.")

    # determine the number of sensors in the sensor grid
    n_sensors = len(daylight_factor_series)

    # create a booean matrix indicating whether the target daylight factor has been met
    achieves_target = daylight_factor_series > target_daylight_factor

    # determine whether at least half of the analysis plane achives the target
    return achieves_target.sum() > (n_sensors * assessment_grid_proportion)
