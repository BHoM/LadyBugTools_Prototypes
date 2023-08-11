import numpy as np
import pandas as pd


def occupancy_schedule() -> pd.Series:
    """The default occupancy profile for a pool. This method returns a pandas
    Series contaiing values between 0 and 1, representing the proportion
    occupancy of a pool hourly across a standard year.

    The profile returned is based on the example given in EnergyPlus for a
    5ZoneSwimmingPool, with an additional factor applied based on month of year,
    where people are more likely to opccupy a pool in summer than in winter.

    Returns:
        pd.Series: A timestep-linked occupancy profile, between 0-1 representing
        the fraction of occupancy.
    """

    idx = pd.date_range("2017-01-01 00:00:00", freq="60T", periods=8760)

    daily_occupancy = np.where(
        idx.hour.isin([0, 1, 2, 3, 4, 5, 20, 21, 22]),
        0,
        np.where(idx.hour.isin([6, 7, 8, 11, 12, 16, 17, 18, 19]), 1, 0.5),
    )
    seasonal_occupancy = 0.75 + (
        -np.cos(np.interp(np.arange(len(idx)), [1, len(idx) - 1], [0, np.pi * 2])) / 4
    )

    return (
        pd.Series(
            daily_occupancy * seasonal_occupancy, index=idx, name="occupancy_profile"
        )
        .interpolate("linear")
        .clip(0, 1)
    )


def cover_schedule() -> pd.Series:
    """A default profile for the covering of a body of water. This method returns
    a pandas Series contaiing values between 0 and 1, representing the proportion
    of the pool covered hourly across a standard year.

    The profile returned is based on the example given in EnergyPlus for a
    5ZoneSwimmingPool.

    Returns:
        pd.Series: A timestep-linked covering profile, between 0-1 representing
        the fraction of covering.
    """

    idx = pd.date_range("2017-01-01 00:00:00", freq="60T", periods=8760)

    return pd.Series(
        np.where(idx.hour.isin([0, 1, 2, 3, 4, 5]), 0.5, 0),
        name="cover_profile",
        index=idx,
    )
