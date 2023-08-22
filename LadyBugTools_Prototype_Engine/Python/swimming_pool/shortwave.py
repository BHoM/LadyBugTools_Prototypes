import pandas as pd


def shortwave_gain(
    surface_area: float, insolation: float, solar_absorbtivity: float = 0.85
) -> float:
    """Calculate the solar gain from radiation incident on a surface.

    Args:
        surface_area (float): Surface area of the pool in m2.
        insolation (float): Insolation incident on the surface in W/m2.
        solar_absorbtivity (float, optional): Solar absorbtivity of the surface.
            Defaults to 0.85 based on ISOTC180 (1995).

    Returns:
        float: Solar gain from radiation incident on the surface in W.
    """
    res = surface_area * insolation * solar_absorbtivity
    if isinstance(res, pd.Series):
        return res.rename("Q_Solar (W)")
    return res
