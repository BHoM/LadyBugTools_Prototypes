import pandas as pd


def longwave_gain(
    surface_area: float,
    sky_temperature: float,
    water_temperature: float,
    water_emissivity: float = 0.95,
) -> float:
    """Calculate the longwave gain from the sky on a surface.

    Args:
        surface_area (float): Surface area of the pool in m2.
        sky_temperature (float): Sky temperature in C.
        water_temperature (float): Temperature of the water in C.
        water_emissivity (float, optional): Emissivity of the water surface. Defaults to 0.95.

    Returns:
        float: Longwave heat exchange with the sky from a water body, in W.
    """

    res = (
        5.670374419e-8
        * water_emissivity
        * (((sky_temperature + 273.15) ** 4) - ((water_temperature + 273.15) ** 4))
        * surface_area
    )
    if isinstance(res, pd.Series):
        return res.rename("Q_Longwave (W)")

    return res
