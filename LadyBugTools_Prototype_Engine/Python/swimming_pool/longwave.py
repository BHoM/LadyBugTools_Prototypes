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
        water_emissivity (float, optional): Emissivity of the water surface.

    Returns:
        float: Longwave gain from the sky on a surface in W.
    """

    stefan_boltzmann = 5.670374419e-8  # Stefan-Boltzmann constant in W/m2K4

    return (
        stefan_boltzmann
        * water_emissivity
        * (((sky_temperature + 273.15) ** 4) - ((water_temperature + 273.15) ** 4))
    ) * surface_area  # W
