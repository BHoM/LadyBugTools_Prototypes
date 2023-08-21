from scipy.interpolate import interp1d


def water_temperature_at_depth(surface_temperature: float, depth: float) -> float:
    """The temperature of water at a given depth.

    Source:
        Based on https://waterproffi.com/water-temperature-at-depth-calculator/\n
        0-10 meters       0.40 - 1.00 degrees Celsius\n
        10-50 meters      0.20 - 0.50 degrees Celsius per 10 meters\n
        50-100 meters     0.10 - 0.30 degrees Celsius per 10 meters\n
        100-500 meters    0.05 - 0.10 degrees Celsius per 10 meters\n
        500-1,000 meters  0.02 - 0.05 degrees Celsius per 10 meters\n
        Deeper regions    Gradual decrease in temperature

    Args:
        surface_temperature (float):
            Temperature of the water at the surface in C. Surface in this method is
            defined as 0.2m before the boundary layer, per measurement using
            the method outlined in https://doi.org/10.1175/JCLI-D-20-0166.1.
        depth (float):
            Depth of the water in m.

    Returns:
        float:
            Temperature of the water at the given depth in C.
    """

    if depth == 0:
        return surface_temperature
    if depth < 0:
        raise ValueError("depth must be greater than 0")

    # construct depth reduction profile
    depths = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    ]
    reductions = [
        0,
        0.4,
        0.475,
        0.55,
        0.625,
        0.7,
        0.775,
        0.85,
        0.925,
        1,
        1.2,
        1.5,
        1.9,
        2.4,
        2.5,
        2.35,
        2.55,
        2.8,
        3.1,
        3.6,
        4.266666667,
        5.1,
        6.1,
        6.3,
        6.575,
        6.925,
        7.35,
        7.85,
    ]

    interpolant = interp1d(depths, reductions, kind="quadratic", bounds_error=True)

    return surface_temperature - interpolant(depth)
