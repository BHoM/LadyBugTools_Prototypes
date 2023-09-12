"""A translation of the Fortran-90 ETM code to Python."""

import numpy as np


def delcalc(ta: float) -> float:
    """Function to calculate the slope of the vapour pressure curve

    Args:
        ta (float): air temperature (deg. C)

    Returns:
        float: slope of the vapour pressure curve (kPa deg. C-1)
    """

    ea = 0.611 * np.exp(17.27 * ta / (ta + 237.15))
    return 4099 * ea / (ta + 237.15) ** 2


def alambdat(t: float) -> float:
    """Function to correct the latent heat of vaporisation for temperature

    Args:
        t (float): temperature (deg. C)

    Returns:
        float: latent heat of vaporisation (MJ kg-1)
    """

    return 2.501 - t * 2.2361e-3


def psyconst(p: float, alambda: float) -> float:
    """Function to calculate the psychrometric constant from atmospheric pressure and latent heat of vaporisation

    Args:
        p (float): atmospheric pressure (kPa)
        alambda (float): latent heat of vaporisation (MJ kg-1)

    Returns:
        float: psychrometric constant (kPa deg. C-1)
    """

    cp = 1.013
    eta = 0.622
    return cp * p / (eta * alambda) * 1.0e-3


def equibtemp(
    albedo: float,
    depth: float,
    tstep: float,
    vpd: float,
    u: float,
    ta: float,
    tn: float,
    solrad: float,
    longrad: float,
    fc: float,
    tw0: float,
) -> (float, float, float, float, float):
    """
    Subroutine to calculate the evaporation, using daily data, from a
    water body using the equilibrium temperature model of de Bruin,
    H.A.R., 1982, j.hydrol, 59, 261-274

    Args:
        albedo (float): ALBEDO OF THE WATER BODY
        depth (float): DEPTH OF THE WATER BODY (m)
        tstep (float): THE TIME STEP FOR THE MODEL TO USE (days)
        vpd (float): VAPOUR PRESSURE DEFICIT (mb)
        u (float): WIND SPEED (m s-1)
        ta (float): AIR TEMPERATURE (deg.C)
        tn (float): WET BULB TEMPERATURE (deg.C)
        solrad (float): DOWNWELLING SOLAR RADIATION (W m-2 per day)
        longrad (float): DOWNWELLING LONG WAVE RADIATION (W m-2 per day)
        fc (float): CLOUDINESS FACTOR
        tw0 (float): TEMPERATURE OF THE WATER ON THE PREVIOUS TIME STEP (deg.C)

    Returns:
        rn (float): NET RADIATION (W m-2 per day)
        le (float): LATENT HEAT FLUX (W m-2 per day)
        deltas (float): CHANGE IN HEAT STORAGE (W m-2 per day)
        tw (float): TEMPERATURE OF THE WATER AT THE END OF THE TIME PERIOD (deg.C)
        evap (float): EVAPORATION CALCULATED USING THE PENMAN-MONTEITH FORMULA (mm per day)
    """

    # setup constants
    _lambda = alambdat(ta)  # LATENT HEAT OF VAPORISATION (MJ kg-1)
    gamma = psyconst(100.0, _lambda)  # PSCHROMETRIC CONSTANT (kPa deg.C-1)
    rhow = 1000.0  # DENSITY OF WATER (kg m-3)
    cw = 0.0042  # SPECIFIC HEAT OF WATER (MJ kg-1 deg.C-1)
    rho = 1.0  # DENSITY OF AIR (kg m-3)
    cp = 1.013  # SPECIFIC HEAT OF AIR (KJ kg-1 deg.C-1
    sigma = 4.9e-9  # STEFAN-BOLTZMANN CONSTANT (MJ m-2 deg.C-4 d-1)
    k = 0.41  # VON KARMAN CONSTANT
    degabs = 273.13  # DIFFERENCE BETWEEN DEGREES KELVIN AND DEGREES CELSIUS
    zr = 10.0  # HEIGHT OF MEASUREMENTS ABOVE WATER SURFACE (m) ASSUMED TO BE SCREEN HEIGHT

    # initialise output variables
    deltas = 0.0
    evap = 0.0
    evappm = 0.0
    le = 0.0
    lepm = 0.0
    rn = 0.0
    tw = 0.0

    # check for simple errors
    if (albedo <= 0.0) or (albedo >= 1.0):
        raise ValueError("albedo must be between 0 and 1")

    if depth <= 0.0:
        raise ValueError("depth must be greater than 0")

    if tn > ta:
        raise ValueError("air temperature must be greater than wet bulb temperature")

    if solrad <= 0.0:
        raise ValueError("downwelling solar radiation must be greater than 0")

    ut = max(u, 0.01)
    vpd = max(vpd, 0.0001)

    #  convert from W m-2 to mJ m-2 d-1
    sradj = solrad * 0.0864
    lradj = longrad * 0.0864

    # convert from mbar to kPa
    vpdp = vpd * 0.1

    # calculate the slope of the temperature-saturation water vapour curve at the wet bulb temperature (kPa deg C-1)
    deltaw = delcalc(tn)

    # calculate the slope of the temperature-saturation water vapour curve at the air temperature (kPa deg C-1)
    deltaa = delcalc(ta)

    # calculate the net radiation for the water temperature (MJ m-2 d-1)
    lradj = fc * sigma * (ta + degabs) ** 4 * (0.53 + 0.067 * np.sqrt(vpd))
    rn = (
        sradj * (1.0 - albedo)
        + lradj
        - fc
        * (sigma * (ta + degabs) ** 4 + 4.0 * sigma * (ta + degabs) ** 3 * (tw0 - ta))
    )

    # calculate the net radiation when the water temperature equals the wet bulb temperature. assumes the emissivity of water is 1 (MJ m-2 d-1)
    rns = (
        sradj * (1.0 - albedo)
        + lradj
        - fc
        * (sigma * (ta + degabs) ** 4 + 4.0 * sigma * (ta + degabs) ** 3 * (tn - ta))
    )

    # CALCULATE THE WIND FUNCTION (MJ m-2 d-1 kPa-1) USING THE METHOD OF Sweers, H.E., 1976, J.Hydrol., 30, 375-401, NOTE THIS IS FOR MEASUREMENTS FROM A LAND BASED MET. STATION AT A HEIGHT OF 10 m BUT WE CAN ASSUME THAT THE DIFFERENCE BETWEEN 2 m AND 10 m IS NEGLIGIBLE
    windf = (4.4 + 1.82 * ut) * 0.864

    # CALCULATE THE TIME CONSTANT (d) TIME CONSTANT OF THE WATER BODY (days)
    tau = (rhow * cw * depth) / (
        4.0 * sigma * (tn + degabs) ** 3 + windf * (deltaw + gamma)
    )

    # CALCULATE THE EQUILIBRIUM TEMPERATURE (deg. C)
    te = tn + rns / (4.0 * sigma * (tn + degabs) ** 3 + windf * (deltaw + gamma))

    # CALCULATE THE TEMPERATURE OF THE WATER (deg. C)
    tw = te + (tw0 - te) * np.exp(-tstep / tau)

    # CALCULATE THE CHANGE IN HEAT STORAGE (MJ m-2 d-1)
    deltas = rhow * cw * depth * (tw - tw0) / tstep

    # z0 - ROUGHNESS LENGTH - DUE TO SMOOTHNESS OF THE SURFACE THE ROUGHNESS LENGTHS OF MOMENTUM AND WATER VAPOUR CAN BE ASSUMED TO BE THE SAME
    z0 = 0.001

    # CALCULATE THE AERODYNAMIC RESISTANCE ra (s m-1)
    ra = np.log(zr / z0) ** 2 / (k * k * ut)

    # CALCULATE THE PENMAN-MONTEITH EVAPORATION
    le = (deltaa * (rn - deltas) + 86.4 * rho * cp * vpdp / ra) / (deltaa + gamma)
    evap = le / _lambda

    # CONVERT THE FLUXES TO W m-2
    rn = rn / 0.0864
    le = le / 0.0864
    deltas = deltas / 0.0864

    return rn, le, deltas, tw, evap
