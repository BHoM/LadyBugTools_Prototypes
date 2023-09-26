import numpy as np
import pandas as pd
import pyet
from ladybugtools_toolkit.ladybug_extension.epw import (
    EPW,
    Sunpath,
    collection_to_series,
    wind_speed_at_height,
)
from ladybug.psychrometrics import saturated_vapor_pressure, wet_bulb_from_db_rh
from scipy.interpolate import interp1d, LinearNDInterpolator


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
        float: latent heat of vaporisation (J kg-1)
    """

    return (2.501 - t * 2.2361e-3) * 1000 * 1000


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


def vpdcalc(svp, rh):
    return svp * (1 - (rh / 100))


def equibtemp1(
    depth: float,
    rh: float,
    u: float,
    ta: float,
    solrad: float,
    longrad: float,
    tw0: float,
    albedo: float = 0.08,
    fc: float = 0,  # atm using 0 as default as not sure how to calc
    tstep: float = 1,  # hours
) -> float:
    """
    Subroutine to calculate the evaporation, using daily data, from a
    water body using the equilibrium temperature model of de Bruin,
    H.A.R., 1982, j.hydrol, 59, 261-274

    Args:
        albedo (float): ALBEDO OF THE WATER BODY
        depth (float): DEPTH OF THE WATER BODY (m)
        tstep (float): THE TIME STEP FOR THE MODEL TO USE (-)
        rh (float): RELATIVE HUMIDITY (%)
        u (float): WIND SPEED (m s-1)
        ta (float): AIR TEMPERATURE (deg.C)
        solrad (float): DOWNWELLING SOLAR RADIATION (W m-2)
        longrad (float): DOWNWELLING LONG WAVE RADIATION (W m-2)
        fc (float): CLOUDINESS FACTOR
        tw0 (float): TEMPERATURE OF THE WATER ON THE PREVIOUS TIME STEP (deg.C)

    Returns:
        rn (float): NET RADIATION (W m-2)
        le (float): LATENT HEAT FLUX (W m-2)
        deltas (float): CHANGE IN HEAT STORAGE (W m-2)
        tw (float): TEMPERATURE OF THE WATER AT THE END OF THE TIME PERIOD (deg.C)
        evap (float): EVAPORATION CALCULATED USING THE PENMAN-MONTEITH FORMULA (mm per time step)
    """

    tn = wet_bulb_from_db_rh(ta, rh)

    vpd = vpdcalc(saturated_vapor_pressure(ta), rh)

    # setup constants
    _lambda = alambdat(ta)  # LATENT HEAT OF VAPORISATION (kJ kg-1)
    gamma = psyconst(100.0, _lambda)  # PSCHROMETRIC CONSTANT (kPa deg.C-1)
    rhow = 1000.0  # DENSITY OF WATER (kg m-3)
    cw = 4.2  # SPECIFIC HEAT OF WATER (KJ kg-1 deg.C-1)
    rho = 1.0  # DENSITY OF AIR (kg m-3)
    cp = 1.013  # SPECIFIC HEAT OF AIR (KJ kg-1 deg.C-1)
    sigma = 5.7e-8 / 1000  # STEFAN-BOLTZMANN CONSTANT (KW m-2 K-4)
    k = 0.41  # VON KARMAN CONSTANT
    degabs = 273.15  # DIFFERENCE BETWEEN KELVIN AND DEGREES CELSIUS
    zr = 10.0  # HEIGHT OF MEASUREMENTS ABOVE WATER SURFACE (m) ASSUMED TO BE SCREEN HEIGHT   (strange anachronism from the paper)

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

    if solrad < 0.0:
        raise ValueError(
            "downwelling solar radiation must be greater than or equal to 0"
        )

    ut = max(u, 0.01)
    vpd = max(vpd, 0.0001)

    # initialise solar and lwav radiation
    sradj = solrad
    lradj = longrad

    # convert from Pa to kPa
    vpdp = vpd

    # calculate the slope of the temperature-saturation water vapour curve at the wet bulb temperature (kPa deg C-1)
    deltaw = delcalc(tn)

    # calculate the slope of the temperature-saturation water vapour curve at the air temperature (kPa deg C-1)
    deltaa = delcalc(ta)

    # calculate the net radiation for the water temperature (KJ m-2)
    lradj = fc * sigma * (ta + degabs) ** 4 * (0.53 + 0.067 * np.sqrt(vpd))
    rn = (
        sradj * (1.0 - albedo)
        + lradj
        - fc
        * (sigma * (ta + degabs) ** 4 + 4.0 * sigma * (ta + degabs) ** 3 * (tw0 - ta))
    )

    # calculate the net radiation when the water temperature equals the wet bulb temperature. assumes the emissivity of water is 1 (KJ m-2)
    rns = (
        sradj * (1.0 - albedo)
        + lradj
        - fc
        * (sigma * (ta + degabs) ** 4 + 4.0 * sigma * (ta + degabs) ** 3 * (tn - ta))
    )

    # CALCULATE THE WIND FUNCTION (MJ m-2 d-1 kPa-1) USING THE METHOD OF Sweers, H.E., 1976, J.Hydrol., 30, 375-401, NOTE THIS IS FOR MEASUREMENTS FROM A LAND BASED MET. STATION AT A HEIGHT OF 10 m BUT WE CAN ASSUME THAT THE DIFFERENCE BETWEEN 2 m AND 10 m IS NEGLIGIBLE
    windf = (4.4 + 1.82 * ut) * 0.864  # MJ m-2 d-1 kPa-1

    # CALCULATE THE TIME CONSTANT OF THE WATER BODY (hours)
    tau = (rhow * cw * depth) / (
        4.0 * sigma * (tn + degabs) ** 3 + windf * (deltaw + gamma)
    )

    # CALCULATE THE EQUILIBRIUM TEMPERATURE (deg. C)
    te = tn + rns / (4.0 * sigma * (tn + degabs) ** 3 + windf * (deltaw + gamma))

    # CALCULATE THE TEMPERATURE OF THE WATER (deg. C)
    tw = te + (tw0 - te) * np.exp(-tstep / tau)

    # CALCULATE THE CHANGE IN HEAT STORAGE (KJ m-2)
    deltas = rhow * cw * depth * (tw - tw0)

    # z0 - ROUGHNESS LENGTH - DUE TO SMOOTHNESS OF THE SURFACE THE ROUGHNESS LENGTHS OF MOMENTUM AND WATER VAPOUR CAN BE ASSUMED TO BE THE SAME
    z0 = 0.001

    # CALCULATE THE AERODYNAMIC RESISTANCE ra (s m-1)
    ra = np.log(zr / z0) ** 2 / (k * k * ut)

    # CALCULATE THE PENMAN-MONTEITH EVAPORATION
    le = deltaa * (rn - deltas) + 86.4 * rho * cp * (vpvp / ra) / (deltaa + gamma)
    evap = le / _lambda

    # CONVERT THE FLUXES TO W m-2
    rn = rn
    le = le
    deltas = deltas

    return evap, tw, deltas, lradj


def equibtemp(
    t_sky, # sky temperature (C)
    depth,  # depth (m)
    u,  # wind speed (m s-1)
    t_air,  # air temperature (C)
    t_wb, #wet bulb temperature (C)
    rh,  # relative humidity (%)
    q_solar,  # incoming solar energy (J)
    t_prev,  # previous water temperature (C)
    wind_height=10,  # wind height above water (m)
    albedo=0.08,  # albedo
    tstep=1
):
    # TODO - sky heat transfer
    # TODO - Documentation!
    # TODO - longer variable names
    # TODO - be clever about albedo based on angle of incidence of sun on water surface. if sun is normal to surface, albedo is low, if sun is low, albedo is high-er
    # TODO - add output for q_occupants (and implement occupants impact on temperature, see later TODO)
    # TODO - add output for q_longwave
    # TODO - add output for q_conduction with pool lining - small value, but would be nice to have
    # TODO - add output for q_convection, use Bowen ratio method
    # MAYBENOTTODO - add output for q_conditioning_water_temp
    # MAYBENOTTODO - add output for q_conditioning_heat_balance

    # conversions to SI units (J, kg, K, m)
    sigma = 5.7e-8  # stefan boltzmann constant
    k = 0.41  # Von Karman constant
    z0 = 0.001  # roughness length assumed to be very low
    ut = max(u, 0.01)  # wind speed (m s-1)
    zr = wind_height  # wind height above water (m)
    e_air = 0.70  # air emmissivity (assumption to be refined)
    e_w = 0.95  # water emmissivity
    lhv = alambdat(t_air)  # latent heat of vaporisation of water (J kg-1)
    t_sky += 273.15 # sky temperature (K)
    t_air += 273.15  # air temperature (K)
    t_wb += 273.15  # wet bulb temperature (K)
    t0 = t_prev + 273.15  # previous water temperature (K)
    cw = 4200  # specific heat capacity of water (J kg-1 K-1)
    c_a = 1013  # specific heat capacity of air (J kg-1 K-1)
    gamma = psyconst(100.0, lhv/(1000*1000))  # psychrometric constant (kPa K-1)
    rho_w = 1000  # density of water (kg m-3)
    rho_a = 1  # density of air (kg m-3)

    # NOTE - The magic numbers are magic. Trust them and they shall do you well. Might need some investigation, but for now the Env Agency says it's good
    wf = 4.4 + 1.82 * ut # W m-2 kPa-1

    # heat echange with the atmosphere/sky
    # TODO - update for sky temperature - https://www.engineeringtoolbox.com/radiation-heat-transfer-d_431.html Eqn3

    lwav_out_wb = e_w * (sigma * ((t_sky**4) - (t_wb**4)))  # W

    lwav_out_t0 = e_w * (sigma * ((t_sky**4) - (t0**4)))  # W

    # Net radiation for wet bulb or t0(previous temperature)
    net_rad_wb = (q_solar * (1 - albedo) + (lwav_out_wb))  # W

    net_rad_t0 = (q_solar * (1 - albedo) + (lwav_out_t0))  # W

    # time constant ... does something ... probably clever
    tau = (rho_w * cw * depth) / (
        4 * sigma * (t_wb**3) + wf * (delcalc(t_wb - 273.15) + gamma)
    )  # seconds

    # equilibrium temperature. Temperature at which no heat exchange occurs
    t_equib = t_wb + ((net_rad_wb) / (
        4 * sigma * (t_wb**3) + wf * (delcalc(t_wb - 273.15) + gamma)
    ))  # K

    # actual tempertaure of water after time step
    t_final = t_equib + (t0 - t_equib) * np.exp(tstep / tau)  # K

    # TODO - add additional heat gains in here, e.g. occupants, convection, conduction

    # net radiation exhange, minus evaporation energy loss (solar + longwave)
    N = -(rho_w * cw * depth * (t_final - t0))/tstep # J

    # aerodynamic resistance, another magic function
    ra = (1/15) * np.log(zr / z0) ** 2 / (k * k * ut) # m-1

    # latent heat flux
    lambda_e = (
        delcalc(t_air - 273.15) * ((net_rad_t0) - (N))
        + rho_a * c_a * (vpdcalc(saturated_vapor_pressure(t_air), rh) / ra)
    ) / (
        delcalc(t_air - 273.15) + gamma
    )  # J
    
    # evaporation rate
    evap = (lambda_e) / lhv  # kg
    
    return evap
