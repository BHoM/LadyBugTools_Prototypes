"""The functions in this module describe the estimation of lift energy consumption within buildings."""

from enum import Enum, auto

import numpy as np
import pandas as pd

from .config import DATA_PATH, INDEX
from .enums import BuildingType, LiftEnergyEfficiency, LiftUsageIntensity


def annual_lift_usage_profile_generic() -> pd.Series:
    """Create a generic lift usage profile for a year, based on the average
    usage profile from

    Tukia, Toni, et al. 'Modeling the Aggregated Power Consumption of
    Elevators - the New York City Case Study'. Applied Energy, vol. 251,
    Oct. 2019, p. 113356. DOI.org (Crossref),
    https://doi.org/10.1016/j.apenergy.2019.113356.

    """
    LIFT_PROFILE = pd.read_csv(
        DATA_PATH / "lift_profile.csv",
        header=0,
        index_col=0,
    )
    usage = []
    for wkday in (
        pd.Series(np.ones(len(INDEX)), index=INDEX).resample("D").mean().index.weekday
    ):
        if wkday in [5, 6]:
            usage += LIFT_PROFILE["weekend"].values.tolist()
        else:
            usage += LIFT_PROFILE["weekday"].values.tolist()
    return pd.Series(usage, index=INDEX, name="Lift Usage Profile")


class LiftType(Enum):
    """From Al-Sharif, Lutfi. “Lift and Escalator Energy Consumption.” In Proceedings of the CIBSE/ASHRAE Joint National Conference, 1:231-39. Harrogate, UK, 1996."""

    Hydraulic = auto()
    Geared = auto()
    Gearless = auto()

    @classmethod
    def from_n_floors(cls, n_floors: int) -> "LiftType":
        """Determine the lift type based on the number of floors in the building."""
        if n_floors <= 6:
            return cls.Hydraulic
        if n_floors <= 17:
            return cls.Geared
        return cls.Gearless

    def average_trip_time(self) -> float:
        """The average trip time (s) for the lift type."""
        return {LiftType.Hydraulic: 6, LiftType.Geared: 8.5, LiftType.Gearless: 4.5}[
            self
        ]


def simple_estimate(n_lifts: int, motor_size: float = 45) -> pd.Series:
    """A very basic estimator based on number of lifts, usage profile and lift power demand.

    Args:
        n_lifts (int):
            The nuimber of lifts in the building.
        motor_size (float):
            The rated motor power, in kW. Default is 45
    """

    # get annual hourly usage profile
    profile = annual_lift_usage_profile_generic()

    return (profile * n_lifts * motor_size).rename("Lifts (kWh)")


def simple_estimate_from_storeys(n_storeys: int, motor_size: float = 45) -> pd.Series:
    """Using number of storeys, make a guess at lift type and number.

    TODO - approximate motor size by lioft type and storeys.
    """
    lift_type = LiftType.from_n_floors(n_storeys)

    # approximate number of lifts based on footprint area
    # this is a hack as there are standards for differnt buildings types
    # based on either area, number of units
    n_lifts = np.ceil(n_storeys / 2.5)  # 1-per every 2.5 floors
    if n_storeys > 10:
        n_lifts += 1  # add a service elevator for buldings over 10-storeys

    return simple_estimate(n_lifts=n_lifts, motor_size=motor_size)


def al_sharif_1996(
    n_occupants: pd.Series, n_storeys: int, building_footprint: float
) -> float:
    """Estimate lift energy consumptions based on

    Args:
        n_occupants (pd.Series): The number of occupants in the building, per hour.
        n_storeys (int): The number of storeys in the building.

    References:
        Al-Sharif, Lutfi. “Lift and Escalator Energy Consumption.” In
        Proceedings of the CIBSE/ASHRAE Joint National Conference, 1:231-39.
        Harrogate, UK, 1996.
    """

    # get the likely lift type based on the number of storeys
    lift_type = LiftType.from_n_floors(n_storeys)

    # assume motor size for individual lift
    motor_size = 45  # kW

    # assume loft speed
    lift_speed = 4  # m/s

    # get the average trip time from the lift type
    trip_time_tp = lift_type.average_trip_time()

    # approximate number of lifts based on footprint area
    # this is a hack as there are standards for differnt buildings types
    # based on either area, number of units
    n_lifts = np.ceil(n_storeys / 2.5)  # 1-per every 2.5 floors
    if n_storeys > 10:
        n_lifts += 1  # add a service elevator for buldings over 10-storeys

    # get the lift profile for the whole year
    profile = annual_lift_usage_profile_generic()

    # multiple n lifts by power demand and profuilke to get energy demand ... ruybbish method
    return None, None, profile * n_lifts * motor_size

    # apply profile to number of occupants
    occ_profile = profile * n_occupants

    # assume 3-people per lift on average
    occ_profile * 3

    # estimate the number of "starts per day" based on occupants and number of floors
    # Assume that occupancy is evenly spread, wiht lift capacity of 6-people

    return n_occupants, n_lifts, profile


def approximate_lift_energy_demand(
    number_of_storeys: float,
    occupants: int,
    lift_energy_efficiency: LiftEnergyEfficiency = LiftEnergyEfficiency.A,
    usage_intensity: LiftUsageIntensity = LiftUsageIntensity.Medium,
    floor_to_floor_height: float = 4,
    nominal_load: float = 400,
    nominal_speed: float = 0.2,
    number_of_person_per_trip: int = 1,
    number_of_trips_per_person: int = 2,
) -> pd.Series:
    """Based on the method defined within Passivhaus, estimate the annual 
    energy demand for lifts based on their efficiency, usage level and basic 
    usage asumptions.
    
    Args:
        number_of_storeys (float):
            The number of storeys in the building.
        occupants (int):
            The peak number of occupants within the building.
        lift_energy_efficiency (LiftEnergyEfficiency):
            The rating for the lift/s, which relates to a pre-defined power demand value.
        usage_intensity (LiftUsageIntensity):
            How frequently the lift/s are in use.
        floor_to_floor_height (float):
            The distance between floors (in m).
        nominal_load (float):
            The typical carrying capcity of the lift/s. 
            TODO - replace nominal load with peak occupant number.
        nominal_speed (float):
            The rated speed of movement of the lift/s.
        number_of_person_per_trip (int):
            The average number of persons within a single lift.
        number_of_trips_per_person (int):
            The average number of trips a person takes in a lift in a day.
    
    Return:
        pd.Series:
            A set of hourly values denoting the hourly energy demand for 
            lifts within a singular building.

    """

    if number_of_storeys <= 1:
        return pd.Series(index=INDEX, name="Lifts (kWh)", data=np.zeros(len(INDEX)))

    # get the max occupants within the building, and assume that 2/3 of tjhese use lifts
    total_number_of_user: int = np.ceil(occupants * 2 / 3)
    
    # based on the number of storeys, extimate the numebr of lifts
    number_of_lifts = np.ceil(number_of_storeys / 2.5)  # 1-per every 2.5 floors
    if number_of_lifts > 10:
        number_of_lifts += 1

    proportion_building_height_travelled = usage_intensity.percentage_average_travel_distance(n_floors=number_of_storeys)
    average_distance_per_trip = proportion_building_height_travelled * floor_to_floor_height * number_of_storeys

    # standby demand
    standby_demand = lift_energy_efficiency.standby_consumption()  # W

    # travel demand
    travel_demand = lift_energy_efficiency.operational_consumption()  # mWh/(kg.m)

    # usage intensity
    time_of_usage = usage_intensity.average_travel_time()
    time_of_standby = usage_intensity.average_standby_time()
    total_number_of_trips = (
        total_number_of_user * number_of_trips_per_person / number_of_person_per_trip
    )

    total_time_of_usage = (
        total_number_of_trips * average_distance_per_trip / nominal_speed / 3600
    )  # h/d
    total_distance = average_distance_per_trip * total_number_of_trips / 1000  # km/d
    trips_per_lift = total_number_of_trips / number_of_lifts  # trips/day
    time_of_usage_per_lift = total_time_of_usage / number_of_lifts
    if total_number_of_trips == 0:
        distance_per_lift = time_of_usage * 3600 * nominal_speed  # m/d
        daily_standby_demand_per_lift = standby_demand * time_of_standby / 1000  # kWh/d
    else:
        distance_per_lift = time_of_usage_per_lift * 3600 * nominal_speed
        daily_standby_demand_per_lift = (
            standby_demand * (24 - time_of_usage_per_lift) / 1000
        )
    daily_travel_demand_per_lift = (
        travel_demand * distance_per_lift * nominal_load / 1000000
    )
    daily_total_energy_demand_per_lift = (
        daily_travel_demand_per_lift + daily_standby_demand_per_lift
    )
    annual_energy_demand_kwh = (
        daily_standby_demand_per_lift * number_of_lifts * 365
    )  # kWh/a

    # load the annual lift energy demand profile
    profile = annual_lift_usage_profile_generic()

    return ((profile / profile.sum()) * annual_energy_demand_kwh).rename("Lifts (kWh)")
