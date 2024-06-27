"""Enums representing constants or symbolic values that have a clear and fixed 
set of options."""

# region: IMPORTS
# pylint: disable=E0401

from enum import Enum

from honeybee_energy.lib.constructionsets import (
    ConstructionSet, construction_set_by_identifier)
from honeybee_energy.lib.programtypes import (
    ProgramType, building_program_type_by_identifier,
    program_type_by_identifier)
from ladybug.epw import EPW

from .config import DEFAULT_SYSTEMS, logger

# pylint: enable=E0401
# endregion: IMPORTS


class BuildingType(Enum):
    """The type of building to simulate."""

    ACCOMODATION_APARTMENT_HIGHRISE = "HighriseApartment"
    CIVIC_CONCERT_HALL = "ConcertHall"
    ACCOMODATION_APARTMENT_MIDRISE = "MidriseApartment"
    ACCOMODATION_HOTEL_LARGE = "LargeHotel"
    ACCOMODATION_HOTEL_SMALL = "SmallHotel"
    ACCOMODATION_HOUSE_LOWRISE = "ResidentialLowRise"
    CIVIC_COURTHOUSE = "Courthouse"
    CIVIC_LIBRARY = "Library"
    COMMERCIAL_OFFICE_LARGE = "LargeOffice"
    COMMERCIAL_OFFICE_MEDIUM = "MediumOffice"
    COMMERCIAL_OFFICE_SMALL = "SmallOffice"
    COMMERCIAL_RESTAURANT_FULL_SERVICE = "FullServiceRestaurant"
    COMMERCIAL_RESTAURANT_QUICK_SERVICE = "QuickServiceRestaurant"
    COMMERCIAL_RETAIL = "Retail"
    COMMERCIAL_STRIP_MALL = "StripMall"
    COMMERCIAL_SUPERMARKET = "SuperMarket"
    DATACENTER_LARGE_HIGH_ITE = "LargeDataCenterHighITE"
    DATACENTER_LARGE_LOW_ITE = "LargeDataCenterLowITE"
    DATACENTER_SMALL_HIGH_ITE = "SmallDataCenterHighITE"
    DATACENTER_SMALL_LOW_ITE = "SmallDataCenterLowITE"
    EDUCATION_COLLEGE = "College"
    EDUCATION_SCHOOL_PRIMARY = "PrimarySchool"
    EDUCATION_SCHOOL_SECONDARY = "SecondarySchool"
    HEALTHCARE_HOSPITAL = "Hospital"
    HEALTHCARE_OUTPATIENT = "Outpatient"
    INDUSTRY_LIGHT = "LightIndustry"
    INDUSTRY_WAREHOUSE = "Warehouse"
    LABORATORY = "Laboratory"
    PARKING = "ParkingBasement"
    PHYSICAL_EVENTS = "PhysicalFitnessEvents"
    PHYSICAL_EXERCISE = "PhysicalFitnessExercise"
    RELIGIOUS = "Religious"


class ConstructionType(Enum):
    """Types of construction for the building envelope."""

    STEEL_FRAMED = "SteelFramed"
    WOOD_FRAMED = "WoodFramed"
    MASS = "Mass"
    METAL_BUILDING = "Metal Building"


class Vintage(Enum):
    """Vintage/historic standard of the building code."""

    ASHRAE_901_2004 = "2004"
    ASHRAE_901_2007 = "2007"
    ASHRAE_901_2010 = "2010"
    ASHRAE_901_2013 = "2013"
    ASHRAE_901_2016 = "2016"
    ASHRAE_901_2019 = "2019"
    CBECS_1980_2004 = "1980_2004"
    CBECS_PRE_1980 = "pre_1980"


class EconomizerType(Enum):
    """A type of air-side economizer for the building HVAC system."""

    NO_ECONOMIZER = "NoEconomizer"
    DIFFERENTIAL_DRY_BULB = "DifferentialDryBulb"
    DIFFERENTIAL_ENTHALPY = "DifferentialEnthalpy"


class TerrainType(Enum):
    """The type of terrain surrounding the building."""

    OCEAN = "Ocean"
    COUNTRY = "Country"
    SUBURBS = "Suburbs"
    URBAN = "Urban"
    CITY = "City"


def default_gfa(building_type: BuildingType) -> float:
    """Get the typical GFA the building type.

    Source: https://www.energy.gov/eere/buildings/commercial-reference-buildings

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        float:
            The typical GFA for the building type, in m2.
    """

    match building_type:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE:
            gfa = 12000
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE:
            gfa = 3100
        case BuildingType.EDUCATION_COLLEGE:
            gfa = 5000
        case BuildingType.CIVIC_COURTHOUSE:
            gfa = 5000
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            gfa = 8000
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            gfa = 8000
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            gfa = 2000
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            gfa = 2000
        case BuildingType.HEALTHCARE_HOSPITAL:
            gfa = 22000
        case BuildingType.ACCOMODATION_HOTEL_LARGE:
            gfa = 11000
        case BuildingType.ACCOMODATION_HOTEL_SMALL:
            gfa = 4000
        case BuildingType.LABORATORY:
            gfa = 1000
        case BuildingType.COMMERCIAL_OFFICE_LARGE:
            gfa = 45000
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM:
            gfa = 5000
        case BuildingType.COMMERCIAL_OFFICE_SMALL:
            gfa = 500
        case BuildingType.HEALTHCARE_OUTPATIENT:
            gfa = 3800
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE:
            gfa = 500
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE:
            gfa = 200
        case BuildingType.COMMERCIAL_RETAIL:
            gfa = 2300
        case BuildingType.EDUCATION_SCHOOL_PRIMARY:
            gfa = 6900
        case BuildingType.EDUCATION_SCHOOL_SECONDARY:
            gfa = 19600
        case BuildingType.COMMERCIAL_STRIP_MALL:
            gfa = 10000
        case BuildingType.COMMERCIAL_SUPERMARKET:
            gfa = 4200
        case BuildingType.INDUSTRY_WAREHOUSE:
            gfa = 4800
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE:
            gfa = 120
        case BuildingType.CIVIC_CONCERT_HALL:
            gfa = 18000
        case BuildingType.PHYSICAL_EXERCISE:
            gfa = 400
        case BuildingType.PHYSICAL_EVENTS:
            gfa = 29000
        case BuildingType.RELIGIOUS:
            gfa = 4000
        case BuildingType.INDUSTRY_LIGHT:
            gfa = 1500
        case BuildingType.PARKING:
            gfa = 3500
        case BuildingType.CIVIC_LIBRARY:
            gfa = 1200
        case _:
            raise ValueError(
                f"No default average footprint area is available for {building_type}."
            )

    return float(gfa)


def default_number_of_floors(building_type: BuildingType) -> float:
    """Get the typical number of floors for the building type.

    Source: https://www.energy.gov/eere/buildings/commercial-reference-buildings

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        float:
            The typical number of floors for the building type.
    """

    match building_type:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE:
            n_floors = 10
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE:
            n_floors = 4
        case BuildingType.EDUCATION_COLLEGE:
            n_floors = 2
        case BuildingType.CIVIC_COURTHOUSE:
            n_floors = 1
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            n_floors = 1
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            n_floors = 1
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            n_floors = 1
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            n_floors = 1
        case BuildingType.HEALTHCARE_HOSPITAL:
            n_floors = 3
        case BuildingType.ACCOMODATION_HOTEL_LARGE:
            n_floors = 5
        case BuildingType.ACCOMODATION_HOTEL_SMALL:
            n_floors = 3
        case BuildingType.LABORATORY:
            n_floors = 1
        case BuildingType.COMMERCIAL_OFFICE_LARGE:
            n_floors = 10
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM:
            n_floors = 3
        case BuildingType.COMMERCIAL_OFFICE_SMALL:
            n_floors = 1
        case BuildingType.HEALTHCARE_OUTPATIENT:
            n_floors = 1
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE:
            n_floors = 1
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE:
            n_floors = 1
        case BuildingType.COMMERCIAL_RETAIL:
            n_floors = 1
        case BuildingType.EDUCATION_SCHOOL_PRIMARY:
            n_floors = 1
        case BuildingType.EDUCATION_SCHOOL_SECONDARY:
            n_floors = 3
        case BuildingType.COMMERCIAL_STRIP_MALL:
            n_floors = 2
        case BuildingType.COMMERCIAL_SUPERMARKET:
            n_floors = 1
        case BuildingType.INDUSTRY_WAREHOUSE:
            n_floors = 1
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE:
            n_floors = 2
        case BuildingType.CIVIC_CONCERT_HALL:
            n_floors = 2
        case BuildingType.PHYSICAL_EXERCISE:
            n_floors = 1
        case BuildingType.PHYSICAL_EVENTS:
            n_floors = 1
        case BuildingType.RELIGIOUS:
            n_floors = 1
        case BuildingType.INDUSTRY_LIGHT:
            n_floors = 1
        case BuildingType.PARKING:
            n_floors = 2
        case BuildingType.CIVIC_LIBRARY:
            n_floors = 1
        case _:
            raise ValueError(
                f"No default average number of floors is available for {building_type}."
            )
    return int(n_floors)


def default_footprint_area(building_type: BuildingType) -> float:
    """Get the typical footprint area for the building type.

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        float:
            The typical footprint area for the building type, in m2.
    """

    return float(
        round(default_gfa(building_type) / default_number_of_floors(building_type), 0)
    )


def default_floor_height(building_type: BuildingType) -> float:
    """Get the typical floor to floor height for the building type.

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        float:
            The typical floor height for the building type, in m.
    """

    match building_type:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE:
            floor_height = 3.5
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE:
            floor_height = 3.5
        case BuildingType.EDUCATION_COLLEGE:
            floor_height = 3.8
        case BuildingType.CIVIC_COURTHOUSE:
            floor_height = 4
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            floor_height = 4
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            floor_height = 4
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            floor_height = 4
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            floor_height = 4
        case BuildingType.HEALTHCARE_HOSPITAL:
            floor_height = 3.8
        case BuildingType.ACCOMODATION_HOTEL_LARGE:
            floor_height = 3.8
        case BuildingType.ACCOMODATION_HOTEL_SMALL:
            floor_height = 3.8
        case BuildingType.LABORATORY:
            floor_height = 3.8
        case BuildingType.COMMERCIAL_OFFICE_LARGE:
            floor_height = 3.8
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM:
            floor_height = 3.8
        case BuildingType.COMMERCIAL_OFFICE_SMALL:
            floor_height = 3.8
        case BuildingType.HEALTHCARE_OUTPATIENT:
            floor_height = 3.8
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE:
            floor_height = 4
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE:
            floor_height = 4
        case BuildingType.COMMERCIAL_RETAIL:
            floor_height = 4
        case BuildingType.EDUCATION_SCHOOL_PRIMARY:
            floor_height = 4
        case BuildingType.EDUCATION_SCHOOL_SECONDARY:
            floor_height = 4
        case BuildingType.COMMERCIAL_STRIP_MALL:
            floor_height = 4.25
        case BuildingType.COMMERCIAL_SUPERMARKET:
            floor_height = 4.5
        case BuildingType.INDUSTRY_WAREHOUSE:
            floor_height = 4.5
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE:
            floor_height = 3.3
        case BuildingType.CIVIC_CONCERT_HALL:
            floor_height = 5
        case BuildingType.PHYSICAL_EXERCISE:
            floor_height = 3.5
        case BuildingType.PHYSICAL_EVENTS:
            floor_height = 5
        case BuildingType.RELIGIOUS:
            floor_height = 5
        case BuildingType.INDUSTRY_LIGHT:
            floor_height = 4.5
        case BuildingType.PARKING:
            floor_height = 3.2
        case BuildingType.CIVIC_LIBRARY:
            floor_height = 3.5
        case _:
            raise ValueError(
                f"No default average floor height is available for {building_type}."
            )
    return float(floor_height)


def default_construction_type(building_type: BuildingType) -> float:
    """Get the typical ConstructionType for the building type.

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        ConstructionType:
            The typical construction type for the building type.
    """

    match building_type:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE:
            constr_type = ConstructionType.MASS
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE:
            constr_type = ConstructionType.MASS
        case BuildingType.EDUCATION_COLLEGE:
            constr_type = ConstructionType.MASS
        case BuildingType.CIVIC_COURTHOUSE:
            constr_type = ConstructionType.MASS
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.HEALTHCARE_HOSPITAL:
            constr_type = ConstructionType.MASS
        case BuildingType.ACCOMODATION_HOTEL_LARGE:
            constr_type = ConstructionType.MASS
        case BuildingType.ACCOMODATION_HOTEL_SMALL:
            constr_type = ConstructionType.MASS
        case BuildingType.LABORATORY:
            constr_type = ConstructionType.MASS
        case BuildingType.COMMERCIAL_OFFICE_LARGE:
            constr_type = ConstructionType.MASS
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM:
            constr_type = ConstructionType.MASS
        case BuildingType.COMMERCIAL_OFFICE_SMALL:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.HEALTHCARE_OUTPATIENT:
            constr_type = ConstructionType.MASS
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.COMMERCIAL_RETAIL:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.EDUCATION_SCHOOL_PRIMARY:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.EDUCATION_SCHOOL_SECONDARY:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.COMMERCIAL_STRIP_MALL:
            constr_type = ConstructionType.MASS
        case BuildingType.COMMERCIAL_SUPERMARKET:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.INDUSTRY_WAREHOUSE:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE:
            constr_type = ConstructionType.WOOD_FRAMED
        case BuildingType.CIVIC_CONCERT_HALL:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.PHYSICAL_EXERCISE:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.PHYSICAL_EVENTS:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.RELIGIOUS:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.INDUSTRY_LIGHT:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.PARKING:
            constr_type = ConstructionType.MASS
        case BuildingType.CIVIC_LIBRARY:
            constr_type = ConstructionType.MASS
        case _:
            raise ValueError(
                f"No default construction type is available for {building_type}."
            )
    return constr_type


def default_glazing_ratio(building_type: BuildingType) -> float:
    """Get the typical glazing ratio (applied across all facades) for the building type.

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        float:
            The typical glazing ratio for the building type.
    """

    match building_type:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE:
            glazing_ratio = 0.3
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE:
            glazing_ratio = 0.3
        case BuildingType.EDUCATION_COLLEGE:
            glazing_ratio = 0.3
        case BuildingType.CIVIC_COURTHOUSE:
            glazing_ratio = 0.3
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            glazing_ratio = 0.05
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            glazing_ratio = 0.05
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            glazing_ratio = 0.05
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            glazing_ratio = 0.05
        case BuildingType.HEALTHCARE_HOSPITAL:
            glazing_ratio = 0.2
        case BuildingType.ACCOMODATION_HOTEL_LARGE:
            glazing_ratio = 0.2
        case BuildingType.ACCOMODATION_HOTEL_SMALL:
            glazing_ratio = 0.2
        case BuildingType.LABORATORY:
            glazing_ratio = 0.2
        case BuildingType.COMMERCIAL_OFFICE_LARGE:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_OFFICE_SMALL:
            glazing_ratio = 0.3
        case BuildingType.HEALTHCARE_OUTPATIENT:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_RETAIL:
            glazing_ratio = 0.3
        case BuildingType.EDUCATION_SCHOOL_PRIMARY:
            glazing_ratio = 0.3
        case BuildingType.EDUCATION_SCHOOL_SECONDARY:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_STRIP_MALL:
            glazing_ratio = 0.2
        case BuildingType.COMMERCIAL_SUPERMARKET:
            glazing_ratio = 0.05
        case BuildingType.INDUSTRY_WAREHOUSE:
            glazing_ratio = 0.05
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE:
            glazing_ratio = 0.4
        case BuildingType.CIVIC_CONCERT_HALL:
            glazing_ratio = 0.05
        case BuildingType.PHYSICAL_EXERCISE:
            glazing_ratio = 0.3
        case BuildingType.PHYSICAL_EVENTS:
            glazing_ratio = 0.05
        case BuildingType.RELIGIOUS:
            glazing_ratio = 0.3
        case BuildingType.INDUSTRY_LIGHT:
            glazing_ratio = 0.05
        case BuildingType.PARKING:
            glazing_ratio = 0.05
        case BuildingType.CIVIC_LIBRARY:
            glazing_ratio = 0.3
        case _:
            raise ValueError(
                f"No default glazing ratio is available for {building_type}."
            )
    return float(glazing_ratio)


def default_skylight_ratio(building_type: BuildingType) -> float:
    """Get the typical skilight ratio for the building type.

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        float:
            The typical skylight ratio for the building type.
    """

    match building_type:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE:
            skylight_ratio = 0
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE:
            skylight_ratio = 0
        case BuildingType.EDUCATION_COLLEGE:
            skylight_ratio = 0
        case BuildingType.CIVIC_COURTHOUSE:
            skylight_ratio = 0
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            skylight_ratio = 0
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            skylight_ratio = 0
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            skylight_ratio = 0
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            skylight_ratio = 0
        case BuildingType.HEALTHCARE_HOSPITAL:
            skylight_ratio = 0
        case BuildingType.ACCOMODATION_HOTEL_LARGE:
            skylight_ratio = 0
        case BuildingType.ACCOMODATION_HOTEL_SMALL:
            skylight_ratio = 0
        case BuildingType.LABORATORY:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_OFFICE_LARGE:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_OFFICE_SMALL:
            skylight_ratio = 0
        case BuildingType.HEALTHCARE_OUTPATIENT:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_RETAIL:
            skylight_ratio = 0
        case BuildingType.EDUCATION_SCHOOL_PRIMARY:
            skylight_ratio = 0
        case BuildingType.EDUCATION_SCHOOL_SECONDARY:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_STRIP_MALL:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_SUPERMARKET:
            skylight_ratio = 0
        case BuildingType.INDUSTRY_WAREHOUSE:
            skylight_ratio = 0
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE:
            skylight_ratio = 0
        case BuildingType.CIVIC_CONCERT_HALL:
            skylight_ratio = 0
        case BuildingType.PHYSICAL_EXERCISE:
            skylight_ratio = 0
        case BuildingType.PHYSICAL_EVENTS:
            skylight_ratio = 0
        case BuildingType.RELIGIOUS:
            skylight_ratio = 0
        case BuildingType.INDUSTRY_LIGHT:
            skylight_ratio = 0
        case BuildingType.PARKING:
            skylight_ratio = 0
        case BuildingType.CIVIC_LIBRARY:
            skylight_ratio = 0
        case _:
            raise ValueError(
                f"No default skylight ratio is available for {building_type}."
            )
    return float(skylight_ratio)


def default_context_shade_distance(terrain_type: TerrainType) -> float:
    """Get the distance to contextual geometry surrounding the building.

    Args:
        terrain_type (TerrainType):
            The type of terrain surrounding the building.

    Returns:
        float:
            The typical distance to contextual geometry surrounding the building, in m.
    """

    match terrain_type:
        case TerrainType.OCEAN:
            context_distance = 1500
        case TerrainType.COUNTRY:
            context_distance = 150
        case TerrainType.SUBURBS:
            context_distance = 50
        case TerrainType.URBAN:
            context_distance = 40
        case TerrainType.CITY:
            context_distance = 30
        case _:
            raise ValueError(
                f"No default context height is available for {terrain_type}."
            )
    return float(context_distance)


def default_program(building_type: BuildingType) -> ProgramType:
    """Get the typical ProgramType for the building type."""

    match building_type:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.EDUCATION_COLLEGE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.CIVIC_COURTHOUSE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.HEALTHCARE_HOSPITAL:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.ACCOMODATION_HOTEL_LARGE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.ACCOMODATION_HOTEL_SMALL:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.LABORATORY:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_OFFICE_LARGE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_OFFICE_SMALL:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.HEALTHCARE_OUTPATIENT:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_RETAIL:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.EDUCATION_SCHOOL_PRIMARY:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.EDUCATION_SCHOOL_SECONDARY:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_STRIP_MALL:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_SUPERMARKET:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.INDUSTRY_WAREHOUSE:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE:
            logger.info(
                "%s currently using %s default program.",
                building_type,
                BuildingType.ACCOMODATION_APARTMENT_MIDRISE,
            )
            program = building_program_type_by_identifier("MidriseApartment")
        case BuildingType.CIVIC_CONCERT_HALL:
            # from AECOM. Cost Model: New-Build Concert Halls. LM00093-0817-v2.0, July 2017.
            bld_mix_dict = {
                "2019::SecondarySchool::Auditorium": 0.20556,
                "2019::Courthouse::Entrance Lobby": 0.11111,
                "2019::QuickServiceRestaurant::Kitchen": 0.02778,
                "2019::Retail::Point_of_Sale": 0.01944,
                "2019::Courthouse::Restrooms": 0.01111,
                "2019::SecondarySchool::Cafeteria": 0.02222,
                "2019::Courthouse::Office": 0.06944,
                "2019::College::Media Center": 0.04444,
                "2019::Courthouse::Storage": 0.10278,
                "2019::Hospital::PhysTherapy": 0.05278,
                "2019::Courthouse::Corridor": 0.16667,
                "2019::Courthouse::Service Shaft": 0.02,
                "2019::SecondarySchool::Mechanical": 0.1,
                "2019::Courthouse::Plenum": 0.04667,
            }
            progs, ratios = [], []
            for key, val in bld_mix_dict.items():
                progs.append(program_type_by_identifier(key))
                ratios.append(val)
            program = ProgramType.average("ConcertHall", progs, ratios)
            program.lock()
        case BuildingType.PHYSICAL_EXERCISE:
            # from https://www.wbdg.org/space-types/physical-fitness-exercise-room'
            bld_mix_dict = {
                "2019::College::Entrance Lobby": 0.01307,
                "2019::Hospital::PhysTherapy": 0.13725,
                "2019::LargeOffice::Restroom": 0.0915,
                "2019::SecondarySchool::Gym": 0.70589,
                "2019::College::Storage": 0.05229,
            }
            progs, ratios = [], []
            for key, val in bld_mix_dict.items():
                progs.append(program_type_by_identifier(key))
                ratios.append(val)
            program = ProgramType.average("PhysicalExercise", progs, ratios)
            program.lock()
        case BuildingType.PHYSICAL_EVENTS:
            # from https://www.wbdg.org/space-types/auditorium
            bld_mix_dict = {
                "2019::Courthouse::Entrance Lobby": 0.19473,
                "2019::Courthouse::Storage": 0.0549,
                "2019::SecondarySchool::Cafeteria": 0.0244,
                "2019::SecondarySchool::Library": 0.0183,
                "2019::SecondarySchool::Auditorium": 0.43924,
                "2019::SecondarySchool::Gym": 0.14642,
                "2019::College::Media Center": 0.08541,
                "2019::Courthouse::Restrooms": 0.0366,
            }
            progs, ratios = [], []
            for key, val in bld_mix_dict.items():
                progs.append(program_type_by_identifier(key))
                ratios.append(val)
            program = ProgramType.average("PhysicalEvents", progs, ratios)
            program.lock()
        case BuildingType.RELIGIOUS:
            # from https://www.wbdg.org/space-types/place-worship
            bld_mix_dict = {
                "2019::Courthouse::Courtroom": 0.49690,
                "2019::Courthouse::Storage": 0.05797,
                "2019::Courthouse::Office": 0.10352,
                "2019::Courthouse::Utility": 0.16770,
                "2019::Courthouse::Corridor": 0.17391,
            }
            progs, ratios = [], []
            for key, val in bld_mix_dict.items():
                progs.append(program_type_by_identifier(key))
                ratios.append(val)
            program = ProgramType.average("Religious", progs, ratios)
            program.lock()
        case BuildingType.INDUSTRY_LIGHT:
            # from https://www.wbdg.org/space-types/place-worship
            bld_mix_dict = {
                "2019::Warehouse::Office": 0.01125,
                "2019::Warehouse::Bulk": 0.53622,
                "2019::SmallDataCenterLowITE::ComputerRoom": 0.34005,
                "2019::Warehouse::Fine": 0.11248,
            }
            progs, ratios = [], []
            for key, val in bld_mix_dict.items():
                progs.append(program_type_by_identifier(key))
                ratios.append(val)
            program = ProgramType.average("LightIndustry", progs, ratios)
            program.lock()
        case BuildingType.PARKING:
            # from https://www.wbdg.org/space-types/place-worship
            bld_mix_dict = {
                "2019::Courthouse::Parking": 0.95000,
                "2019::SuperMarket::Elec/MechRoom": 0.05000,
            }
            progs, ratios = [], []
            for key, val in bld_mix_dict.items():
                progs.append(program_type_by_identifier(key))
                ratios.append(val)
            program = ProgramType.average("Parking", progs, ratios)
            program.lock()
        case BuildingType.CIVIC_LIBRARY:
            #  from https://www.wbdg.org/space-types/library
            bld_mix_dict = {
                "2019::College::Entrance Lobby": 0.0594,
                "2019::Courthouse::Jury Deliberation": 0.02121,
                "2019::LargeOffice::PrintRoom": 0.09418,
                "2019::Courthouse::Office": 0.18244,
                "2019::College::Media Center": 0.1171,
                "2019::Courthouse::Library": 0.39839,
                "2019::College::Lounge": 0.11031,
                "2019::Courthouse::Utility": 0.01697,
            }
            progs, ratios = [], []
            for key, val in bld_mix_dict.items():
                progs.append(program_type_by_identifier(key))
                ratios.append(val)
            program = ProgramType.average("Library", progs, ratios)
            program.lock()
        case _:
            raise ValueError(f"No default program is available for {building_type}.")
    return program


def default_constructionset(
    construction_type: ConstructionType, epw: EPW, vintage: Vintage
) -> ConstructionSet:
    """Return the default construction set for the building type, age and climate."""
    id_string = f"{vintage.value}::ClimateZone{int(epw.ashrae_climate_zone[0])}::{construction_type.value}"
    cset = construction_set_by_identifier(construction_set_identifier=id_string)
    return cset


def default_economizer_type(
    building_type: BuildingType, epw: EPW, vintage: Vintage = Vintage.ASHRAE_901_2019
) -> EconomizerType:
    """Get the typical EconomizerType for the building type and vintage."""
    ashrae_climate = int(epw.ashrae_climate_zone[0])
    return EconomizerType[
        DEFAULT_SYSTEMS[
            (DEFAULT_SYSTEMS.building_type == building_type.name)
            & (DEFAULT_SYSTEMS.vintage == vintage.name)
            & (DEFAULT_SYSTEMS.ashrae_climate == ashrae_climate)
        ].squeeze()["economizer_type"]
    ]


def default_hr_effectiveness(
    building_type: BuildingType, epw: EPW, vintage: Vintage = Vintage.ASHRAE_901_2019
) -> tuple[float]:
    """Get the typical sensible and latent heat recovery effectiveness for the building type, vintage and climate."""
    ashrae_climate = int(epw.ashrae_climate_zone[0])
    # filter the dataframe to get the row that matches the building type, vintage and climate
    s = DEFAULT_SYSTEMS[
        (DEFAULT_SYSTEMS.building_type == building_type.name)
        & (DEFAULT_SYSTEMS.vintage == vintage.name)
        & (DEFAULT_SYSTEMS.ashrae_climate == ashrae_climate)
    ].squeeze()
    return (
        s["sensible_heat_recovery_effectiveness"],
        s["latent_heat_recovery_effectiveness"],
    )


def default_demand_controlled_ventilation(
    building_type: BuildingType, vintage: Vintage
) -> bool:
    """Get the typical demand controlled ventilation for the building type and vintage."""

    return DEFAULT_SYSTEMS[
        (DEFAULT_SYSTEMS.building_type == building_type.name)
        & (DEFAULT_SYSTEMS.vintage == vintage.name)
    ].squeeze()["demand_controlled_ventilation"].values[0]


def default_daylight_dimming(building_type: BuildingType, vintage: Vintage) -> bool:
    """Get the typical daylight dimming for the building type and vintage."""

    return DEFAULT_SYSTEMS[
        (DEFAULT_SYSTEMS.building_type == building_type.name)
        & (DEFAULT_SYSTEMS.vintage == vintage.name)
    ].squeeze()["daylight_dimming"].values[0]


def default_heating_cop(
    building_type: BuildingType, epw: EPW, vintage: Vintage
) -> float:
    """Get the typical heating COP for the building type, vintage and climate."""

    ashrae_climate = int(epw.ashrae_climate_zone[0])
    # filter the dataframe to get the row that matches the building type, vintage and climate
    s = DEFAULT_SYSTEMS[
        (DEFAULT_SYSTEMS.building_type == building_type.name)
        & (DEFAULT_SYSTEMS.vintage == vintage.name)
        & (DEFAULT_SYSTEMS.ashrae_climate == ashrae_climate)
    ].squeeze()
    return s["heating_cop"]


def default_cooling_eer(
    building_type: BuildingType, epw: EPW, vintage: Vintage
) -> float:
    """Get the typical cooling EER for the building type, vintage and climate."""

    ashrae_climate = int(epw.ashrae_climate_zone[0])
    # filter the dataframe to get the row that matches the building type, vintage and climate
    s = DEFAULT_SYSTEMS[
        (DEFAULT_SYSTEMS.building_type == building_type.name)
        & (DEFAULT_SYSTEMS.vintage == vintage.name)
        & (DEFAULT_SYSTEMS.ashrae_climate == ashrae_climate)
    ].squeeze()
    return s["cooling_eer"]


def default_fan_power(building_type: BuildingType, vintage: Vintage) -> float:
    """Get the typical fan power for the building type and vintage."""

    return DEFAULT_SYSTEMS[
        (DEFAULT_SYSTEMS.building_type == building_type.name)
        & (DEFAULT_SYSTEMS.vintage == vintage.name)
    ].squeeze()["fan_power"].values[0]


def default_pump_power(building_type: BuildingType, vintage: Vintage) -> float:
    """Get the typical pump power for the building type and vintage."""

    return DEFAULT_SYSTEMS[
        (DEFAULT_SYSTEMS.building_type == building_type.name)
        & (DEFAULT_SYSTEMS.vintage == vintage.name)
    ].squeeze()["pump_power"].values[0]
