"""..."""

# pylint: disable=too-few-public-methods, import-error
from enum import Enum, auto


class BuildingType(Enum):
    """The type of building to simulate."""

    APARTMENT_HIGHRISE = "HighriseApartment"
    APARTMENT_MIDRISE = "MidriseApartment"
    COLLEGE = "College"
    COURTHOUSE = "Courthouse"
    DATACENTER_LARGE_HIGH_ITE = "LargeDataCenterHighITE"
    DATACENTER_LARGE_LOW_ITE = "LargeDataCenterLowITE"
    DATACENTER_SMALL_HIGH_ITE = "SmallDataCenterHighITE"
    DATACENTER_SMALL_LOW_ITE = "SmallDataCenterLowITE"
    HOSPITAL = "Hospital"
    HOTEL_LARGE = "LargeHotel"
    HOTEL_SMALL = "SmallHotel"
    LABORATORY = "Laboratory"
    OFFICE_LARGE = "LargeOffice"
    OFFICE_MEDIUM = "MediumOffice"
    OFFICE_SMALL = "SmallOffice"
    OUTPATIENT = "Outpatient"
    RESTAURANT_FULL_SERVICE = "FullServiceRestaurant"
    RESTAURANT_QUICK_SERVICE = "QuickServiceRestaurant"
    RETAIL = "Retail"
    SCHOOL_PRIMARY = "PrimarySchool"
    SCHOOL_SECONDARY = "SecondarySchool"
    STRIP_MALL = "StripMall"
    SUPER_MARKET = "SuperMarket"
    WAREHOUSE = "Warehouse"
    RESIDENTIAL_LOWRISE = "ResidentialLowRise"
    CONCERT_HALL = "ConcertHall"
    PHYSICAL_FITNESS_EXERCISE = "PhysicalFitnessExercise"
    PHYSICAL_FITNESS_EVENTS = "PhysicalFitnessEvents"
    PLACE_OF_WORSHIP = "PlaceOfWorship"
    LIGHT_INDUSTRY = "LightIndustry"
    PARKING_BASEMENT = "ParkingBasement"
    LIBRARY = "Library"


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


class TerrainType(Enum):
    """The type of terrain surrounding the building."""

    OCEAN = "Ocean"
    COUNTRY = "Country"
    SUBURBS = "Suburbs"
    URBAN = "Urban"
    CITY = "City"


def typical_context_distance(terrain_type: TerrainType) -> float:
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
    return context_distance


class EconomizerType(Enum):
    """A type of air-side economizer for the building HVAC system."""

    NO_ECONOMIZER = "NoEconomizer"
    DIFFERENTIAL_DRY_BULB = "DifferentialDryBulb"
    DIFFERENTIAL_ENTHALPY = "DifferentialEnthalpy"


class BuildingForm(Enum):
    """The form of the building to simulate."""

    CUBOID = auto()
    L_SHAPED = auto()
    U_SHAPED = auto()


def typical_gfa(building_type: BuildingType) -> float:
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
        case BuildingType.APARTMENT_HIGHRISE:
            gfa = 12000
        case BuildingType.APARTMENT_MIDRISE:
            gfa = 3100
        case BuildingType.COLLEGE:
            gfa = 5000
        case BuildingType.COURTHOUSE:
            gfa = 5000
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            gfa = 8000
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            gfa = 8000
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            gfa = 2000
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            gfa = 2000
        case BuildingType.HOSPITAL:
            gfa = 22000
        case BuildingType.HOTEL_LARGE:
            gfa = 11000
        case BuildingType.HOTEL_SMALL:
            gfa = 4000
        case BuildingType.LABORATORY:
            gfa = 1000
        case BuildingType.OFFICE_LARGE:
            gfa = 45000
        case BuildingType.OFFICE_MEDIUM:
            gfa = 5000
        case BuildingType.OFFICE_SMALL:
            gfa = 500
        case BuildingType.OUTPATIENT:
            gfa = 3800
        case BuildingType.RESTAURANT_FULL_SERVICE:
            gfa = 500
        case BuildingType.RESTAURANT_QUICK_SERVICE:
            gfa = 200
        case BuildingType.RETAIL:
            gfa = 2300
        case BuildingType.SCHOOL_PRIMARY:
            gfa = 6900
        case BuildingType.SCHOOL_SECONDARY:
            gfa = 19600
        case BuildingType.STRIP_MALL:
            gfa = 10000
        case BuildingType.SUPER_MARKET:
            gfa = 4200
        case BuildingType.WAREHOUSE:
            gfa = 4800
        case BuildingType.RESIDENTIAL_LOWRISE:
            gfa = 120
        case BuildingType.CONCERT_HALL:
            gfa = 18000
        case BuildingType.PHYSICAL_FITNESS_EXERCISE:
            gfa = 400
        case BuildingType.PHYSICAL_FITNESS_EVENTS:
            gfa = 29000
        case BuildingType.PLACE_OF_WORSHIP:
            gfa = 4000
        case BuildingType.LIGHT_INDUSTRY:
            gfa = 1500
        case BuildingType.PARKING_BASEMENT:
            gfa = 3500
        case BuildingType.LIBRARY:
            gfa = 1200
        case _:
            raise ValueError(
                f"No default average footprint area is available for {building_type}."
            )

    return gfa


def typical_num_floors(building_type: BuildingType) -> float:
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
        case BuildingType.APARTMENT_HIGHRISE:
            n_floors = 10
        case BuildingType.APARTMENT_MIDRISE:
            n_floors = 4
        case BuildingType.COLLEGE:
            n_floors = 2
        case BuildingType.COURTHOUSE:
            n_floors = 1
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            n_floors = 1
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            n_floors = 1
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            n_floors = 1
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            n_floors = 1
        case BuildingType.HOSPITAL:
            n_floors = 3
        case BuildingType.HOTEL_LARGE:
            n_floors = 5
        case BuildingType.HOTEL_SMALL:
            n_floors = 3
        case BuildingType.LABORATORY:
            n_floors = 1
        case BuildingType.OFFICE_LARGE:
            n_floors = 10
        case BuildingType.OFFICE_MEDIUM:
            n_floors = 3
        case BuildingType.OFFICE_SMALL:
            n_floors = 1
        case BuildingType.OUTPATIENT:
            n_floors = 1
        case BuildingType.RESTAURANT_FULL_SERVICE:
            n_floors = 1
        case BuildingType.RESTAURANT_QUICK_SERVICE:
            n_floors = 1
        case BuildingType.RETAIL:
            n_floors = 1
        case BuildingType.SCHOOL_PRIMARY:
            n_floors = 1
        case BuildingType.SCHOOL_SECONDARY:
            n_floors = 3
        case BuildingType.STRIP_MALL:
            n_floors = 2
        case BuildingType.SUPER_MARKET:
            n_floors = 1
        case BuildingType.WAREHOUSE:
            n_floors = 1
        case BuildingType.RESIDENTIAL_LOWRISE:
            n_floors = 2
        case BuildingType.CONCERT_HALL:
            n_floors = 2
        case BuildingType.PHYSICAL_FITNESS_EXERCISE:
            n_floors = 1
        case BuildingType.PHYSICAL_FITNESS_EVENTS:
            n_floors = 1
        case BuildingType.PLACE_OF_WORSHIP:
            n_floors = 1
        case BuildingType.LIGHT_INDUSTRY:
            n_floors = 1
        case BuildingType.PARKING_BASEMENT:
            n_floors = 2
        case BuildingType.LIBRARY:
            n_floors = 1
        case _:
            raise ValueError(
                f"No default average number of floors is available for {building_type}."
            )
    return n_floors


def typical_footprint_area(building_type: BuildingType) -> float:
    """Get the typical footprint area for the building type.

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        float:
            The typical footprint area for the building type, in m2.
    """

    return round(typical_gfa(building_type) / typical_num_floors(building_type), 0)


def typical_floor_height(building_type: BuildingType) -> float:
    """Get the typical floor to floor height for the building type.

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        float:
            The typical floor height for the building type, in m.
    """

    match building_type:
        case BuildingType.APARTMENT_HIGHRISE:
            floor_height = 3.5
        case BuildingType.APARTMENT_MIDRISE:
            floor_height = 3.5
        case BuildingType.COLLEGE:
            floor_height = 3.8
        case BuildingType.COURTHOUSE:
            floor_height = 4
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            floor_height = 4
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            floor_height = 4
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            floor_height = 4
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            floor_height = 4
        case BuildingType.HOSPITAL:
            floor_height = 3.8
        case BuildingType.HOTEL_LARGE:
            floor_height = 3.8
        case BuildingType.HOTEL_SMALL:
            floor_height = 3.8
        case BuildingType.LABORATORY:
            floor_height = 3.8
        case BuildingType.OFFICE_LARGE:
            floor_height = 3.8
        case BuildingType.OFFICE_MEDIUM:
            floor_height = 3.8
        case BuildingType.OFFICE_SMALL:
            floor_height = 3.8
        case BuildingType.OUTPATIENT:
            floor_height = 3.8
        case BuildingType.RESTAURANT_FULL_SERVICE:
            floor_height = 4
        case BuildingType.RESTAURANT_QUICK_SERVICE:
            floor_height = 4
        case BuildingType.RETAIL:
            floor_height = 4
        case BuildingType.SCHOOL_PRIMARY:
            floor_height = 4
        case BuildingType.SCHOOL_SECONDARY:
            floor_height = 4
        case BuildingType.STRIP_MALL:
            floor_height = 4.25
        case BuildingType.SUPER_MARKET:
            floor_height = 4.5
        case BuildingType.WAREHOUSE:
            floor_height = 4.5
        case BuildingType.RESIDENTIAL_LOWRISE:
            floor_height = 3.3
        case BuildingType.CONCERT_HALL:
            floor_height = 5
        case BuildingType.PHYSICAL_FITNESS_EXERCISE:
            floor_height = 3.5
        case BuildingType.PHYSICAL_FITNESS_EVENTS:
            floor_height = 5
        case BuildingType.PLACE_OF_WORSHIP:
            floor_height = 5
        case BuildingType.LIGHT_INDUSTRY:
            floor_height = 4.5
        case BuildingType.PARKING_BASEMENT:
            floor_height = 3.2
        case BuildingType.LIBRARY:
            floor_height = 3.5
        case _:
            raise ValueError(
                f"No default average floor height is available for {building_type}."
            )
    return floor_height


def typical_construction_type(building_type: BuildingType) -> float:
    """Get the typical construction_type for the building type.

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        ConstructionType:
            The typical construction type for the building type.
    """

    match building_type:
        case BuildingType.APARTMENT_HIGHRISE:
            constr_type = ConstructionType.MASS
        case BuildingType.APARTMENT_MIDRISE:
            constr_type = ConstructionType.MASS
        case BuildingType.COLLEGE:
            constr_type = ConstructionType.MASS
        case BuildingType.COURTHOUSE:
            constr_type = ConstructionType.MASS
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.HOSPITAL:
            constr_type = ConstructionType.MASS
        case BuildingType.HOTEL_LARGE:
            constr_type = ConstructionType.MASS
        case BuildingType.HOTEL_SMALL:
            constr_type = ConstructionType.MASS
        case BuildingType.LABORATORY:
            constr_type = ConstructionType.MASS
        case BuildingType.OFFICE_LARGE:
            constr_type = ConstructionType.MASS
        case BuildingType.OFFICE_MEDIUM:
            constr_type = ConstructionType.MASS
        case BuildingType.OFFICE_SMALL:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.OUTPATIENT:
            constr_type = ConstructionType.MASS
        case BuildingType.RESTAURANT_FULL_SERVICE:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.RESTAURANT_QUICK_SERVICE:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.RETAIL:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.SCHOOL_PRIMARY:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.SCHOOL_SECONDARY:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.STRIP_MALL:
            constr_type = ConstructionType.MASS
        case BuildingType.SUPER_MARKET:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.WAREHOUSE:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.RESIDENTIAL_LOWRISE:
            constr_type = ConstructionType.WOOD_FRAMED
        case BuildingType.CONCERT_HALL:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.PHYSICAL_FITNESS_EXERCISE:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.PHYSICAL_FITNESS_EVENTS:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.PLACE_OF_WORSHIP:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.LIGHT_INDUSTRY:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.PARKING_BASEMENT:
            constr_type = ConstructionType.MASS
        case BuildingType.LIBRARY:
            constr_type = ConstructionType.MASS
        case _:
            raise ValueError(
                f"No default construction type is available for {building_type}."
            )
    return constr_type


def typical_glazing_ratio(building_type: BuildingType) -> float:
    """Get the typical glazing ratio (applied across all facades) for the building type.

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        float:
            The typical glazing ratio for the building type.
    """

    match building_type:
        case BuildingType.APARTMENT_HIGHRISE:
            glazing_ratio = 0.3
        case BuildingType.APARTMENT_MIDRISE:
            glazing_ratio = 0.3
        case BuildingType.COLLEGE:
            glazing_ratio = 0.3
        case BuildingType.COURTHOUSE:
            glazing_ratio = 0.3
        case BuildingType.DATACENTER_LARGE_HIGH_ITE:
            glazing_ratio = 0.05
        case BuildingType.DATACENTER_LARGE_LOW_ITE:
            glazing_ratio = 0.05
        case BuildingType.DATACENTER_SMALL_HIGH_ITE:
            glazing_ratio = 0.05
        case BuildingType.DATACENTER_SMALL_LOW_ITE:
            glazing_ratio = 0.05
        case BuildingType.HOSPITAL:
            glazing_ratio = 0.2
        case BuildingType.HOTEL_LARGE:
            glazing_ratio = 0.2
        case BuildingType.HOTEL_SMALL:
            glazing_ratio = 0.2
        case BuildingType.LABORATORY:
            glazing_ratio = 0.2
        case BuildingType.OFFICE_LARGE:
            glazing_ratio = 0.3
        case BuildingType.OFFICE_MEDIUM:
            glazing_ratio = 0.3
        case BuildingType.OFFICE_SMALL:
            glazing_ratio = 0.3
        case BuildingType.OUTPATIENT:
            glazing_ratio = 0.3
        case BuildingType.RESTAURANT_FULL_SERVICE:
            glazing_ratio = 0.3
        case BuildingType.RESTAURANT_QUICK_SERVICE:
            glazing_ratio = 0.3
        case BuildingType.RETAIL:
            glazing_ratio = 0.3
        case BuildingType.SCHOOL_PRIMARY:
            glazing_ratio = 0.3
        case BuildingType.SCHOOL_SECONDARY:
            glazing_ratio = 0.3
        case BuildingType.STRIP_MALL:
            glazing_ratio = 0.2
        case BuildingType.SUPER_MARKET:
            glazing_ratio = 0.05
        case BuildingType.WAREHOUSE:
            glazing_ratio = 0.05
        case BuildingType.RESIDENTIAL_LOWRISE:
            glazing_ratio = 0.4
        case BuildingType.CONCERT_HALL:
            glazing_ratio = 0.05
        case BuildingType.PHYSICAL_FITNESS_EXERCISE:
            glazing_ratio = 0.3
        case BuildingType.PHYSICAL_FITNESS_EVENTS:
            glazing_ratio = 0.05
        case BuildingType.PLACE_OF_WORSHIP:
            glazing_ratio = 0.3
        case BuildingType.LIGHT_INDUSTRY:
            glazing_ratio = 0.05
        case BuildingType.PARKING_BASEMENT:
            glazing_ratio = 0.05
        case BuildingType.LIBRARY:
            glazing_ratio = 0.3
        case _:
            raise ValueError(
                f"No default glazing ratio is available for {building_type}."
            )
    return glazing_ratio
