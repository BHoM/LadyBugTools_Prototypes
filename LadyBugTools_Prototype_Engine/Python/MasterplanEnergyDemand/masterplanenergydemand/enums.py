"""Enums representing constants or symbolic values that have a clear and fixed 
set of options."""

# pylint: disable E1101

# region: IMPORTS
# pylint: disable=E0401

import json
from enum import Enum, auto
from pathlib import Path

import numpy as np
from honeybee_energy.config import folders as hbe_folders
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.lib.constructionsets import (
    ConstructionSet, construction_set_by_identifier)
from honeybee_energy.lib.programtypes import (
    ProgramType, building_program_type_by_identifier,
    program_type_by_identifier)
from ladybug.epw import EPW

from .config import DEFAULT_SYSTEMS, logger

# pylint: enable=E0401
# endregion: IMPORTS

# TODO - convert these methods to be properties of the enums ... definitily .. it'll be neater

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
    EXHIBITION = "Exhibition"


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


class AllAirHVACSystemType(Enum):
    """A set of all-air HVAC system types."""
    VAV_CHILLER_WITH_GAS_BOILER_REHEAT = "VAV_Chiller_Boiler"
    VAV_CHILLER_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP_REHEAT = "VAV_Chiller_ASHP"
    VAV_CHILLER_WITH_DISTRICT_HOT_WATER_REHEAT = "VAV_Chiller_DHW"
    VAV_CHILLER_WITH_PFP_BOXES = "VAV_Chiller_PFP"
    VAV_CHILLER_WITH_GAS_COIL_REHEAT = "VAV_Chiller_GasCoil"
    VAV_AIR_COOLED_CHILLER_WITH_GAS_BOILER_REHEAT = "VAV_ACChiller_Boiler"
    VAV_AIR_COOLED_CHILLER_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP_REHEAT = "VAV_ACChiller_ASHP"
    VAV_AIR_COOLED_CHILLER_WITH_DISTRICT_HOT_WATER_REHEAT = "VAV_ACChiller_DHW"
    VAV_AIR_COOLED_CHILLER_WITH_PFP_BOXES = "VAV_ACChiller_PFP"
    VAV_AIR_COOLED_CHILLER_WITH_GAS_COIL_REHEAT = "VAV_ACChiller_GasCoil"
    VAV_DISTRICT_CHILLED_WATER_WITH_GAS_BOILER_REHEAT = "VAV_DCW_Boiler"
    VAV_DISTRICT_CHILLED_WATER_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP_REHEAT = "VAV_DCW_ASHP"
    VAV_DISTRICT_CHILLED_WATER_WITH_DISTRICT_HOT_WATER_REHEAT = "VAV_DCW_DHW"
    VAV_DISTRICT_CHILLED_WATER_WITH_PFP_BOXES = "VAV_DCW_PFP"
    VAV_DISTRICT_CHILLED_WATER_WITH_GAS_COIL_REHEAT = "VAV_DCW_GasCoil"
    PVAV_WITH_GAS_BOILER_REHEAT = "PVAV_Boiler"
    PVAV_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP_REHEAT = "PVAV_ASHP"
    PVAV_WITH_DISTRICT_HOT_WATER_REHEAT = "PVAV_DHW"
    PVAV_WITH_PFP_BOXES = "PVAV_PFP"
    PVAV_WITH_GAS_HEAT_WITH_ELECTRIC_REHEAT = "PVAV_BoilerElectricReheat"
    PSZ_AC_WITH_BASEBOARD_ELECTRIC = "PSZAC_ElectricBaseboard"
    PSZ_AC_WITH_BASEBOARD_GAS_BOILER = "PSZAC_BoilerBaseboard"
    PSZ_AC_WITH_BASEBOARD_DISTRICT_HOT_WATER = "PSZAC_DHWBaseboard"
    PSZ_AC_WITH_GAS_UNIT_HEATERS = "PSZAC_GasHeaters"
    PSZ_AC_WITH_ELECTRIC_COIL = "PSZAC_ElectricCoil"
    PSZ_AC_WITH_GAS_COIL = "PSZAC_GasCoil"
    PSZ_AC_WITH_GAS_BOILER = "PSZAC_Boiler"
    PSZ_AC_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP = "PSZAC_ASHP"
    PSZ_AC_WITH_DISTRICT_HOT_WATER = "PSZAC_DHW"
    PSZ_AC_WITH_NO_HEAT = "PSZAC"
    PSZ_AC_DISTRICT_CHILLED_WATER_WITH_BASEBOARD_ELECTRIC = "PSZAC_DCW_ElectricBaseboard"
    PSZ_AC_DISTRICT_CHILLED_WATER_WITH_BASEBOARD_GAS_BOILER = "PSZAC_DCW_BoilerBaseboard"
    PSZ_AC_DISTRICT_CHILLED_WATER_WITH_GAS_UNIT_HEATERS = "PSZAC_DCW_GasHeaters"
    PSZ_AC_DISTRICT_CHILLED_WATER_WITH_ELECTRIC_COIL = "PSZAC_DCW_ElectricCoil"
    PSZ_AC_DISTRICT_CHILLED_WATER_WITH_GAS_COIL = "PSZAC_DCW_GasCoil"
    PSZ_AC_DISTRICT_CHILLED_WATER_WITH_GAS_BOILER = "PSZAC_DCW_Boiler"
    PSZ_AC_DISTRICT_CHILLED_WATER_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP = "PSZAC_DCW_ASHP"
    PSZ_AC_DISTRICT_CHILLED_WATER_WITH_DISTRICT_HOT_WATER = "PSZAC_DCW_DHW"
    PSZ_AC_DISTRICT_CHILLED_WATER_WITH_NO_HEAT = "PSZAC_DCW"
    PSZ_HP = "PSZHP"
    PTAC_WITH_BASEBOARD_ELECTRIC = "PTAC_ElectricBaseboard"
    PTAC_WITH_BASEBOARD_GAS_BOILER = "PTAC_BoilerBaseboard"
    PTAC_WITH_BASEBOARD_DISTRICT_HOT_WATER = "PTAC_DHWBaseboard"
    PTAC_WITH_GAS_UNIT_HEATERS = "PTAC_GasHeaters"
    PTAC_WITH_ELECTRIC_COIL = "PTAC_ElectricCoil"
    PTAC_WITH_GAS_COIL = "PTAC_GasCoil"
    PTAC_WITH_GAS_BOILER = "PTAC_Boiler"
    PTAC_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP = "PTAC_ASHP"
    PTAC_WITH_DISTRICT_HOT_WATER = "PTAC_DHW"
    PTAC_WITH_NO_HEAT = "PTAC"
    PTHP = "PTHP"
    FORCED_AIR_FURNACE = "Furnace"
    FORCED_AIR_ELECTRIC_FURNACE = "Furnace_Electric"

class DOASHVACSystemType(Enum):
    """A set of DOAS HVAC system types."""
    DOAS_WITH_FAN_COIL_CHILLER_WITH_BOILER = "DOAS_FCU_Chiller_Boiler"
    DOAS_WITH_FAN_COIL_CHILLER_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP = "DOAS_FCU_Chiller_ASHP"
    DOAS_WITH_FAN_COIL_CHILLER_WITH_DISTRICT_HOT_WATER = "DOAS_FCU_Chiller_DHW"
    DOAS_WITH_FAN_COIL_CHILLER_WITH_BASEBOARD_ELECTRIC = "DOAS_FCU_Chiller_ElectricBaseboard"
    DOAS_WITH_FAN_COIL_CHILLER_WITH_GAS_UNIT_HEATERS = "DOAS_FCU_Chiller_GasHeaters"
    DOAS_WITH_FAN_COIL_CHILLER_WITH_NO_HEAT = "DOAS_FCU_Chiller"
    DOAS_WITH_FAN_COIL_AIR_COOLED_CHILLER_WITH_BOILER = "DOAS_FCU_ACChiller_Boiler"
    DOAS_WITH_FAN_COIL_AIR_COOLED_CHILLER_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP = "DOAS_FCU_ACChiller_ASHP"
    DOAS_WITH_FAN_COIL_AIR_COOLED_CHILLER_WITH_DISTRICT_HOT_WATER = "DOAS_FCU_ACChiller_DHW"
    DOAS_WITH_FAN_COIL_AIR_COOLED_CHILLER_WITH_BASEBOARD_ELECTRIC = "DOAS_FCU_ACChiller_ElectricBaseboard"
    DOAS_WITH_FAN_COIL_AIR_COOLED_CHILLER_WITH_GAS_UNIT_HEATERS = "DOAS_FCU_ACChiller_GasHeaters"
    DOAS_WITH_FAN_COIL_AIR_COOLED_CHILLER_WITH_NO_HEAT = "DOAS_FCU_ACChiller"
    DOAS_WITH_FAN_COIL_DISTRICT_CHILLED_WATER_WITH_BOILER = "DOAS_FCU_DCW_Boiler"
    DOAS_WITH_FAN_COIL_DISTRICT_CHILLED_WATER_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP = "DOAS_FCU_DCW_ASHP"
    DOAS_WITH_FAN_COIL_DISTRICT_CHILLED_WATER_WITH_DISTRICT_HOT_WATER = "DOAS_FCU_DCW_DHW"
    DOAS_WITH_FAN_COIL_DISTRICT_CHILLED_WATER_WITH_BASEBOARD_ELECTRIC = "DOAS_FCU_DCW_ElectricBaseboard"
    DOAS_WITH_FAN_COIL_DISTRICT_CHILLED_WATER_WITH_GAS_UNIT_HEATERS = "DOAS_FCU_DCW_GasHeaters"
    DOAS_WITH_FAN_COIL_DISTRICT_CHILLED_WATER_WITH_NO_HEAT = "DOAS_FCU_DCW"
    DOAS_WITH_VRF = "DOAS_VRF"
    DOAS_WITH_WATER_SOURCE_HEAT_PUMPS_FLUID_COOLER_WITH_BOILER = "DOAS_WSHP_FluidCooler_Boiler"
    DOAS_WITH_WATER_SOURCE_HEAT_PUMPS_COOLING_TOWER_WITH_BOILER = "DOAS_WSHP_CoolingTower_Boiler"
    DOAS_WITH_WATER_SOURCE_HEAT_PUMPS_WITH_GROUND_SOURCE_HEAT_PUMP = "DOAS_WSHP_GSHP"
    DOAS_WITH_WATER_SOURCE_HEAT_PUMPS_DISTRICT_CHILLED_WATER_WITH_DISTRICT_HOT_WATER = "DOAS_WSHP_DCW_DHW"
    DOAS_WITH_LOW_TEMPERATURE_RADIANT_CHILLER_WITH_BOILER = "DOAS_Radiant_Chiller_Boiler"
    DOAS_WITH_LOW_TEMPERATURE_RADIANT_CHILLER_WITH_AIR_SOURCE_HEAT_PUMP = "DOAS_Radiant_Chiller_ASHP"
    DOAS_WITH_LOW_TEMPERATURE_RADIANT_CHILLER_WITH_DISTRICT_HOT_WATER = "DOAS_Radiant_Chiller_DHW"
    DOAS_WITH_LOW_TEMPERATURE_RADIANT_AIR_COOLED_CHILLER_WITH_BOILER = "DOAS_Radiant_ACChiller_Boiler"
    DOAS_WITH_LOW_TEMPERATURE_RADIANT_AIR_COOLED_CHILLER_WITH_DISTRICT_HOT_WATER = "DOAS_Radiant_ACChiller_DHW"
    DOAS_WITH_LOW_TEMPERATURE_RADIANT_DISTRICT_CHILLED_WATER_WITH_BOILER = "DOAS_Radiant_DCW_Boiler"
    DOAS_WITH_LOW_TEMPERATURE_RADIANT_DISTRICT_CHILLED_WATER_WITH_AIR_SOURCE_HEAT_PUMP = "DOAS_Radiant_DCW_ASHP"
    DOAS_WITH_LOW_TEMPERATURE_RADIANT_DISTRICT_CHILLED_WATER_WITH_DISTRICT_HOT_WATER = "DOAS_Radiant_DCW_DHW"

class HeatCoolNoVentHVACSystemType(Enum):
    """A set of Heating & Cooling only (no ventilation) HVAC system types."""
    BASEBOARD_ELECTRIC = "ElectricBaseboard"
    BASEBOARD_GAS_BOILER = "BoilerBaseboard"
    BASEBOARD_CENTRAL_AIR_SOURCE_HEAT_PUMP = "ASHPBaseboard"
    BASEBOARD_DISTRICT_HOT_WATER = "DHWBaseboard"
    DIRECT_EVAP_COOLERS_WITH_BASEBOARD_ELECTRIC = "EvapCoolers_ElectricBaseboard"
    DIRECT_EVAP_COOLERS_WITH_BASEBOARD_GAS_BOILER = "EvapCoolers_BoilerBaseboard"
    DIRECT_EVAP_COOLERS_WITH_BASEBOARD_CENTRAL_AIR_SOURCE_HEAT_PUMP = "EvapCoolers_ASHPBaseboard"
    DIRECT_EVAP_COOLERS_WITH_BASEBOARD_DISTRICT_HOT_WATER = "EvapCoolers_DHWBaseboard"
    DIRECT_EVAP_COOLERS_WITH_FORCED_AIR_FURNACE = "EvapCoolers_Furnace"
    DIRECT_EVAP_COOLERS_WITH_GAS_UNIT_HEATERS = "EvapCoolers_UnitHeaters"
    DIRECT_EVAP_COOLERS_WITH_NO_HEAT = "EvapCoolers"
    FAN_COIL_CHILLER_WITH_BOILER = "FCU_Chiller_Boiler"
    FAN_COIL_CHILLER_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP = "FCU_Chiller_ASHP"
    FAN_COIL_CHILLER_WITH_DISTRICT_HOT_WATER = "FCU_Chiller_DHW"
    FAN_COIL_CHILLER_WITH_BASEBOARD_ELECTRIC = "FCU_Chiller_ElectricBaseboard"
    FAN_COIL_CHILLER_WITH_GAS_UNIT_HEATERS = "FCU_Chiller_GasHeaters"
    FAN_COIL_CHILLER_WITH_NO_HEAT = "FCU_Chiller"
    FAN_COIL_AIR_COOLED_CHILLER_WITH_BOILER = "FCU_ACChiller_Boiler"
    FAN_COIL_AIR_COOLED_CHILLER_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP = "FCU_ACChiller_ASHP"
    FAN_COIL_AIR_COOLED_CHILLER_WITH_DISTRICT_HOT_WATER = "FCU_ACChiller_DHW"
    FAN_COIL_AIR_COOLED_CHILLER_WITH_BASEBOARD_ELECTRIC = "FCU_ACChiller_ElectricBaseboard"
    FAN_COIL_AIR_COOLED_CHILLER_WITH_GAS_UNIT_HEATERS = "FCU_ACChiller_GasHeaters"
    FAN_COIL_AIR_COOLED_CHILLER_WITH_NO_HEAT = "FCU_ACChiller"
    FAN_COIL_DISTRICT_CHILLED_WATER_WITH_BOILER = "FCU_DCW_Boiler"
    FAN_COIL_DISTRICT_CHILLED_WATER_WITH_CENTRAL_AIR_SOURCE_HEAT_PUMP = "FCU_DCW_ASHP"
    FAN_COIL_DISTRICT_CHILLED_WATER_WITH_DISTRICT_HOT_WATER = "FCU_DCW_DHW"
    FAN_COIL_DISTRICT_CHILLED_WATER_WITH_BASEBOARD_ELECTRIC = "FCU_DCW_ElectricBaseboard"
    FAN_COIL_DISTRICT_CHILLED_WATER_WITH_GAS_UNIT_HEATERS = "FCU_DCW_GasHeaters"
    FAN_COIL_DISTRICT_CHILLED_WATER_WITH_NO_HEAT = "FCU_DCW"
    GAS_UNIT_HEATERS = "GasHeaters"
    RESIDENTIAL_AC_WITH_BASEBOARD_ELECTRIC = "ResidentialAC_ElectricBaseboard"
    RESIDENTIAL_AC_WITH_BASEBOARD_GAS_BOILER = "ResidentialAC_BoilerBaseboard"
    RESIDENTIAL_AC_WITH_BASEBOARD_CENTRAL_AIR_SOURCE_HEAT_PUMP = "ResidentialAC_ASHPBaseboard"
    RESIDENTIAL_AC_WITH_BASEBOARD_DISTRICT_HOT_WATER = "ResidentialAC_DHWBaseboard"
    RESIDENTIAL_AC_WITH_RESIDENTIAL_FORCED_AIR_FURNACE = "ResidentialAC_ResidentialFurnace"
    RESIDENTIAL_AC_WITH_NO_HEAT = "ResidentialAC"
    RESIDENTIAL_HEAT_PUMP = "ResidentialHP"
    RESIDENTIAL_HEAT_PUMP_WITH_NO_COOLING = "ResidentialHPNoCool"
    RESIDENTIAL_FORCED_AIR_FURNACE = "ResidentialFurnace"
    VRF = "VRF"
    WATER_SOURCE_HEAT_PUMPS_FLUID_COOLER_WITH_BOILER = "WSHP_FluidCooler_Boiler"
    WATER_SOURCE_HEAT_PUMPS_COOLING_TOWER_WITH_BOILER = "WSHP_CoolingTower_Boiler"
    WATER_SOURCE_HEAT_PUMPS_WITH_GROUND_SOURCE_HEAT_PUMP = "WSHP_GSHP"
    WATER_SOURCE_HEAT_PUMPS_DISTRICT_CHILLED_WATER_WITH_DISTRICT_HOT_WATER = "WSHP_DCW_DHW"
    WINDOW_AC_WITH_BASEBOARD_ELECTRIC = "WindowAC_ElectricBaseboard"
    WINDOW_AC_WITH_BASEBOARD_GAS_BOILER = "WindowAC_BoilerBaseboard"
    WINDOW_AC_WITH_BASEBOARD_CENTRAL_AIR_SOURCE_HEAT_PUMP = "WindowAC_ASHPBaseboard"
    WINDOW_AC_WITH_BASEBOARD_DISTRICT_HOT_WATER = "WindowAC_DHWBaseboard"
    WINDOW_AC_WITH_FORCED_AIR_FURNACE = "WindowAC_Furnace"
    WINDOW_AC_WITH_UNIT_HEATERS = "WindowAC_GasHeaters"
    WINDOW_AC_WITH_NO_HEAT = "WindowAC"
    LOW_TEMPERATURE_RADIANT_CHILLER_WITH_BOILER = "Radiant_Chiller_Boiler"
    LOW_TEMPERATURE_RADIANT_CHILLER_WITH_AIR_SOURCE_HEAT_PUMP = "Radiant_Chiller_ASHP"
    LOW_TEMPERATURE_RADIANT_CHILLER_WITH_DISTRICT_HOT_WATER = "Radiant_Chiller_DHW"
    LOW_TEMPERATURE_RADIANT_AIR_COOLED_CHILLER_WITH_BOILER = "Radiant_ACChiller_Boiler"
    LOW_TEMPERATURE_RADIANT_AIR_COOLED_CHILLER_WITH_DISTRICT_HOT_WATER = "Radiant_ACChiller_DHW"
    LOW_TEMPERATURE_RADIANT_DISTRICT_CHILLED_WATER_WITH_BOILER = "Radiant_DCW_Boiler"
    LOW_TEMPERATURE_RADIANT_DISTRICT_CHILLED_WATER_WITH_AIR_SOURCE_HEAT_PUMP = "Radiant_DCW_ASHP"
    LOW_TEMPERATURE_RADIANT_DISTRICT_CHILLED_WATER_WITH_DISTRICT_HOT_WATER = "Radiant_DCW_DHW"


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

    match building_type.name:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE.name:
            gfa = 12000
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE.name:
            gfa = 3100
        case BuildingType.EDUCATION_COLLEGE.name:
            gfa = 5000
        case BuildingType.CIVIC_COURTHOUSE.name:
            gfa = 5000
        case BuildingType.DATACENTER_LARGE_HIGH_ITE.name:
            gfa = 8000
        case BuildingType.DATACENTER_LARGE_LOW_ITE.name:
            gfa = 8000
        case BuildingType.DATACENTER_SMALL_HIGH_ITE.name:
            gfa = 2000
        case BuildingType.DATACENTER_SMALL_LOW_ITE.name:
            gfa = 2000
        case BuildingType.HEALTHCARE_HOSPITAL.name:
            gfa = 22000
        case BuildingType.ACCOMODATION_HOTEL_LARGE.name:
            gfa = 11000
        case BuildingType.ACCOMODATION_HOTEL_SMALL.name:
            gfa = 4000
        case BuildingType.LABORATORY.name:
            gfa = 1000
        case BuildingType.COMMERCIAL_OFFICE_LARGE.name:
            gfa = 45000
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM.name:
            gfa = 5000
        case BuildingType.COMMERCIAL_OFFICE_SMALL.name:
            gfa = 500
        case BuildingType.HEALTHCARE_OUTPATIENT.name:
            gfa = 3800
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE.name:
            gfa = 500
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE.name:
            gfa = 200
        case BuildingType.COMMERCIAL_RETAIL.name:
            gfa = 2300
        case BuildingType.EDUCATION_SCHOOL_PRIMARY.name:
            gfa = 6900
        case BuildingType.EDUCATION_SCHOOL_SECONDARY.name:
            gfa = 19600
        case BuildingType.COMMERCIAL_STRIP_MALL.name:
            gfa = 10000
        case BuildingType.COMMERCIAL_SUPERMARKET.name:
            gfa = 4200
        case BuildingType.INDUSTRY_WAREHOUSE.name:
            gfa = 4800
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE.name:
            gfa = 120
        case BuildingType.CIVIC_CONCERT_HALL.name:
            gfa = 18000
        case BuildingType.PHYSICAL_EXERCISE.name:
            gfa = 400
        case BuildingType.PHYSICAL_EVENTS.name:
            gfa = 29000
        case BuildingType.RELIGIOUS.name:
            gfa = 4000
        case BuildingType.INDUSTRY_LIGHT.name:
            gfa = 1500
        case BuildingType.PARKING.name:
            gfa = 3500
        case BuildingType.CIVIC_LIBRARY.name:
            gfa = 1200
        case BuildingType.EXHIBITION.name:
            gfa = 40000
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

    match building_type.name:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE.name:
            n_floors = 10
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE.name:
            n_floors = 4
        case BuildingType.EDUCATION_COLLEGE.name:
            n_floors = 2
        case BuildingType.CIVIC_COURTHOUSE.name:
            n_floors = 1
        case BuildingType.DATACENTER_LARGE_HIGH_ITE.name:
            n_floors = 1
        case BuildingType.DATACENTER_LARGE_LOW_ITE.name:
            n_floors = 1
        case BuildingType.DATACENTER_SMALL_HIGH_ITE.name:
            n_floors = 1
        case BuildingType.DATACENTER_SMALL_LOW_ITE.name:
            n_floors = 1
        case BuildingType.HEALTHCARE_HOSPITAL.name:
            n_floors = 3
        case BuildingType.ACCOMODATION_HOTEL_LARGE.name:
            n_floors = 5
        case BuildingType.ACCOMODATION_HOTEL_SMALL.name:
            n_floors = 3
        case BuildingType.LABORATORY.name:
            n_floors = 1
        case BuildingType.COMMERCIAL_OFFICE_LARGE.name:
            n_floors = 10
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM.name:
            n_floors = 3
        case BuildingType.COMMERCIAL_OFFICE_SMALL.name:
            n_floors = 1
        case BuildingType.HEALTHCARE_OUTPATIENT.name:
            n_floors = 1
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE.name:
            n_floors = 1
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE.name:
            n_floors = 1
        case BuildingType.COMMERCIAL_RETAIL.name:
            n_floors = 1
        case BuildingType.EDUCATION_SCHOOL_PRIMARY.name:
            n_floors = 1
        case BuildingType.EDUCATION_SCHOOL_SECONDARY.name:
            n_floors = 3
        case BuildingType.COMMERCIAL_STRIP_MALL.name:
            n_floors = 2
        case BuildingType.COMMERCIAL_SUPERMARKET.name:
            n_floors = 1
        case BuildingType.INDUSTRY_WAREHOUSE.name:
            n_floors = 1
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE.name:
            n_floors = 2
        case BuildingType.CIVIC_CONCERT_HALL.name:
            n_floors = 2
        case BuildingType.PHYSICAL_EXERCISE.name:
            n_floors = 1
        case BuildingType.PHYSICAL_EVENTS.name:
            n_floors = 1
        case BuildingType.RELIGIOUS.name:
            n_floors = 1
        case BuildingType.INDUSTRY_LIGHT.name:
            n_floors = 1
        case BuildingType.PARKING.name:
            n_floors = 2
        case BuildingType.CIVIC_LIBRARY.name:
            n_floors = 1
        case BuildingType.EXHIBITION.name:
            n_floors = 2
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

    match building_type.name:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE.name:
            floor_height = 3.5
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE.name:
            floor_height = 3.5
        case BuildingType.EDUCATION_COLLEGE.name:
            floor_height = 3.8
        case BuildingType.CIVIC_COURTHOUSE.name:
            floor_height = 4
        case BuildingType.DATACENTER_LARGE_HIGH_ITE.name:
            floor_height = 4
        case BuildingType.DATACENTER_LARGE_LOW_ITE.name:
            floor_height = 4
        case BuildingType.DATACENTER_SMALL_HIGH_ITE.name:
            floor_height = 4
        case BuildingType.DATACENTER_SMALL_LOW_ITE.name:
            floor_height = 4
        case BuildingType.HEALTHCARE_HOSPITAL.name:
            floor_height = 3.8
        case BuildingType.ACCOMODATION_HOTEL_LARGE.name:
            floor_height = 3.8
        case BuildingType.ACCOMODATION_HOTEL_SMALL.name:
            floor_height = 3.8
        case BuildingType.LABORATORY.name:
            floor_height = 3.8
        case BuildingType.COMMERCIAL_OFFICE_LARGE.name:
            floor_height = 3.8
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM.name:
            floor_height = 3.8
        case BuildingType.COMMERCIAL_OFFICE_SMALL.name:
            floor_height = 3.8
        case BuildingType.HEALTHCARE_OUTPATIENT.name:
            floor_height = 3.8
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE.name:
            floor_height = 4
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE.name:
            floor_height = 4
        case BuildingType.COMMERCIAL_RETAIL.name:
            floor_height = 4
        case BuildingType.EDUCATION_SCHOOL_PRIMARY.name:
            floor_height = 4
        case BuildingType.EDUCATION_SCHOOL_SECONDARY.name:
            floor_height = 4
        case BuildingType.COMMERCIAL_STRIP_MALL.name:
            floor_height = 4.25
        case BuildingType.COMMERCIAL_SUPERMARKET.name:
            floor_height = 4.5
        case BuildingType.INDUSTRY_WAREHOUSE.name:
            floor_height = 4.5
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE.name:
            floor_height = 3.3
        case BuildingType.CIVIC_CONCERT_HALL.name:
            floor_height = 5
        case BuildingType.PHYSICAL_EXERCISE.name:
            floor_height = 3.5
        case BuildingType.PHYSICAL_EVENTS.name:
            floor_height = 5
        case BuildingType.RELIGIOUS.name:
            floor_height = 5
        case BuildingType.INDUSTRY_LIGHT.name:
            floor_height = 4.5
        case BuildingType.PARKING.name:
            floor_height = 3.2
        case BuildingType.CIVIC_LIBRARY.name:
            floor_height = 3.5
        case BuildingType.EXHIBITION.name:
            floor_height = 5
        case _:
            raise ValueError(
                f"No default average floor height is available for {building_type}."
            )
    return float(floor_height)


def default_construction_type(building_type: BuildingType) -> ConstructionType:
    """Get the typical ConstructionType for the building type.

    Args:
        building_type (BuildingType):
            The type of building to calculate for.

    Returns:
        ConstructionType:
            The typical construction type for the building type.
    """

    match building_type.name:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE.name:
            constr_type = ConstructionType.MASS
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE.name:
            constr_type = ConstructionType.MASS
        case BuildingType.EDUCATION_COLLEGE.name:
            constr_type = ConstructionType.MASS
        case BuildingType.CIVIC_COURTHOUSE.name:
            constr_type = ConstructionType.MASS
        case BuildingType.DATACENTER_LARGE_HIGH_ITE.name:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.DATACENTER_LARGE_LOW_ITE.name:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.DATACENTER_SMALL_HIGH_ITE.name:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.DATACENTER_SMALL_LOW_ITE.name:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.HEALTHCARE_HOSPITAL.name:
            constr_type = ConstructionType.MASS
        case BuildingType.ACCOMODATION_HOTEL_LARGE.name:
            constr_type = ConstructionType.MASS
        case BuildingType.ACCOMODATION_HOTEL_SMALL.name:
            constr_type = ConstructionType.MASS
        case BuildingType.LABORATORY.name:
            constr_type = ConstructionType.MASS
        case BuildingType.COMMERCIAL_OFFICE_LARGE.name:
            constr_type = ConstructionType.MASS
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM.name:
            constr_type = ConstructionType.MASS
        case BuildingType.COMMERCIAL_OFFICE_SMALL.name:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.HEALTHCARE_OUTPATIENT.name:
            constr_type = ConstructionType.MASS
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE.name:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE.name:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.COMMERCIAL_RETAIL.name:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.EDUCATION_SCHOOL_PRIMARY.name:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.EDUCATION_SCHOOL_SECONDARY.name:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.COMMERCIAL_STRIP_MALL.name:
            constr_type = ConstructionType.MASS
        case BuildingType.COMMERCIAL_SUPERMARKET.name:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.INDUSTRY_WAREHOUSE.name:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE.name:
            constr_type = ConstructionType.WOOD_FRAMED
        case BuildingType.CIVIC_CONCERT_HALL.name:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.PHYSICAL_EXERCISE.name:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.PHYSICAL_EVENTS.name:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.RELIGIOUS.name:
            constr_type = ConstructionType.STEEL_FRAMED
        case BuildingType.INDUSTRY_LIGHT.name:
            constr_type = ConstructionType.METAL_BUILDING
        case BuildingType.PARKING.name:
            constr_type = ConstructionType.MASS
        case BuildingType.CIVIC_LIBRARY.name:
            constr_type = ConstructionType.MASS
        case BuildingType.EXHIBITION.name:
            constr_type = ConstructionType.STEEL_FRAMED
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

    match building_type.name:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE.name:
            glazing_ratio = 0.3
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE.name:
            glazing_ratio = 0.3
        case BuildingType.EDUCATION_COLLEGE.name:
            glazing_ratio = 0.3
        case BuildingType.CIVIC_COURTHOUSE.name:
            glazing_ratio = 0.3
        case BuildingType.DATACENTER_LARGE_HIGH_ITE.name:
            glazing_ratio = 0.05
        case BuildingType.DATACENTER_LARGE_LOW_ITE.name:
            glazing_ratio = 0.05
        case BuildingType.DATACENTER_SMALL_HIGH_ITE.name:
            glazing_ratio = 0.05
        case BuildingType.DATACENTER_SMALL_LOW_ITE.name:
            glazing_ratio = 0.05
        case BuildingType.HEALTHCARE_HOSPITAL.name:
            glazing_ratio = 0.2
        case BuildingType.ACCOMODATION_HOTEL_LARGE.name:
            glazing_ratio = 0.2
        case BuildingType.ACCOMODATION_HOTEL_SMALL.name:
            glazing_ratio = 0.2
        case BuildingType.LABORATORY.name:
            glazing_ratio = 0.2
        case BuildingType.COMMERCIAL_OFFICE_LARGE.name:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM.name:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_OFFICE_SMALL.name:
            glazing_ratio = 0.3
        case BuildingType.HEALTHCARE_OUTPATIENT.name:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE.name:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE.name:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_RETAIL.name:
            glazing_ratio = 0.3
        case BuildingType.EDUCATION_SCHOOL_PRIMARY.name:
            glazing_ratio = 0.3
        case BuildingType.EDUCATION_SCHOOL_SECONDARY.name:
            glazing_ratio = 0.3
        case BuildingType.COMMERCIAL_STRIP_MALL.name:
            glazing_ratio = 0.2
        case BuildingType.COMMERCIAL_SUPERMARKET.name:
            glazing_ratio = 0.05
        case BuildingType.INDUSTRY_WAREHOUSE.name:
            glazing_ratio = 0.05
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE.name:
            glazing_ratio = 0.4
        case BuildingType.CIVIC_CONCERT_HALL.name:
            glazing_ratio = 0.05
        case BuildingType.PHYSICAL_EXERCISE.name:
            glazing_ratio = 0.3
        case BuildingType.PHYSICAL_EVENTS.name:
            glazing_ratio = 0.05
        case BuildingType.RELIGIOUS.name:
            glazing_ratio = 0.3
        case BuildingType.INDUSTRY_LIGHT.name:
            glazing_ratio = 0.05
        case BuildingType.PARKING.name:
            glazing_ratio = 0.05
        case BuildingType.CIVIC_LIBRARY.name:
            glazing_ratio = 0.3
        case BuildingType.EXHIBITION.name:
            glazing_ratio = 0.15
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

    match building_type.name:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE.name:
            skylight_ratio = 0
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE.name:
            skylight_ratio = 0
        case BuildingType.EDUCATION_COLLEGE.name:
            skylight_ratio = 0
        case BuildingType.CIVIC_COURTHOUSE.name:
            skylight_ratio = 0
        case BuildingType.DATACENTER_LARGE_HIGH_ITE.name:
            skylight_ratio = 0
        case BuildingType.DATACENTER_LARGE_LOW_ITE.name:
            skylight_ratio = 0
        case BuildingType.DATACENTER_SMALL_HIGH_ITE.name:
            skylight_ratio = 0
        case BuildingType.DATACENTER_SMALL_LOW_ITE.name:
            skylight_ratio = 0
        case BuildingType.HEALTHCARE_HOSPITAL.name:
            skylight_ratio = 0
        case BuildingType.ACCOMODATION_HOTEL_LARGE.name:
            skylight_ratio = 0
        case BuildingType.ACCOMODATION_HOTEL_SMALL.name:
            skylight_ratio = 0
        case BuildingType.LABORATORY.name:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_OFFICE_LARGE.name:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM.name:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_OFFICE_SMALL.name:
            skylight_ratio = 0
        case BuildingType.HEALTHCARE_OUTPATIENT.name:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE.name:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE.name:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_RETAIL.name:
            skylight_ratio = 0
        case BuildingType.EDUCATION_SCHOOL_PRIMARY.name:
            skylight_ratio = 0
        case BuildingType.EDUCATION_SCHOOL_SECONDARY.name:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_STRIP_MALL.name:
            skylight_ratio = 0
        case BuildingType.COMMERCIAL_SUPERMARKET.name:
            skylight_ratio = 0
        case BuildingType.INDUSTRY_WAREHOUSE.name:
            skylight_ratio = 0
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE.name:
            skylight_ratio = 0
        case BuildingType.CIVIC_CONCERT_HALL.name:
            skylight_ratio = 0
        case BuildingType.PHYSICAL_EXERCISE.name:
            skylight_ratio = 0
        case BuildingType.PHYSICAL_EVENTS.name:
            skylight_ratio = 0
        case BuildingType.RELIGIOUS.name:
            skylight_ratio = 0
        case BuildingType.INDUSTRY_LIGHT.name:
            skylight_ratio = 0
        case BuildingType.PARKING.name:
            skylight_ratio = 0
        case BuildingType.CIVIC_LIBRARY.name:
            skylight_ratio = 0
        case BuildingType.EXHIBITION.name:
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

    match terrain_type.name:
        case TerrainType.OCEAN.name:
            context_distance = 1500
        case TerrainType.COUNTRY.name:
            context_distance = 150
        case TerrainType.SUBURBS.name:
            context_distance = 50
        case TerrainType.URBAN.name:
            context_distance = 40
        case TerrainType.CITY.name:
            context_distance = 30
        case _:
            raise ValueError(
                f"No default context height is available for {terrain_type}."
            )
    return float(context_distance)


def default_program_type(building_type: BuildingType) -> ProgramType:
    """Get the typical ProgramType for the building type."""

    match building_type.name:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE.name:
            # logger.info(
            #     "%s currently using %s default program.",
            #     building_type,
            #     BuildingType.ACCOMODATION_APARTMENT_MIDRISE,
            # )
            program = building_program_type_by_identifier("MidriseApartment")
        case BuildingType.EDUCATION_COLLEGE.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.CIVIC_COURTHOUSE.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.DATACENTER_LARGE_HIGH_ITE.name:
            program = building_program_type_by_identifier(building_type.value)
            program.unlock()
            program.setpoint.humidifying_setpoint = 45
            program.setpoint.dehumidifying_setpoint = 55
            program.lock()
        case BuildingType.DATACENTER_LARGE_LOW_ITE.name:
            program = building_program_type_by_identifier(building_type.value)
            program.unlock()
            program.setpoint.humidifying_setpoint = 45
            program.setpoint.dehumidifying_setpoint = 55
            program.lock()
        case BuildingType.DATACENTER_SMALL_HIGH_ITE.name:
            program = building_program_type_by_identifier(building_type.value)
            program.unlock()
            program.setpoint.humidifying_setpoint = 45
            program.setpoint.dehumidifying_setpoint = 55
            program.lock()
        case BuildingType.DATACENTER_SMALL_LOW_ITE.name:
            program = building_program_type_by_identifier(building_type.value)
            program.unlock()
            program.setpoint.humidifying_setpoint = 45
            program.setpoint.dehumidifying_setpoint = 55
            program.lock()
        case BuildingType.HEALTHCARE_HOSPITAL.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.ACCOMODATION_HOTEL_LARGE.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.ACCOMODATION_HOTEL_SMALL.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.LABORATORY.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_OFFICE_LARGE.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_OFFICE_SMALL.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.HEALTHCARE_OUTPATIENT.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_RETAIL.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.EDUCATION_SCHOOL_PRIMARY.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.EDUCATION_SCHOOL_SECONDARY.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_STRIP_MALL.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.COMMERCIAL_SUPERMARKET.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.INDUSTRY_WAREHOUSE.name:
            program = building_program_type_by_identifier(building_type.value)
        case BuildingType.CIVIC_CONCERT_HALL.name:
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
        case BuildingType.PHYSICAL_EXERCISE.name:
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
        case BuildingType.PHYSICAL_EVENTS.name:
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
        case BuildingType.RELIGIOUS.name:
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
        case BuildingType.INDUSTRY_LIGHT.name:
            # from https://www.wbdg.org/space-types/light-industrial
            bld_mix_dict = {
                "2019::Warehouse::Office": 0.01125,
                "2019::Warehouse::Bulk": 0.62627,
                "2019::SmallDataCenterLowITE::ComputerRoom": 0.25,
                "2019::Warehouse::Fine": 0.11248,
            }
            progs, ratios = [], []
            for key, val in bld_mix_dict.items():
                progs.append(program_type_by_identifier(key))
                ratios.append(val)
            program = ProgramType.average("LightIndustry", progs, ratios)
            program.lock()
        case BuildingType.PARKING.name:
            # from ... an approximation
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
        case BuildingType.CIVIC_LIBRARY.name:
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
        case BuildingType.EXHIBITION.name:
            # from https://www.wbdg.org/space-types/exhibition-center
            bld_mix_dict = {
                "2019::Courthouse::Entrance Lobby": 0.19473,
                "2019::Courthouse::Storage": 0.0549,
                "2019::SecondarySchool::Cafeteria": 0.0244,
                "2019::SecondarySchool::Library": 0.43924,
                "2019::SecondarySchool::Auditorium": 0.0183,
                "2019::College::Media Center": 0.08541,
                "2019::Courthouse::Restrooms": 0.0366,
                "2019::Courthouse::Office": 0.14642,
            }
            progs, ratios = [], []
            for key, val in bld_mix_dict.items():
                progs.append(program_type_by_identifier(key))
                ratios.append(val)
            program = ProgramType.average("Exhibition", progs, ratios)
            program.lock()
        case _:
            raise ValueError(f"No default program is available for {building_type}.")
    return program

def default_number_of_lifts(building_type: BuildingType, n_floors: float = None, building_area: float = None) -> int:
    """Approximate the number of lifts needed for the building type.
    
    Several inputs are given, but not all are needed for all calculations.

    Returns:
        int:
            The number of lifts needed for the building type (and configuration)
    """

    if n_floors <= 1:
        return 0
    
    match building_type.name:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE.name | BuildingType.ACCOMODATION_APARTMENT_MIDRISE.name | BuildingType.ACCOMODATION_HOUSE_LOWRISE.name:
            # TODO - estimate number of lifts for highrise apartment
            # Based on the number of units, with typical unit area of 75m2
            lifts = 1
        case BuildingType.EDUCATION_COLLEGE.name | BuildingType.EDUCATION_SCHOOL_PRIMARY.name | BuildingType.EDUCATION_SCHOOL_SECONDARY.name:
            # TODO - estimate number of lifts for school
            lifts = 1
        case BuildingType.CIVIC_LIBRARY.name | BuildingType.CIVIC_COURTHOUSE.name:
            lifts = 1
        case BuildingType.INDUSTRY_WAREHOUSE.name | BuildingType.INDUSTRY_LIGHT.name | BuildingType.DATACENTER_LARGE_HIGH_ITE.name | BuildingType.DATACENTER_LARGE_LOW_ITE.name | BuildingType.DATACENTER_SMALL_HIGH_ITE.name | BuildingType.DATACENTER_SMALL_LOW_ITE.name:
            lifts = 1
        case BuildingType.HEALTHCARE_HOSPITAL.name | BuildingType.HEALTHCARE_OUTPATIENT.name:
            lifts = 1
        case BuildingType.ACCOMODATION_HOTEL_LARGE.name | BuildingType.ACCOMODATION_HOTEL_SMALL.name:
            lifts = 1
        case BuildingType.LABORATORY.name:
            lifts = 1
        case BuildingType.COMMERCIAL_OFFICE_LARGE.name | BuildingType.COMMERCIAL_OFFICE_MEDIUM.name | BuildingType.COMMERCIAL_OFFICE_SMALL.name:
            lifts = 1
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE.name:
            lifts = 1
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE.name | BuildingType.COMMERCIAL_RETAIL.name | BuildingType.COMMERCIAL_STRIP_MALL.name | BuildingType.COMMERCIAL_SUPERMARKET.name:
            lifts = 1
        case BuildingType.EXHIBITION.name | BuildingType.CIVIC_CONCERT_HALL.name | BuildingType.PHYSICAL_EVENTS.name | BuildingType.RELIGIOUS.name:
            lifts = 1
        case BuildingType.PHYSICAL_EXERCISE.name:
            lifts = 1
        case BuildingType.PARKING.name:
            lifts = 1
        case _:
            raise ValueError(f"No default number of lifts is available for {building_type}.")
    return lifts


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


def default_hvac_system(building_type: BuildingType, epw: EPW, vintage: Vintage) -> None:
    """Get the typical HVAC system for the building type, vintage and climate."""

    raise NotImplementedError("This function is not yet implemented.")
    
    ashrae_climate = int(epw.ashrae_climate_zone[0])

    sys = DEFAULT_SYSTEMS[
        (DEFAULT_SYSTEMS.building_type == building_type.name)
        & (DEFAULT_SYSTEMS.vintage == vintage.name)
        & (DEFAULT_SYSTEMS.ashrae_climate == ashrae_climate)
    ].squeeze()["default_system"]

    # dictionary of HVAC template names
    ext_folder = hbe_folders.standards_extension_folders[0]
    hvac_reg = Path(ext_folder) / 'hvac_registry.json'
    with open(hvac_reg, 'r') as f:
        hvac_dict = json.load(f)
    
    # create the default ideal_air system
    shr, lhr = default_hr_effectiveness(building_type, epw, vintage)
    ideal_air = IdealAirSystem(
        identifier=f"{building_type.value}_IdealAirSystem",
        economizer_type=default_economizer_type(building_type, epw, vintage).value,
        demand_controlled_ventilation=default_demand_controlled_ventilation(building_type, vintage),
        sensible_heat_recovery=shr,
        latent_heat_recovery=lhr
    )

    return ideal_air

class LiftUsageIntensity(Enum):
    """The usage intensity of a lift, based on ISO 25745-2:2015."""
    VeryLow = auto()
    Low = auto()
    Medium = auto()
    High = auto()
    VeryHigh = auto()
    ExtremelyHigh = auto()

    def trips_per_day(self) -> int:
        """The number of trips per day (n_d) for the usage intensity."""
        return {
            LiftUsageIntensity.VeryLow.name: 50,
            LiftUsageIntensity.Low.name: 125,
            LiftUsageIntensity.Medium.name: 300,
            LiftUsageIntensity.High.name: 750,
            LiftUsageIntensity.VeryHigh.name: 1500,
            LiftUsageIntensity.ExtremelyHigh.name: 2500
        }[self.name]
    
    def typical_range(self) -> tuple[float]:
        """The typical range of the lift (m)."""
        return {
            LiftUsageIntensity.VeryLow.name: (0, 75),
            LiftUsageIntensity.Low.name: (75, 200),
            LiftUsageIntensity.Medium.name: (200, 500),
            LiftUsageIntensity.High.name: (500, 1000),
            LiftUsageIntensity.VeryHigh.name: (1000, 2000),
            LiftUsageIntensity.ExtremelyHigh.name: (2000, np.inf)
        }[self.name]
    
    def percentage_average_travel_distance(self, n_floors: int) -> float:
        """The average travel distance (s_av) for the usage intensity, as a 
        percentage of the total height of the building.
        
        Args:
            n_floors (int): The number of floors in the building.

        Returns:
            float: The average travel distance as a percentage of the total 
                height of the building.
        """

        if n_floors < 2:
            raise ValueError("The number of floors must be greater than 1.")
        
        if n_floors == 2:
            return 1
        
        if n_floors == 3:
            return 0.67
        
        return {
            LiftUsageIntensity.VeryLow.name: 0.49,
            LiftUsageIntensity.Low.name: 0.49,
            LiftUsageIntensity.Medium.name: 0.49,
            LiftUsageIntensity.High.name: 0.44,
            LiftUsageIntensity.VeryHigh.name: 0.39,
            LiftUsageIntensity.ExtremelyHigh.name: 0.32
        }[self.name]
    
    def average_car_load(self, rated_load: float) -> float:
        """The average car load (m) for the usage intensity.
        
        Args:
            rated_load (float): The rated load of the lift (kg).

        Returns:
            float: The percentage of rated car load (for use in estimating 
            the load factor (k_L)).
        """
        if rated_load <= 800:
            return {
                LiftUsageIntensity.VeryLow.name: 0.075,
                LiftUsageIntensity.Low.name: 0.075,
                LiftUsageIntensity.Medium.name: 7.5,
                LiftUsageIntensity.High.name: 0.09,
                LiftUsageIntensity.VeryHigh.name: 0.16,
                LiftUsageIntensity.ExtremelyHigh.name: 0.16
            }[self.name]

        if rated_load <= 1275:
            return {
                LiftUsageIntensity.VeryLow.name: 0.045,
                LiftUsageIntensity.Low.name: 0.045,
                LiftUsageIntensity.Medium.name: 0.045,
                LiftUsageIntensity.High.name: 0.06,
                LiftUsageIntensity.VeryHigh.name: 0.11,
                LiftUsageIntensity.ExtremelyHigh.name: 0.11
            }[self.name]
        
        if rated_load <= 2000:
            return {
                LiftUsageIntensity.VeryLow.name: 0.03,
                LiftUsageIntensity.Low.name: 0.03,
                LiftUsageIntensity.Medium.name: 0.03,
                LiftUsageIntensity.High.name: 0.035,
                LiftUsageIntensity.VeryHigh.name: 0.07,
                LiftUsageIntensity.ExtremelyHigh.name: 0.07
            }[self.name]

        return {
            LiftUsageIntensity.VeryLow.name: 0.02,
            LiftUsageIntensity.Low.name: 0.02,
            LiftUsageIntensity.Medium.name: 0.02,
            LiftUsageIntensity.High.name: 0.022,
            LiftUsageIntensity.VeryHigh.name: 0.045,
            LiftUsageIntensity.ExtremelyHigh.name: 0.045
        }[self.name]
    
    def average_travel_time(self) -> float:
        """Return the decimal hours over a single day when travel occurs. 
        Values given are <=X where X is hours-per-day."""
        return {
            LiftUsageIntensity.VeryLow.name: 0.2,
            LiftUsageIntensity.Low.name: 0.5,
            LiftUsageIntensity.Medium.name: 1.5,
            LiftUsageIntensity.High.name: 3,
            LiftUsageIntensity.VeryHigh.name: 6,
            LiftUsageIntensity.ExtremelyHigh.name: 12
        }[self.name]

    def average_standby_time(self) -> float:
        """Return the decimal hours over a single day when standby occurs."""
        return 24 - self.average_travel_time()

class LiftEnergyEfficiency(Enum):
    """According to VDI 4707 2009-3"""
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()

    def standby_consumption(self) -> float:
        """Return the standby power demand in W."""
        return {
            LiftEnergyEfficiency.A.name: 50,
            LiftEnergyEfficiency.B.name: 100,
            LiftEnergyEfficiency.C.name: 200,
            LiftEnergyEfficiency.D.name: 400,
            LiftEnergyEfficiency.E.name: 800,
            LiftEnergyEfficiency.F.name: 1600,
            LiftEnergyEfficiency.G.name: 3200,
        }[self.name]
    
    def operational_consumption(self) -> float:
        """Return the operational energy demand per travel in mWh/(kg.m)."""
        return {
            LiftEnergyEfficiency.A.name: 0.56,
            LiftEnergyEfficiency.B.name: 0.84,
            LiftEnergyEfficiency.C.name: 1.26,
            LiftEnergyEfficiency.D.name: 1.89,
            LiftEnergyEfficiency.E.name: 2.8,
            LiftEnergyEfficiency.F.name: 4.2,
            LiftEnergyEfficiency.G.name: 8.4,
        }[self.name]


def default_lift_usage_intensity(building_type: BuildingType) -> LiftUsageIntensity:
    match building_type.name:
        case BuildingType.ACCOMODATION_APARTMENT_HIGHRISE.name:
            usage_intensity = LiftUsageIntensity.High
        case BuildingType.ACCOMODATION_APARTMENT_MIDRISE.name:
            usage_intensity = LiftUsageIntensity.Medium
        case BuildingType.EDUCATION_COLLEGE.name:
            usage_intensity = LiftUsageIntensity.Low
        case BuildingType.CIVIC_COURTHOUSE.name:
            usage_intensity = LiftUsageIntensity.Low
        case BuildingType.DATACENTER_LARGE_HIGH_ITE.name:
            usage_intensity = LiftUsageIntensity.VeryLow
        case BuildingType.DATACENTER_LARGE_LOW_ITE.name:
            usage_intensity = LiftUsageIntensity.VeryLow
        case BuildingType.DATACENTER_SMALL_HIGH_ITE.name:
            usage_intensity = LiftUsageIntensity.VeryLow
        case BuildingType.DATACENTER_SMALL_LOW_ITE.name:
            usage_intensity = LiftUsageIntensity.VeryLow
        case BuildingType.HEALTHCARE_HOSPITAL.name:
            usage_intensity = LiftUsageIntensity.High
        case BuildingType.ACCOMODATION_HOTEL_LARGE.name:
            usage_intensity = LiftUsageIntensity.High
        case BuildingType.ACCOMODATION_HOTEL_SMALL.name:
            usage_intensity = LiftUsageIntensity.Medium
        case BuildingType.LABORATORY.name:
            usage_intensity = LiftUsageIntensity.Low
        case BuildingType.COMMERCIAL_OFFICE_LARGE.name:
            usage_intensity = LiftUsageIntensity.VeryHigh
        case BuildingType.COMMERCIAL_OFFICE_MEDIUM.name:
            usage_intensity = LiftUsageIntensity.High
        case BuildingType.COMMERCIAL_OFFICE_SMALL.name:
            usage_intensity = LiftUsageIntensity.Medium
        case BuildingType.HEALTHCARE_OUTPATIENT.name:
            usage_intensity = LiftUsageIntensity.High
        case BuildingType.COMMERCIAL_RESTAURANT_FULL_SERVICE.name:
            usage_intensity = LiftUsageIntensity.Low
        case BuildingType.COMMERCIAL_RESTAURANT_QUICK_SERVICE.name:
            usage_intensity = LiftUsageIntensity.VeryLow
        case BuildingType.COMMERCIAL_RETAIL.name:
            usage_intensity = LiftUsageIntensity.Medium
        case BuildingType.EDUCATION_SCHOOL_PRIMARY.name:
            usage_intensity = LiftUsageIntensity.VeryLow
        case BuildingType.EDUCATION_SCHOOL_SECONDARY.name:
            usage_intensity = LiftUsageIntensity.Low
        case BuildingType.COMMERCIAL_STRIP_MALL.name:
            usage_intensity = LiftUsageIntensity.Medium
        case BuildingType.COMMERCIAL_SUPERMARKET.name:
            usage_intensity = LiftUsageIntensity.Low
        case BuildingType.INDUSTRY_WAREHOUSE.name:
            usage_intensity = LiftUsageIntensity.VeryLow
        case BuildingType.ACCOMODATION_HOUSE_LOWRISE.name:
            usage_intensity = LiftUsageIntensity.VeryLow
        case BuildingType.CIVIC_CONCERT_HALL.name:
            usage_intensity = LiftUsageIntensity.Medium
        case BuildingType.PHYSICAL_EXERCISE.name:
            usage_intensity = LiftUsageIntensity.Low
        case BuildingType.PHYSICAL_EVENTS.name:
            usage_intensity = LiftUsageIntensity.Medium
        case BuildingType.RELIGIOUS.name:
            usage_intensity = LiftUsageIntensity.Medium
        case BuildingType.INDUSTRY_LIGHT.name:
            usage_intensity = LiftUsageIntensity.VeryLow
        case BuildingType.PARKING.name:
            usage_intensity = LiftUsageIntensity.Low
        case BuildingType.CIVIC_LIBRARY.name:
            usage_intensity = LiftUsageIntensity.Low
        case BuildingType.EXHIBITION.name:
            usage_intensity = LiftUsageIntensity.Medium
        case _:
            raise ValueError(
                f"No default lift usage intensity is available for {building_type}."
            )
    return float(usage_intensity)