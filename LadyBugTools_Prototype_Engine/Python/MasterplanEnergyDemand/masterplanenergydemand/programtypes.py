"""A set of custom program types for the Master Plan Energy Demand Model 
that are not included by default in the Honeybee library."""

from honeybee_energy.lib.programtypes import (
    building_program_type_by_identifier,
    program_type_by_identifier,
    ProgramType,
)

_CUSTOM_BUILDING_DICT = {
    "ResidentialLowRise": {
        "2019::MidriseApartment::Apartment": 1.0,
    },
    # from AECOM. Cost Model: New-Build Concert Halls. LM00093-0817-v2.0, July 2017.
    "ConcertHall": {
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
    },
    # https://www.wbdg.org/space-types/physical-fitness-exercise-room
    "PhysicalFitnessExercise": {
        "2019::College::Entrance Lobby": 0.01307,
        "2019::Hospital::PhysTherapy": 0.13725,
        "2019::LargeOffice::Restroom": 0.0915,
        "2019::SecondarySchool::Gym": 0.70589,
        "2019::College::Storage": 0.05229,
    },
    # from https://www.wbdg.org/space-types/auditorium
    "PhysicalFitnessEvents": {
        "2019::Courthouse::Entrance Lobby": 0.19473,
        "2019::Courthouse::Storage": 0.0549,
        "2019::SecondarySchool::Cafeteria": 0.0244,
        "2019::SecondarySchool::Library": 0.0183,
        "2019::SecondarySchool::Auditorium": 0.43924,
        "2019::SecondarySchool::Gym": 0.14642,
        "2019::College::Media Center": 0.08541,
        "2019::Courthouse::Restrooms": 0.0366,
    },
    # from https://www.wbdg.org/space-types/place-worship
    "PlaceOfWorship": {
        "2019::Courthouse::Courtroom": 0.49690,
        "2019::Courthouse::Storage": 0.05797,
        "2019::Courthouse::Office": 0.10352,
        "2019::Courthouse::Utility": 0.16770,
        "2019::Courthouse::Corridor": 0.17391,
    },
    # from https://www.wbdg.org/space-types/light-industrial
    "LightIndustry": {
        "2019::Warehouse::Office": 0.01125,
        "2019::Warehouse::Bulk": 0.53622,
        "2019::SmallDataCenterLowITE::ComputerRoom": 0.34005,
        "2019::Warehouse::Fine": 0.11248,
    },
    "ParkingBasement": {
        "2019::Courthouse::Parking": 0.95000,
        "2019::SuperMarket::Elec/MechRoom": 0.05000,
    },
    #  from https://www.wbdg.org/space-types/library
    "Library": {
        "2019::College::Entrance Lobby": 0.0594,
        "2019::Courthouse::Jury Deliberation": 0.02121,
        "2019::LargeOffice::PrintRoom": 0.09418,
        "2019::Courthouse::Office": 0.18244,
        "2019::College::Media Center": 0.1171,
        "2019::Courthouse::Library": 0.39839,
        "2019::College::Lounge": 0.11031,
        "2019::Courthouse::Utility": 0.01697,
    },
}


# pylint: disable=E0102
def _building_program_type_by_identifier(building_type: str) -> ProgramType:
    """A wrapper around the default building_program_type_by_identifier function that
    also includes custom building types defined here.

    Args:
        building_type: The building type identifier.

    Returns:
        ProgramType: The program type for the building type.
    """

    program_id = f"{building_type} Building"

    try:
        return building_program_type_by_identifier(building_type)
    except ValueError:
        bld_mix_dict = _CUSTOM_BUILDING_DICT[building_type]
        progs, ratios = [], []
        for key, val in bld_mix_dict.items():
            progs.append(program_type_by_identifier(key))
            ratios.append(val)
        bld_program = ProgramType.average(program_id, progs, ratios)
        bld_program.lock()
        return bld_program
    except KeyError as e:
        raise e


# pylint: enable=E0102
