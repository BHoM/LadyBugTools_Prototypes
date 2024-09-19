"""
A module containing the definition of a Typology describing the 
configuration of a building type within a masterplan.
"""

# region: IMPORTS
# pylint: disable=E0401

import inspect
import json
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from honeybee.boundarycondition import Ground, Outdoors
from honeybee.config import folders as hb_folders
from honeybee.model import Aperture, Face, Model, Room, Shade
from honeybee.typing import valid_string
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.internalmass import InternalMass
from honeybee_energy.lib.scheduletypelimits import humidity, temperature
from honeybee_energy.programtype import ProgramType
from honeybee_energy.result.loadbalance import SQLiteResult
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval
from honeybee_energy.simulation.parameter import (RunPeriod, ShadowCalculation,
                                                  SimulationControl,
                                                  SimulationOutput,
                                                  SimulationParameter,
                                                  SizingParameter)
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import Header, HourlyContinuousCollection
from ladybug.datatype.energy import Energy, EnergyIntensity
from ladybug.datatype.energyflux import EnergyFlux
from ladybug.datatype.volumeflowrate import VolumeFlowRate
from ladybug_geometry.geometry2d import Point2D, Polygon2D, Vector2D
from ladybug_geometry.geometry3d import (Face3D, LineSegment3D, Point3D,
                                         Vector3D)
from pydantic import BaseModel, Field, root_validator  # pylint: disable=E0611

from .config import INDEX, logger
from .enums import (EPW, BuildingType, ConstructionType, EconomizerType,
                    TerrainType, Vintage, default_construction_type,
                    default_constructionset, default_context_shade_distance,
                    default_cooling_eer, default_daylight_dimming,
                    default_demand_controlled_ventilation,
                    default_economizer_type, default_fan_power,
                    default_floor_height, default_footprint_area,
                    default_glazing_ratio, default_heating_cop,
                    default_hr_effectiveness, default_number_of_floors,
                    default_program_type, default_pump_power,
                    default_skylight_ratio)
from .plot import diurnal, duration_curve, pie, stacked_bar
from .util import (aggregate_collection, annual_eui, collection_from_series,
                   collection_to_series, construction_sri,
                   describe_analysis_period, estimate_sri_properties,
                   eui_from_sql, face_orientation, face_result, get_unit,
                   load_balance, occupancy_from_program,
                   occupancy_schedule_from_program, random_id,
                   room_comfort_result, room_energy_result)

# pylint: enable=E0401
# endregion: IMPORTS

ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DEFAULT_MASTERPLAN_IDENTIFIER = "UnnamedMasterplan"
DEFAULT_SIMULATION_DIRECTORY = Path(hb_folders.default_simulation_folder)


class Typology(BaseModel):
    """The building typology to be simulated as part of a masterplan energy model."""

    # region: ATTRIBUTES
    identifier: str = Field(
        description="A unique identifier for the typology.", unit="str"
    )
    total_area: float = Field(
        description="The total area of this building typology, across the entire masterplan (m2).",
        gt=0,
        unit="m2",
    )
    epw_file: Path = Field(
        description="The filepath for the EPW to use for the simulation.", unit="str"
    )
    building_type: BuildingType = Field(
        description="The type of building.", unit="enum"
    )
    vintage: Vintage = Field(description="The vintage of this typology.", unit="enum")
    # FORM
    average_footprint_area: float = Field(
        description="The average footprint area of the building (m2).", gt=0, unit="m2"
    )
    average_num_floors: int = Field(
        description="The average number of floors in the building.", gt=0, unit="int"
    )
    average_floor_height: float = Field(
        description="The average floor height of the building (m).", gt=0, unit="m"
    )
    rotation: float = Field(
        description="The rotation of the building (degrees).",
        ge=0,
        lt=360,
        unit="degrees",
    )
    terrain: TerrainType = Field(
        description="The terrain type for the building.", unit="enum"
    )
    glazing_ratio_N: float = Field(
        description="The ratio of glazing area to wall area for the N orientation.",
        ge=0,
        le=0.95,
        unit="unitless",
    )
    glazing_ratio_NE: float = Field(
        description="The ratio of glazing area to wall area for the NE orientation.",
        ge=0,
        le=0.95,
        unit="unitless",
    )
    glazing_ratio_E: float = Field(
        description="The ratio of glazing area to wall area for the E orientation.",
        ge=0,
        le=0.95,
        unit="unitless",
    )
    glazing_ratio_SE: float = Field(
        description="The ratio of glazing area to wall area for the SE orientation.",
        ge=0,
        le=0.95,
        unit="unitless",
    )
    glazing_ratio_S: float = Field(
        description="The ratio of glazing area to wall area for the S orientation.",
        ge=0,
        le=0.95,
        unit="unitless",
    )
    glazing_ratio_SW: float = Field(
        description="The ratio of glazing area to wall area for the SW orientation.",
        ge=0,
        le=0.95,
        unit="unitless",
    )
    glazing_ratio_W: float = Field(
        description="The ratio of glazing area to wall area for the W orientation.",
        ge=0,
        le=0.95,
        unit="unitless",
    )
    glazing_ratio_NW: float = Field(
        description="The ratio of glazing area to wall area for the NW orientation.",
        ge=0,
        le=0.95,
        unit="unitless",
    )
    skylight_ratio: float = Field(
        description="The ratio of skylight area to floor area.",
        ge=0,
        le=0.95,
        unit="unitless",
    )
    # FABRIC
    wall_u_value_N: float = Field(
        description="The U-value of walls in the N orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    wall_u_value_NE: float = Field(
        description="The U-value of walls in the NE orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    wall_u_value_E: float = Field(
        description="The U-value of walls in the E orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    wall_u_value_SE: float = Field(
        description="The U-value of walls in the SE orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    wall_u_value_S: float = Field(
        description="The U-value of walls in the S orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    wall_u_value_SW: float = Field(
        description="The U-value of walls in the SW orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    wall_u_value_W: float = Field(
        description="The U-value of walls in the W orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    wall_u_value_NW: float = Field(
        description="The U-value of walls in the NW orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    wall_sri: float = Field(
        description="The Solar Reflectance Index of the walls.",
        ge=0,
        le=122,
        unit="unitless",
    )
    floor_u_value: float = Field(
        description="The U-value of the ground floor.", ge=0.05, le=6, unit="W/m2K"
    )
    roof_u_value: float = Field(
        description="The U-value of the roof.", ge=0.05, le=6, unit="W/m2K"
    )
    window_u_value_N: float = Field(
        description="The U-value of the windows to the N orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    window_u_value_NE: float = Field(
        description="The U-value of the windows to the NE orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    window_u_value_E: float = Field(
        description="The U-value of the windows to the E orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    window_u_value_SE: float = Field(
        description="The U-value of the windows to the SE orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    window_u_value_S: float = Field(
        description="The U-value of the windows to the S orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    window_u_value_SW: float = Field(
        description="The U-value of the windows to the SW orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    window_u_value_W: float = Field(
        description="The U-value of the windows to the W orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    window_u_value_NW: float = Field(
        description="The U-value of the windows to the NW orientation.",
        ge=0.05,
        le=6,
        unit="W/m2K",
    )
    window_shgc_N: float = Field(
        description="The Solar Heat Gain Coefficient of the windows in the N orientation.",
        ge=0,
        le=1,
        unit="unitless",
    )
    window_shgc_NE: float = Field(
        description="The Solar Heat Gain Coefficient of the windows in the NE orientation.",
        ge=0,
        le=1,
        unit="unitless",
    )
    window_shgc_E: float = Field(
        description="The Solar Heat Gain Coefficient of the windows in the E orientation.",
        ge=0,
        le=1,
        unit="unitless",
    )
    window_shgc_SE: float = Field(
        description="The Solar Heat Gain Coefficient of the windows in the SE orientation.",
        ge=0,
        le=1,
        unit="unitless",
    )
    window_shgc_S: float = Field(
        description="The Solar Heat Gain Coefficient of the windows in the S orientation.",
        ge=0,
        le=1,
        unit="unitless",
    )
    window_shgc_SW: float = Field(
        description="The Solar Heat Gain Coefficient of the windows in the SW orientation.",
        ge=0,
        le=1,
        unit="unitless",
    )
    window_shgc_W: float = Field(
        description="The Solar Heat Gain Coefficient of the windows in the W orientation.",
        ge=0,
        le=1,
        unit="unitless",
    )
    window_shgc_NW: float = Field(
        description="The Solar Heat Gain Coefficient of the windows in the NW orientation.",
        ge=0,
        le=1,
        unit="unitless",
    )
    skylight_u_value: float = Field(
        description="The U-value of the skylight.", ge=0.05, le=6, unit="W/m2K"
    )
    skylight_shgc: float = Field(
        description="The Solar Heat Gain Coefficient of the skylight.",
        ge=0,
        le=1,
        unit="unitless",
    )
    roof_sri: float = Field(
        description="The Solar Reflectance Index of the roof.",
        ge=0,
        le=122,
        unit="unitless",
    )
    # PROGRAM
    occupant_density: float = Field(
        description="The number of occupants per area (person/m2).",
        ge=0,
        unit="person/m2",
    )
    lighting_power_density: float = Field(
        description="The lighting power per area (W/m2).", ge=0, unit="W/m2"
    )
    equipment_power_density: float = Field(
        description="The equipment power per area (W/m2).", ge=0, unit="W/m2"
    )
    infiltration_rate: float = Field(
        description="The infiltration rate (m3/s/m2).", ge=0, unit="m3/s/m2"
    )
    ventilation_rate_flowperperson: float = Field(
        description="The ventilation rate (m3/s/person).", ge=0, unit="m3/s/person"
    )
    ventilation_rate_ach: float = Field(
        description="The ventilation rate (ach).", ge=0, unit="ach"
    )
    ventilation_rate_flowperarea: float = Field(
        description="The ventilation rate (m3/s/m2).", ge=0, unit="m3/s/m2"
    )
    heating_setpoint: float = Field(
        description="The heating setpoint temperature (C).", ge=0, unit="C"
    )
    heating_setback: float = Field(
        description="The heating setback temperature (C).", ge=0, unit="C"
    )
    cooling_setpoint: float = Field(
        description="The cooling setpoint temperature (C).", ge=0, unit="C"
    )
    cooling_setback: float = Field(
        description="The cooling setback temperature (C).", ge=0, unit="C"
    )
    humidifying_setpoint: float = Field(
        description="The humidifying setpoint (%RH).", ge=0, le=100, unit="%"
    )
    humidifying_setback: float = Field(
        description="The humidifying setback (%RH).", ge=0, le=100, unit="%"
    )
    dehumidifying_setpoint: float = Field(
        description="The dehumidifying setpoint (%RH).", ge=0, le=100, unit="%"
    )
    dehumidifying_setback: float = Field(
        description="The dehumidifying setback (%RH).", ge=0, le=100, unit="%"
    )
    # SYSTEM
    economizer_type: EconomizerType = Field(
        description="The type of economizer.", unit="enum"
    )
    sensible_heat_recovery_effectiveness: float = Field(
        description="The effectiveness of sensible heat recovery.",
        ge=0,
        le=1,
        unit="unitless",
    )
    latent_heat_recovery_effectiveness: float = Field(
        description="The effectiveness of latent heat recovery.",
        ge=0,
        le=1,
        unit="unitless",
    )
    demand_controlled_ventilation: bool = Field(
        description="Whether demand-controlled ventilation is used.", unit="bool"
    )
    daylight_dimming: bool = Field(
        description="Whether daylight dimming is used.", unit="bool"
    )
    heating_cop: float = Field(
        description="The heating coefficient of performance (COP).",
        ge=0,
        le=5,
        unit="unitless",
    )
    cooling_eer: float = Field(
        description="The cooling energy efficiency ratio (EER).",
        ge=0,
        le=13,
        unit="unitless",
    )
    fan_power: float = Field(description="The fan power (W/l/s).", ge=0, unit="W/l/s")
    pump_power: float = Field(description="The pump power (W/l/s).", ge=0, unit="W/l/s")
    # META
    metadata: dict = Field(
        description="Any additional metadata to store with the typology - useful for sorting and grouping later.",
        unit="dict",
    )
    masterplan_identifier: str = Field(
        description="The name of the masterplan this typology is associated with.",
        unit="str",
        default=DEFAULT_MASTERPLAN_IDENTIFIER,
    )
    # endregion: ATTRIBUTES

    # region: DUNDER

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.masterplan_identifier}::{self.identifier})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: "Typology") -> bool:
        """Override for equality as enum handling gets a bit tricky."""
        for k, v in self.dict().items():
            if isinstance(v, Enum):
                if v.value != other.dict()[k].value:
                    return False
            else:
                if v != other.dict()[k]:
                    return False
        return True

    # endregion: DUNDER

    # pylint: disable=no-self-argument,too-many-branches
    @root_validator(pre=True)
    def validate_atts(cls, values):
        """Validate the attributes, ensuring that certain values work with each other."""

        identifier = values.get("identifier")
        valid_string(identifier)

        masterplan_identifier = values.get("masterplan_identifier")
        valid_string(masterplan_identifier)

        epw_file = Path(values.get("epw_file"))
        assert epw_file.exists(), f"{epw_file.absolute()} file does not exist."

        metadata = values.get("metadata")
        if pd.isnull(metadata):
            metadata = {}
        elif isinstance(metadata, str):
            try:
                metadata = dict(
                    (k.strip(), v.strip())
                    for k, v in (item.split(":") for item in metadata.split(","))
                )
            except ValueError:
                raise ValueError(
                    f"Metadata must be a dictionary or a comma-separated string of key:value pairs."
                )
        for k, v in metadata.items():
            if not isinstance(k, str):
                raise ValueError(f"Metadata key {k} must be a string.")
            if not isinstance(v, (str, int, float, bool)):
                raise ValueError(
                    f"Metadata value {v} must be a string, int, float or bool."
                )

        if values["heating_setpoint"] < values["heating_setback"]:
            raise ValueError(
                f"Heating setpoint ({values['heating_setpoint']}) must be greater than heating setback ({values['heating_setback']})."
            )

        if values["cooling_setpoint"] > values["cooling_setback"]:
            raise ValueError(
                f"Cooling setpoint ({values['cooling_setpoint']}) must be less than cooling setback ({values['cooling_setback']})."
            )

        if values["heating_setpoint"] > values["cooling_setpoint"]:
            raise ValueError(
                f"Heating setpoint ({values['heating_setpoint']}) must be less than cooling setpoint ({values['cooling_setpoint']})."
            )

        if values["heating_setback"] > values["cooling_setback"]:
            raise ValueError(
                f"Heating setback ({values['heating_setback']}) must be less than cooling setback ({values['cooling_setback']})."
            )

        if values["humidifying_setpoint"] < values["humidifying_setback"]:
            raise ValueError(
                f"Humidifying setpoint ({values['humidifying_setpoint']}) must be less than humidifying setback ({values['humidifying_setback']})."
            )

        if values["humidifying_setpoint"] > values["dehumidifying_setpoint"]:
            raise ValueError(
                f"Humidifying setpoint ({values['humidifying_setpoint']}) must be less than dehumidifying setpoint ({values['dehumidifying_setpoint']})."
            )

        if values["humidifying_setback"] > values["dehumidifying_setback"]:
            raise ValueError(
                f"Humidifying setback ({values['humidifying_setback']}) must be less than dehumidifying setback ({values['dehumidifying_setback']})."
            )

        if values["dehumidifying_setpoint"] > values["dehumidifying_setback"]:
            raise ValueError(
                f"Dehumidifying setpoint ({values['dehumidifying_setpoint']}) must be less than dehumidifying setback ({values['dehumidifying_setback']})."
            )

        return values

    # pylint: enable=no-self-argument,too-many-branches

    # region: CLASSMETHODS

    @classmethod
    def random(cls, epw_file: Path = None, seed: int = None) -> "Typology":
        """Generate a random typology."""

        np.random.seed(seed)

        if epw_file is None:
            # reference the test EPW here ... this is bad practice, but meh
            epw_file = Path(__file__).absolute().parent / "test" / "test.epw"
            logger.info(f"{__class__.__name__} - Using default EPW: {epw_file}")

        return cls(
            identifier=random_id(seed),
            total_area=np.random.uniform(0.1, 1000),
            epw_file=epw_file,
            building_type=np.random.choice(list(BuildingType)),
            vintage=np.random.choice(list(Vintage)),
            average_footprint_area=np.random.uniform(0.1, 1000),
            average_num_floors=np.random.uniform(1, 10),
            average_floor_height=np.random.uniform(0.5, 5),
            rotation=np.random.uniform(0, 360),
            terrain=np.random.choice(list(TerrainType)),
            skylight_ratio=np.random.uniform(0, 0.95),
            glazing_ratio_N=np.random.uniform(0, 0.95),
            glazing_ratio_NE=np.random.uniform(0, 0.95),
            glazing_ratio_E=np.random.uniform(0, 0.95),
            glazing_ratio_SE=np.random.uniform(0, 0.95),
            glazing_ratio_S=np.random.uniform(0, 0.95),
            glazing_ratio_SW=np.random.uniform(0, 0.95),
            glazing_ratio_W=np.random.uniform(0, 0.95),
            glazing_ratio_NW=np.random.uniform(0, 0.95),
            # #
            wall_u_value_N=np.random.uniform(0.05, 6),
            wall_u_value_NE=np.random.uniform(0.05, 6),
            wall_u_value_E=np.random.uniform(0.05, 6),
            wall_u_value_SE=np.random.uniform(0.05, 6),
            wall_u_value_S=np.random.uniform(0.05, 6),
            wall_u_value_SW=np.random.uniform(0.05, 6),
            wall_u_value_W=np.random.uniform(0.05, 6),
            wall_u_value_NW=np.random.uniform(0.05, 6),
            wall_sri=np.random.uniform(0, 122),
            floor_u_value=np.random.uniform(0.05, 6),
            roof_u_value=np.random.uniform(0.05, 6),
            window_u_value_N=np.random.uniform(0.05, 6),
            window_u_value_NE=np.random.uniform(0.05, 6),
            window_u_value_E=np.random.uniform(0.05, 6),
            window_u_value_SE=np.random.uniform(0.05, 6),
            window_u_value_S=np.random.uniform(0.05, 6),
            window_u_value_SW=np.random.uniform(0.05, 6),
            window_u_value_W=np.random.uniform(0.05, 6),
            window_u_value_NW=np.random.uniform(0.05, 6),
            window_shgc_N=np.random.uniform(0, 1),
            window_shgc_NE=np.random.uniform(0, 1),
            window_shgc_E=np.random.uniform(0, 1),
            window_shgc_SE=np.random.uniform(0, 1),
            window_shgc_S=np.random.uniform(0, 1),
            window_shgc_SW=np.random.uniform(0, 1),
            window_shgc_W=np.random.uniform(0, 1),
            window_shgc_NW=np.random.uniform(0, 1),
            skylight_u_value=np.random.uniform(0.05, 6),
            skylight_shgc=np.random.uniform(0, 1),
            roof_sri=np.random.uniform(0, 122),
            # #
            occupant_density=np.random.uniform(0.1, 0.5),
            lighting_power_density=np.random.uniform(5, 10),
            equipment_power_density=np.random.uniform(5, 10),
            infiltration_rate=np.random.uniform(0, 0.0005),
            ventilation_rate_flowperperson=np.random.uniform(0, 0.0005),
            ventilation_rate_ach=np.random.uniform(0, 1),
            ventilation_rate_flowperarea=np.random.uniform(0, 0.0004),
            heating_setpoint=np.random.uniform(20, 22),
            heating_setback=np.random.uniform(15, 17),
            cooling_setpoint=np.random.uniform(24, 26),
            cooling_setback=np.random.uniform(27, 29),
            humidifying_setpoint=np.random.uniform(40, 59),
            humidifying_setback=np.random.uniform(0, 20),
            dehumidifying_setpoint=np.random.uniform(60, 79),
            dehumidifying_setback=np.random.uniform(80, 100),
            # #
            economizer_type=np.random.choice(list(EconomizerType)),
            sensible_heat_recovery_effectiveness=np.random.uniform(0, 1),
            latent_heat_recovery_effectiveness=np.random.uniform(0, 1),
            demand_controlled_ventilation=np.random.choice([True, False]),
            daylight_dimming=np.random.choice([True, False]),
            heating_cop=np.random.uniform(0, 5),
            cooling_eer=np.random.uniform(0, 13),
            fan_power=np.random.uniform(0, 5),
            pump_power=np.random.uniform(0, 5),
            # #
            metadata={
                "random": True,
                "seed": seed,
                "category_example": np.random.choice(["A", "B", "C"]),
            },
            masterplan_identifier="RandomMasterplan",
        )

    @classmethod
    def from_building_type(
        cls,
        building_type: BuildingType | str,
        total_area: float,
        epw_file: Path,
        identifier: str = None,
        vintage: Vintage | str = None,
        rotation: float = None,
        terrain: TerrainType | str = None,
        metadata: dict = None,
        masterplan_identifier: str = None,
    ) -> "Typology":
        """Create this object based on defaults for the given building type.

        Args:
            building_type (BuildingType): The building type.
            total_area (float): The total area of the building.
            epw_file (Path): The EPW file to use for the simulation.
            identifier (str, optional): A unique identifier for the typology. Defaults to the building_type.
            vintage (Vintage, optional): The vintage of the building. Defaults to ASHRAE_901_2019.
            rotation (float, optional): The rotation of the building. Defaults to 0.
            terrain (TerrainType, optional): The terrain type. Defaults to TerrainType.URBAN.
            metadata (dict, optional): Any additional metadata to store with the typology. Defaults to None.
            masterplan_identifier (str, optional): The name of the masterplan this typology is associated with. Defaults to MASTERPLAN_IDENTIFIER.

        Returns:
            Typology: The typology object.
        """

        epw = EPW(epw_file)

        if isinstance(building_type, str):
            building_type = BuildingType(building_type)

        if identifier is None:
            identifier = building_type.value
            logger.info(
                f"{__class__.__name__} - Using default identifier: {identifier}"
            )

        if masterplan_identifier is None:
            masterplan_identifier = DEFAULT_MASTERPLAN_IDENTIFIER
            logger.info(
                f"{__class__.__name__} - Using default masterplan_identifier: {masterplan_identifier}"
            )

        if vintage is None:
            vintage = Vintage.ASHRAE_901_2019
            logger.info(f"{__class__.__name__} - Using default vintage: {vintage}")
        elif isinstance(vintage, str):
            vintage = Vintage(vintage)

        if rotation is None:
            rotation = 0
            logger.info(f"{__class__.__name__} - Using default rotation: {rotation}")

        if terrain is None:
            terrain = TerrainType.URBAN
            logger.info(f"{__class__.__name__} - Using default terrain: {terrain}")
        elif isinstance(terrain, str):
            terrain = TerrainType(terrain)

        if metadata is None:
            metadata = {}
            logger.info(f"{__class__.__name__} - Using default metadata: {metadata}")

        d = {
            "identifier": building_type.value if identifier is None else identifier,
            "total_area": total_area,
            "epw_file": Path(epw_file),
            "building_type": building_type,
            "vintage": vintage,
            "rotation": rotation,
            "terrain": terrain,
            "metadata": metadata,
            "masterplan_identifier": masterplan_identifier,
        }

        # region: DEFAULT_FORM

        d["average_footprint_area"] = default_footprint_area(building_type)
        d["average_num_floors"] = default_number_of_floors(building_type)
        d["average_floor_height"] = default_floor_height(building_type)
        gr = default_glazing_ratio(building_type)
        d["glazing_ratio_N"] = gr
        d["glazing_ratio_NE"] = gr
        d["glazing_ratio_E"] = gr
        d["glazing_ratio_SE"] = gr
        d["glazing_ratio_S"] = gr
        d["glazing_ratio_SW"] = gr
        d["glazing_ratio_W"] = gr
        d["glazing_ratio_NW"] = gr
        d["skylight_ratio"] = default_skylight_ratio(building_type)

        # endregion: DEFAULT_FORM

        # region: DEFAULT_FABRIC

        construction_type = default_construction_type(building_type)
        constr_set = default_constructionset(
            construction_type=construction_type, epw=epw, vintage=vintage
        )

        d["wall_u_value_N"] = constr_set.wall_set.exterior_construction.u_factor
        d["wall_u_value_NE"] = constr_set.wall_set.exterior_construction.u_factor
        d["wall_u_value_E"] = constr_set.wall_set.exterior_construction.u_factor
        d["wall_u_value_SE"] = constr_set.wall_set.exterior_construction.u_factor
        d["wall_u_value_S"] = constr_set.wall_set.exterior_construction.u_factor
        d["wall_u_value_SW"] = constr_set.wall_set.exterior_construction.u_factor
        d["wall_u_value_W"] = constr_set.wall_set.exterior_construction.u_factor
        d["wall_u_value_NW"] = constr_set.wall_set.exterior_construction.u_factor
        d["wall_sri"] = construction_sri(constr_set.wall_set.exterior_construction)
        d["floor_u_value"] = constr_set.floor_set.ground_construction.u_factor
        d["roof_u_value"] = constr_set.roof_ceiling_set.exterior_construction.u_factor
        d["window_u_value_N"] = constr_set.aperture_set.window_construction.u_factor
        d["window_u_value_NE"] = constr_set.aperture_set.window_construction.u_factor
        d["window_u_value_E"] = constr_set.aperture_set.window_construction.u_factor
        d["window_u_value_SE"] = constr_set.aperture_set.window_construction.u_factor
        d["window_u_value_S"] = constr_set.aperture_set.window_construction.u_factor
        d["window_u_value_SW"] = constr_set.aperture_set.window_construction.u_factor
        d["window_u_value_W"] = constr_set.aperture_set.window_construction.u_factor
        d["window_u_value_NW"] = constr_set.aperture_set.window_construction.u_factor
        d["window_shgc_N"] = constr_set.aperture_set.window_construction.shgc
        d["window_shgc_NE"] = constr_set.aperture_set.window_construction.shgc
        d["window_shgc_E"] = constr_set.aperture_set.window_construction.shgc
        d["window_shgc_SE"] = constr_set.aperture_set.window_construction.shgc
        d["window_shgc_S"] = constr_set.aperture_set.window_construction.shgc
        d["window_shgc_SW"] = constr_set.aperture_set.window_construction.shgc
        d["window_shgc_W"] = constr_set.aperture_set.window_construction.shgc
        d["window_shgc_NW"] = constr_set.aperture_set.window_construction.shgc
        d["skylight_u_value"] = constr_set.aperture_set.skylight_construction.u_factor
        d["skylight_shgc"] = constr_set.aperture_set.skylight_construction.shgc
        d["roof_sri"] = construction_sri(
            constr_set.roof_ceiling_set.exterior_construction
        )
        # endregion: DEFAULT_FABRIC

        # region: DEFAULT_PROGRAM
        program_type = default_program_type(building_type)

        try:
            d["occupant_density"] = program_type.people.people_per_area
        except (AttributeError, KeyError):
            d["occupant_density"] = 0

        try:
            d["lighting_power_density"] = program_type.lighting.watts_per_area
        except (AttributeError, KeyError):
            d["lighting_power_density"] = 0

        try:
            d["equipment_power_density"] = (
                program_type.electric_equipment.watts_per_area
            )
        except (AttributeError, KeyError):
            d["equipment_power_density"] = 0

        try:
            d["infiltration_rate"] = program_type.infiltration.flow_per_exterior_area
        except (AttributeError, KeyError):
            d["infiltration_rate"] = 0

        try:
            d["ventilation_rate_flowperperson"] = (
                program_type.ventilation.flow_per_person
            )
        except (AttributeError, KeyError):
            d["ventilation_rate_flowperperson"] = 0

        try:
            d["ventilation_rate_ach"] = program_type.ventilation.air_changes_per_hour
        except (AttributeError, KeyError):
            d["ventilation_rate_ach"] = 0

        try:
            d["ventilation_rate_flowperarea"] = program_type.ventilation.flow_per_area
        except (AttributeError, KeyError):
            d["ventilation_rate_flowperarea"] = 0

        try:
            d["heating_setpoint"] = program_type.setpoint.heating_setpoint
        except (AttributeError, KeyError):
            d["heating_setpoint"] = 20

        try:
            d["heating_setback"] = program_type.setpoint.heating_setback
        except (AttributeError, KeyError):
            d["heating_setback"] = 15

        try:
            d["cooling_setpoint"] = program_type.setpoint.cooling_setpoint
        except (AttributeError, KeyError):
            d["cooling_setpoint"] = 24

        try:
            d["cooling_setback"] = program_type.setpoint.cooling_setback
        except (AttributeError, KeyError):
            d["cooling_setback"] = 27

        if program_type.setpoint.humidifying_setpoint is None:
            d["humidifying_setpoint"] = 40
        else:
            d["humidifying_setpoint"] = program_type.setpoint.humidifying_setpoint

        if program_type.setpoint.humidifying_setback is None:
            d["humidifying_setback"] = 0
        else:
            d["humidifying_setback"] = program_type.setpoint.humidifying_setback

        if program_type.setpoint.dehumidifying_setpoint is None:
            d["dehumidifying_setpoint"] = 60
        else:
            d["dehumidifying_setpoint"] = program_type.setpoint.dehumidifying_setpoint

        if program_type.setpoint.dehumidifying_setback is None:
            d["dehumidifying_setback"] = 80
        else:
            d["dehumidifying_setback"] = program_type.setpoint.dehumidifying_setback

        # endregion: DEFAULT_PROGRAM

        # region: DEFAULT_SYSTEM

        d["economizer_type"] = default_economizer_type(
            building_type=building_type, vintage=vintage, epw=epw
        )

        shre, lhre = default_hr_effectiveness(
            building_type=building_type, vintage=vintage, epw=epw
        )
        d["sensible_heat_recovery_effectiveness"] = shre
        d["latent_heat_recovery_effectiveness"] = lhre
        d["demand_controlled_ventilation"] = default_demand_controlled_ventilation(
            building_type=building_type, vintage=vintage
        )
        d["daylight_dimming"] = default_daylight_dimming(
            building_type=building_type, vintage=vintage
        )
        d["heating_cop"] = default_heating_cop(
            building_type=building_type, vintage=vintage, epw=epw
        )
        d["cooling_eer"] = default_cooling_eer(
            building_type=building_type, vintage=vintage, epw=epw
        )
        d["fan_power"] = default_fan_power(building_type=building_type, vintage=vintage)
        d["pump_power"] = default_pump_power(
            building_type=building_type, vintage=vintage
        )

        # endregion: DEFAULT_SYSTEM

        obj = cls.parse_obj(d)

        # logger.info(f"{obj} - Created from {program_type}")

        return obj

    @classmethod
    def from_dict(cls, d: dict, use_defaults: bool = True) -> "Typology":
        """Create a typology from a dictionary, with the optional filling of missing data using default values.

        Args:
            data (dict): The data to create the typology from.
            use_defaults (bool, optional): Whether to fill missing data with default values. Defaults to True.

        Returns:
            Typology: The typology object.
        """

        if not use_defaults:
            return cls.parse_obj(d)

        # ensure all keys are present
        for k in cls.__fields__.keys():
            if k not in d:
                raise ValueError(f"'{k}' must be provided in the dictionary.")

        # make sure at least the bare minimum of values are present
        for k in ["identifier", "building_type", "total_area", "epw_file"]:
            if pd.isnull(d[k]):
                raise ValueError(
                    f"'{k}' must be provided if defaults are to be populated."
                )

        # handle conversion of any enums
        if d["building_type"] is not None:
            if isinstance(d["building_type"], str):
                d["building_type"] = BuildingType(str(d["building_type"]))

        if d["terrain"] is not None:
            if isinstance(d["terrain"], str):
                d["terrain"] = TerrainType(str(d["terrain"]))

        if d["vintage"] is not None:
            if isinstance(d["vintage"], (str, int)):
                d["vintage"] = Vintage(str(d["vintage"]))

        if d["economizer_type"] is not None:
            if isinstance(d["economizer_type"], str):
                d["economizer_type"] = EconomizerType(str(d["economizer_type"]))

        # handle conversion of other remaining fields
        if d["metadata"] is not None:
            if pd.isnull(d["metadata"]):
                d["metadata"] = {}
            if isinstance(d["metadata"], str):
                try:
                    d["metadata"] = dict(
                        (k.strip(), v.strip())
                        for k, v in (
                            item.split(":") for item in d["metadata"].split(",")
                        )
                    )
                except ValueError:
                    raise ValueError(
                        f"Metadata must be a dictionary or a comma-separated string of key:value pairs."
                    )
            for k, v in d["metadata"].items():
                if not all([isinstance(k, str), isinstance(v, str)]):
                    raise ValueError(
                        f"Metadata key {k} and value {v} must both be strings."
                    )

        # create the default typology
        default_typology = cls.from_building_type(
            building_type=d["building_type"],
            total_area=d["total_area"],
            epw_file=d["epw_file"],
            identifier=d["identifier"],
            vintage=d["vintage"],
            rotation=d["rotation"],
            terrain=d["terrain"],
            metadata=d["metadata"],
            masterplan_identifier=d["masterplan_identifier"],
        )

        # overwrite any nulls with default values
        for k, v in d.items():
            if pd.isnull(v):
                val = getattr(default_typology, k)
                logger.info(f"Using default value {val} for {k} in {d['identifier']}.")
                d[k] = val

        return cls.parse_obj(d)

    def series(self) -> pd.Series:
        """Return the typology as a pandas Series."""
        return pd.Series(self.dict())

    @classmethod
    def from_series(cls, s: pd.Series, use_defaults: bool = True) -> "Typology":
        """Create a typology from a pandas Series, with the optional filling of missing data using default values.

        Args:
            s (pd.Series): The series to create the typology from.
            use_defaults (bool, optional): Whether to fill missing data with default values. Defaults to True.

        Returns:
            Typology: The typology object.
        """

        d = s.to_dict()

        return cls.from_dict(d, use_defaults)

    # endregion: CLASSMETHODS

    # region: PROPERTIES

    @property
    def basic_summary(self) -> str:
        """Return a basic summary of the typology."""

        print(
            "\n".join(
                [
                    f"masterplan_identifier = {self.masterplan_identifier}",
                    f"identifier = {self.identifier}",
                    f"building_type = {self.building_type}",
                    f"vintage = {self.vintage}",
                    f"total_area = {self.total_area}",
                    f"typical_gfa = {self.typical_gfa}",
                    f"number_of_buildings = {self.number_of_buildings}",
                    f"average_num_floors = {self.average_num_floors}",
                ]
            )
        )

    @property
    def epw(self) -> EPW:
        """Return the EPW object."""
        return EPW(self.epw_file)

    @property
    def _glazing_ratios(self) -> list[float]:
        """The glazing ratio for each cardinal orientation."""
        return [
            self.glazing_ratio_N,
            self.glazing_ratio_NE,
            self.glazing_ratio_E,
            self.glazing_ratio_SE,
            self.glazing_ratio_S,
            self.glazing_ratio_SW,
            self.glazing_ratio_W,
            self.glazing_ratio_NW,
        ]

    @property
    def _wall_u_values(self) -> list[float]:
        """The U-values of the walls in each orientation."""
        return [
            self.wall_u_value_N,
            self.wall_u_value_NE,
            self.wall_u_value_E,
            self.wall_u_value_SE,
            self.wall_u_value_S,
            self.wall_u_value_SW,
            self.wall_u_value_W,
            self.wall_u_value_NW,
        ]

    @property
    def _window_u_values(self) -> list[float]:
        """The U-values of the windows in each orientation."""
        return [
            self.window_u_value_N,
            self.window_u_value_NE,
            self.window_u_value_E,
            self.window_u_value_SE,
            self.window_u_value_S,
            self.window_u_value_SW,
            self.window_u_value_W,
            self.window_u_value_NW,
        ]

    @property
    def _window_shgcs(self) -> list[float]:
        """The Solar Heat Gain Coefficients of the windows in each cardinal
        orientation."""
        return [
            self.window_shgc_N,
            self.window_shgc_NE,
            self.window_shgc_E,
            self.window_shgc_SE,
            self.window_shgc_S,
            self.window_shgc_SW,
            self.window_shgc_W,
            self.window_shgc_NW,
        ]

    @property
    def typical_gfa(self) -> float:
        """Get the typical gross floor area for a building of this typology."""
        return self.total_area / self.number_of_buildings

    @property
    def building_height(self) -> float:
        """Get the typical height for an individual building."""
        return self.average_num_floors * self.average_floor_height

    @property
    def number_of_buildings(self) -> float:
        """Return the number of buildings this Typology represents"""
        return self.total_area / (self.average_footprint_area * self.average_num_floors)

    @property
    def _simulation_directory(self) -> Path:
        """Lightweight helper method to get the target directory for the
        typology simulation results."""
        return (
            DEFAULT_SIMULATION_DIRECTORY / self.masterplan_identifier / self.identifier
        )

    @property
    def _openstudio_directory(self) -> Path:
        """Lightweight helper method to get the OpenStudio directory for the
        typology simulation results."""
        return self._simulation_directory / "openstudio"

    @property
    def _sql_file(self) -> Path:
        """Lightweight helper method to get the SQL file for the typology
        simulation results."""
        return self._openstudio_directory / "run/eplusout.sql"

    @property
    def _osw_file(self) -> Path:
        """Lightweight helper method to get the OSW file for the typology
        simulation results."""
        return self._openstudio_directory / "workflow.osw"

    @property
    def _hbjson_file(self) -> Path:
        """Lightweight helper method to get the HBJSON file for the typology
        simulation results."""
        return self._openstudio_directory / f"{self.identifier}.hbjson"

    @property
    def _config_file(self) -> Path:
        """Lightweight helper method to get the config file for the typology
        simulation results."""
        return self._simulation_directory / "mped_config.json"

    def _sql_file_exists(self) -> bool:
        """Check if the sql file containing results for this typology already
        exist.

        Returns:
            bool:
                True if results exist, False otherwise.
        """

        # check config is the same
        if not self._config_file.exists():
            return False
        if Typology.parse_file(self._config_file) != self:
            return False

        # check EPW is the same
        if not self._osw_file.exists():
            return False
        with open(self._osw_file, "r", encoding="utf-8") as fp:
            old_epw_file = Path(json.load(fp)["weather_file"])
        if old_epw_file.name != self.epw_file.name:
            return False

        # check SQL file exists
        if self._sql_file.exists():
            return True

        return False

    # endregion: PROPERTIES

    # region: MODEL_METHODS

    def _program_type(self) -> ProgramType:
        """Create a programtype from this object."""

        base_program = default_program_type(self.building_type)
        program = base_program.duplicate()
        program.unlock()

        if program.people is not None:
            program.people.people_per_area = self.occupant_density

        if program.lighting is not None:
            program.lighting.watts_per_area = self.lighting_power_density

        if program.electric_equipment is not None:
            program.electric_equipment.watts_per_area = self.equipment_power_density

        if program.infiltration is not None:
            program.infiltration.flow_per_exterior_area = self.infiltration_rate

        if program.ventilation is not None:
            program.ventilation.flow_per_person = self.ventilation_rate_flowperperson
            program.ventilation.air_changes_per_hour = self.ventilation_rate_ach
            program.ventilation.flow_per_area = 0

        if program.setpoint is not None:
            old_heating_schedule = np.array(
                program.setpoint.heating_schedule.data_collection().values
            )
            new_heating_schedule = np.interp(
                old_heating_schedule,
                [old_heating_schedule.min(), old_heating_schedule.max()],
                [self.heating_setback, self.heating_setpoint],
            )
            program.setpoint.heating_schedule = ScheduleFixedInterval(
                "Heating Schedule",
                new_heating_schedule,
                temperature,
            )

            old_cooling_schedule = np.array(
                program.setpoint.cooling_schedule.data_collection().values
            )
            new_cooling_schedule = np.interp(
                old_cooling_schedule,
                [old_cooling_schedule.max(), old_cooling_schedule.min()],
                [self.cooling_setback, self.cooling_setpoint],
            )
            program.setpoint.cooling_schedule = ScheduleFixedInterval(
                "Cooling Schedule",
                new_cooling_schedule,
                temperature,
            )

            if program.setpoint.humidifying_setpoint is None:
                program.setpoint.humidifying_schedule = ScheduleFixedInterval(
                    "Humidifying Schedule",
                    np.where(
                        np.array(
                            program.setpoint.heating_schedule.data_collection.values
                        )
                        == program.setpoint.heating_setpoint,
                        self.humidifying_setpoint,
                        self.humidifying_setback,
                    ),
                    humidity,
                )
                program.setpoint.dehumidifying_schedule = ScheduleFixedInterval(
                    "Dehumidifying Schedule",
                    np.where(
                        np.array(
                            program.setpoint.heating_schedule.data_collection.values
                        )
                        == program.setpoint.heating_setpoint,
                        self.dehumidifying_setpoint,
                        self.dehumidifying_setback,
                    ),
                    humidity,
                )
            else:
                try:
                    program.setpoint.humidifying_setback = self.humidifying_setback
                except AttributeError:
                    pass
                program.setpoint.dehumidifying_setpoint = self.dehumidifying_setpoint
                try:
                    program.setpoint.dehumidifying_setback = self.dehumidifying_setback
                except AttributeError:
                    pass
                program.lock()

        return program

    def _internal_gains(
        self, as_dataframe: bool = False
    ) -> HourlyContinuousCollection | pd.DataFrame:
        """Get the internal gains for the typology."""

        program = self._program_type()

        people_gain = (
            collection_to_series(occupancy_schedule_from_program(program=program))
            * collection_to_series(program.people.activity_schedule.data_collection())
            * program.people.people_per_area
        ).rename("Energy Intensity (Wh/m2)")

        lighting_gain = (
            collection_to_series(program.lighting.schedule.data_collection())
            * program.lighting.watts_per_area
        ).rename("Energy Intensity (Wh/m2)")

        electric_equipment_gain = (
            collection_to_series(program.electric_equipment.schedule.data_collection())
            * program.electric_equipment.watts_per_area
        ).rename("Energy Intensity (Wh/m2)")

        if as_dataframe:
            return pd.concat(
                [
                    people_gain,
                    lighting_gain,
                    electric_equipment_gain,
                ],
                axis=1,
                keys=["People (Wh/m2)", "Lighting (Wh/m2)", "Electric Equipment (Wh/m2)"],
            )

        return {
            "people": collection_from_series(people_gain),
            "lighting": collection_from_series(lighting_gain),
            "electric_equipment": collection_from_series(electric_equipment_gain),
        }

    def _population(self, per_building: bool = True) -> pd.Series:
        """Get the number of occpants within this typology, either per
        building of per the entire typology.

        Args:
            per_building (bool, optional): Whether to return the number of
                occupants per building or for the entire typology. Defaults to
                True.

        Returns:
            pd.Series: The number of occupants.
        """
        occupancy_profile = (
            collection_to_series(occupancy_schedule_from_program(self._program_type()))
            * self.occupant_density
        )  # person/m2
        if per_building:
            return occupancy_profile * self.typical_gfa
        return occupancy_profile * self.total_area

    def _ideal_air(self) -> IdealAirSystem:
        """Return the ideal air system associated with the system."""

        return IdealAirSystem(
            identifier=self.identifier,
            economizer_type=self.economizer_type.value,
            demand_controlled_ventilation=self.demand_controlled_ventilation,
            sensible_heat_recovery=self.sensible_heat_recovery_effectiveness,
            latent_heat_recovery=self.latent_heat_recovery_effectiveness,
        )

    def _footprint(self) -> Polygon2D:
        """Create the footprint for the building."""

        footprint = Polygon2D.from_rectangle(
            base=1,
            height=1,
            base_point=Point2D(),
            height_vector=Vector2D(0, 1),
        )

        footprint = footprint.rotate(
            angle=-np.deg2rad(self.rotation), origin=footprint.center
        )

        # scale the footprint
        if footprint.area < self.average_footprint_area:
            new_area = footprint.area
            n = 1.01
            while new_area < self.average_footprint_area:
                new_area = footprint.scale(n).area
                n += 0.005
        else:
            new_area = footprint.area
            n = 0.99
            while new_area > self.average_footprint_area:
                new_area = footprint.scale(n).area
                n -= 0.005

        return footprint.scale(n)

    def _base_model(
        self,
    ) -> Model:
        """Create the base model containing all geometry, without construction assignment."""

        # TODO - upgrade to ACTUAL model and assign programs

        # create the lookup dict for apertures in each orientation
        gr_lookup = dict(
            zip(
                *[
                    ORIENTATIONS,
                    self._glazing_ratios,
                ]
            )
        )

        # construct rooms for model
        rooms = []
        for level in range(self.average_num_floors):
            room = Room.from_box(
                identifier=f"level_{level:02d}",
                width=self.average_footprint_area**0.5,
                depth=self.average_footprint_area**0.5,
                height=self.average_floor_height,
                orientation_angle=self.rotation,
                origin=Point3D(0, 0, level * self.average_floor_height),
            )
            for face in room.walls:
                face: Face
                if isinstance(face.boundary_condition, Outdoors):
                    cardinal_orientation = face_orientation(face.geometry)
                    face.apertures_by_ratio(gr_lookup[cardinal_orientation], 0.1)
                    face._identifier = f"{room.identifier}_{cardinal_orientation}"  # pylint: disable=W0212
            rooms.append(room)

        # add context shades based on terrain
        context_distance = default_context_shade_distance(self.terrain)
        shades = []
        if context_distance < 500:
            context_height = self.building_height * 0.75
            for segment in self._footprint().offset(-context_distance).segments:
                _shd = Shade(
                    identifier="context_shade",
                    geometry=Face3D.from_extrusion(
                        LineSegment3D.from_line_segment2d(segment),
                        extrusion_vector=Vector3D(0, 0, context_height),
                    ),
                )
                shades.append(_shd)

        model = Model(identifier="base_model", rooms=rooms, orphaned_shades=shades)

        model.solve_adjacency(
            merge_coplanar=False,
            intersect=False,
            overwrite=False,
            air_boundary=False,
            adiabatic=False,
            tolerance=None,
        )

        # add skylight if needed
        model.skylight_apertures_by_ratio(self.skylight_ratio, 0.1)

        # rename apertures just because
        for aperture in model.apertures:
            aperture: Aperture
            aperture._identifier = f"{aperture.parent.identifier}_{aperture._identifier}"  # pylint: disable=W0212

        return model

    def _internal_mass(
        self,
        room: Room,
        epw: EPW,
        vintage: Vintage,
        construction_type: ConstructionType,
    ) -> dict:
        """Determine the internal mass for a building type."""

        constr_set = default_constructionset(
            vintage=vintage, construction_type=construction_type, epw=epw
        )

        d = {
            "ext_wall_area": 0,
            "ground_floor_area": 0,
            "roof_area": 0,
        }

        for face in room.walls:
            d["ext_wall_area"] += face.area
        for face in room.floors:
            if isinstance(face.boundary_condition, Ground):
                d["ground_floor_area"] += face.area
        for face in room.roof_ceilings:
            if isinstance(face.boundary_condition, Outdoors):
                d["roof_area"] += face.area

        masses = [
            InternalMass(
                identifier=f"{room.identifier}_exterior_wall_mass",
                area=d["ext_wall_area"],
                construction=constr_set.wall_set.exterior_construction,
            ),
            InternalMass(
                identifier=f"{room.identifier}_ground_floor_mass",
                area=d["ground_floor_area"],
                construction=constr_set.floor_set.ground_construction,
            ),
            InternalMass(
                identifier=f"{room.identifier}_roof_mass",
                area=d["roof_area"],
                construction=constr_set.roof_ceiling_set.exterior_construction,
            ),
        ]

        # remove massses of 0 area
        masses = [mass for mass in masses if mass.area > 0]

        return masses

    def _constructions(self) -> dict[str, OpaqueConstruction | WindowConstruction]:
        """Create a dictionary of constructions per face type and orientation."""

        # calculate SRI values for walls and roof
        wall_sa, wall_ta = estimate_sri_properties(self.wall_sri)
        roof_sa, roof_ta = estimate_sri_properties(self.roof_sri)

        d = {
            "ground_floor": OpaqueConstruction.from_simple_parameters(
                identifier=f"U {self.floor_u_value:0.2f} Ground Floor",
                r_value=1 / self.floor_u_value,
                roughness="MediumRough",
                thermal_absorptance=0.9,
                solar_absorptance=0.7,
            ),
            "roof": OpaqueConstruction.from_simple_parameters(
                identifier=f"U {self.roof_u_value:0.2f} SRI {self.roof_sri} Roof",
                r_value=1 / self.roof_u_value,
                roughness="MediumRough",
                thermal_absorptance=roof_ta,
                solar_absorptance=roof_sa,
            ),
            "walls": {},
            "windows": {},
            "skylight": WindowConstruction.from_simple_parameters(
                identifier=f"U {self.skylight_u_value:0.2f} SHGC {self.skylight_shgc:0.2f} Skylight",
                u_factor=self.skylight_u_value,
                shgc=self.skylight_shgc,
                vt=0.6,
            ),
        }

        for wall_u, window_u, window_shgc, _dir in zip(
            *[
                self._wall_u_values,
                self._window_u_values,
                self._window_shgcs,
                ORIENTATIONS,
            ]
        ):
            d["walls"][_dir] = OpaqueConstruction.from_simple_parameters(
                identifier=f"U {wall_u:0.2f} SRI {self.wall_sri} Wall",
                r_value=1 / wall_u,
                roughness="MediumRough",
                thermal_absorptance=wall_ta,
                solar_absorptance=wall_sa,
            )
            d["windows"][_dir] = WindowConstruction.from_simple_parameters(
                identifier=f"U {window_u:0.2f} SHGC {window_shgc} Window",
                u_factor=window_u,
                shgc=window_shgc,
                vt=0.6,
            )

        return d

    def _model(self) -> Model:
        """Create a honeybee model for the typology.

        Args:
            epw (EPW):
                The EPW file to use for the simulation.

        Returns:
            Model:
                The honeybee model of the typology.
        """

        model = self._base_model().duplicate()

        program = self._program_type()
        construction_type = default_construction_type(self.building_type)

        constructions = self._constructions()

        for room in model.rooms:
            room: Room

            # assign constructions to faces/apertures by orientation
            for face in room.walls:
                face: Face
                orientation = face_orientation(face)
                face.properties.energy.construction = constructions["walls"][
                    orientation
                ]
                for aperture in face.apertures:
                    aperture.properties.energy.construction = constructions["windows"][
                        orientation
                    ]
            for face in room.roof_ceilings:
                face: Face
                if isinstance(face.boundary_condition, Outdoors):
                    face.properties.energy.construction = constructions["roof"]
                    for aperture in face.apertures:
                        aperture.properties.energy.construction = constructions[
                            "skylight"
                        ]
            for face in room.floors:
                face: Face
                if isinstance(face.boundary_condition, Ground):
                    face.properties.energy.construction = constructions["ground_floor"]

            # assign internal masses to rooms
            for mass in self._internal_mass(
                room, self.epw, self.vintage, construction_type
            ):
                room.properties.energy.add_internal_mass(mass)

            # assign programs to rooms
            room.properties.energy.program_type = program

            # # assign SHW to rooms
            # if room.properties.energy.service_hot_water is not None:
            #     room.properties.energy.shw = self._shw()

            # add daylight dimming
            if self.daylight_dimming:
                room.properties.energy.add_daylight_control_to_center(
                    distance_from_floor=0.8, control_fraction=0.5
                )

            # add system
            room.properties.energy.add_default_ideal_air()
            room.properties.energy.hvac = self._ideal_air()

        # rename
        model.identifier = self.identifier

        return model

    # endregion: MODEL_METHODS

    # region: SIMULATION_METHODS

    def _simulate(self) -> pd.DataFrame:
        """Simulate the typology for a given EPW file and return the results in a DataFrame.

        Args:
            directory (Path):
                The directory to save the results in.

        Returns:
            pd.DataFrame:
                All results of the simulation.
        """

        # TODO - replace with purpose specific simulate method, rather than having all the functionality within this function

        # create model
        logger.disabled = True
        model = self._model()
        logger.disabled = False

        # run simulation if it hasn't already been run
        if not self._sql_file_exists():

            # run simulation
            logger.info(f"{self} - Simulating results")

            # remove old files in target directory just in case
            for f in self._simulation_directory.glob("*"):
                if f.is_file():
                    f.unlink()

            self._openstudio_directory.mkdir(parents=True, exist_ok=True)

            # write config file
            with open(self._config_file, "w", encoding="utf-8") as fp:
                fp.write(self.json(indent=4))

            # write model to target directory to reference in simulation
            model.to_hbjson(folder=self._openstudio_directory)

            # set which outputs are going to be returned
            simulation_control = SimulationControl(
                do_zone_sizing=True,
                do_system_sizing=True,
                do_plant_sizing=True,
                run_for_sizing_periods=False,
                run_for_run_periods=True,
            )
            sizing_parameter = SizingParameter(
                design_days=[
                    self.epw.approximate_design_day("WinterDesignDay"),
                    self.epw.approximate_design_day("SummerDesignDay"),
                ]
            )
            simulation_output = SimulationOutput(
                include_sqlite=True,
                summary_reports=None,
                include_html=False,
            )
            simulation_output.add_zone_energy_use()
            simulation_output.add_hvac_energy_use()
            simulation_output.add_gains_and_losses()
            simulation_output.add_energy_balance_variables()
            for i in [
                "Site Outdoor Air Drybulb Temperature",
                "Site Outdoor Air Relative Humidity",
                # "Water Use Equipment Heating Rate",
                # "Water Use Equipment Hot Water Volume",
                "Zone Air Relative Humidity",
                "Zone Mean Air Temperature",
                "Zone Mean Radiant Temperature",
                "Zone Mechanical Ventilation Air Changes per Hour",
                "Zone Mechanical Ventilation Current Density Volume Flow Rate",
                "Zone Mechanical Ventilation Current Density Volume",
                "Zone Mechanical Ventilation Standard Density Volume Flow Rate",
                "Zone Mechanical Ventilation Standard Density Volume",
                "Zone Thermostat Cooling Setpoint Temperature",
                "Zone Thermostat Heating Setpoint Temperature",
            ]:
                simulation_output.add_output(output_name=i)

            run_period = RunPeriod()
            shadow_calc = ShadowCalculation(solar_distribution="MinimalShadowing")

            simulation_parameter = SimulationParameter(
                simulation_control=simulation_control,
                output=simulation_output,
                north_angle=0,
                run_period=run_period,
                shadow_calculation=shadow_calc,
                terrain_type=self.terrain.value,
                sizing_parameter=sizing_parameter,
            )

            sim_par_file = self._openstudio_directory / "simulation_parameters.json"
            with open(sim_par_file, "w", encoding="utf-8") as fp:
                json.dump(simulation_parameter.to_dict(), fp)

            osw = to_openstudio_osw(
                self._openstudio_directory.as_posix(),
                self._hbjson_file.as_posix(),
                sim_par_file.as_posix(),
                additional_measures=None,
                epw_file=self.epw_file.as_posix(),
            )
            _, idf = run_osw(osw, silent=True)

            # TODO - check in here that all gains are being represented in teh IDF< and add them if not

            _, _, _, _, _ = run_idf(
                idf_file_path=idf,
                epw_file_path=self.epw_file.as_posix(),
                expand_objects=True,
                silent=True,
            )

        else:
            logger.info(f"{self} - Existing results found")

        return None

    # endregion: SIMULATION_METHODS

    # region: RESULTS_LOADING

    def _sql_result(self) -> SQLiteResult:
        """Return the SQLite result of the simulation."""

        if not self._sql_file_exists():
            self._simulate()

        return SQLiteResult(self._sql_file.as_posix())

    def space_conditions(
        self, as_dataframe: bool = False
    ) -> dict[str, HourlyContinuousCollection] | pd.DataFrame:
        """Return the space conditions of the simulation.

        Args:
            dataframe (bool, optional):
                Whether to return the results as a DataFrame. Defaults to False.

        Returns:
            dict[str, HourlyContinuousCollection] | pd.DataFrame:
                The space conditions of the simulation.
        """
        sql_obj = self._sql_result()

        variables = {
            "Space Temperature (C)": "Zone Mean Air Temperature",
            "Space Mean Radiant Temperature (C)": "Zone Mean Radiant Temperature",
            "Space Relative Humidity (%)": "Zone Air Relative Humidity",
            "Space Heating Setpoint Temperature (C)": "Zone Thermostat Heating Setpoint Temperature",
            "Space Cooling Setpoint Temperature (C)": "Zone Thermostat Cooling Setpoint Temperature",
        }
        d = {}
        for k, v in variables.items():
            collections = sql_obj.data_collections_by_output_name(v)
            d[k] = aggregate_collection(collections=collections, agg="mean")

        if as_dataframe:
            df = pd.concat(
                [collection_to_series(v) for k, v in d.items()], axis=1, keys=d.keys()
            )
            return df

        return d

    def ventilation_flowrate(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Helper method to provide the ventilation flowrate for the typology.

        Args:
            as_series (bool, optional):
                Return the results as a pandas Series. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                The ventilation flowrate in L/s-m2.
        """

        sql_obj = self._sql_result()

        # get total volume of air being delivered to single building model
        supply_air_flowrate = aggregate_collection(
            collections=sql_obj.data_collections_by_output_name(
                "Zone Mechanical Ventilation Current Density Volume Flow Rate"
            ),
            agg="sum",
        ).to_unit("L/s")
        supply_air_flowrate_series = collection_to_series(supply_air_flowrate)

        # normalise by area
        supply_air_flowrate_series /= self.typical_gfa
        supply_air_flowrate_series.rename(
            "Ventilation Flowrate Intensity (L/s-m2)", inplace=True
        )

        if not as_series:
            return collection_from_series(supply_air_flowrate_series)

        return supply_air_flowrate_series

        # convert m3/s to l/s
        supply_air_ls = supply_air_flowrate * 1000

        # rename
        supply_air_ls.name = "Volume Flow Rate (l/s)"

        return supply_air_ls

    def external_conditions(
        self, as_dataframe: bool = False
    ) -> dict[str, HourlyContinuousCollection] | pd.DataFrame:
        """Return the external conditions of the simulation.

        Args:
            dataframe (bool, optional):
                Whether to return the results as a DataFrame. Defaults to False.

        Returns:
            dict[str, HourlyContinuousCollection] | pd.DataFrame:
                The external conditions of the simulation.
        """
        d = {
            "External Dry Bulb Temperature (C)": self.epw.dry_bulb_temperature,
            "External Relative Humidity (%)": self.epw.relative_humidity,
        }

        if as_dataframe:
            df = pd.concat(
                [collection_to_series(v) for k, v in d.items()], axis=1, keys=d.keys()
            )
            return df

        return d

    def eui_result(self, as_series: bool = False) -> dict[str, float] | pd.Series:
        """Return the EUI results of the simulation, in units of kWh/m2/year.

        Args:
            as_series (bool, optional):
                Whether to return the results as a pandas Series. Defaults to False.

        Returns:
            dict[str, float] | pd.Series:
                The EUI results of the simulation.
        """

        if not self._sql_file_exists():
            self._simulate()

        results = eui_from_sql(self._sql_file.as_posix())["end_uses"]

        # determine if results are all included, and include 0's if not
        keys = [
            "Cooling",
            "Heating",
            "Interior Lighting",
            "Electric Equipment",
            "Gas Equipment",
            "Water Systems",
        ]
        for k in keys:
            if k not in results:
                results[k] = 0

        if as_series:
            return pd.Series(results, name="EUI (kWh/m2/year)")

        return results

    def load_balance_result(
        self, as_dataframe: bool = False, normalised=True
    ) -> dict[str, HourlyContinuousCollection] | pd.DataFrame:
        """Return the load balance results of the simulation.

        Args:
            as_dataframe (bool, optional):
                Whether to return the results as a pandas DataFrame. Defaults to False.
            normalised (bool, optional):
                Whether to return the results in kWh/m2. Defaults to True.

        Returns:
            dict[str, float]:
                The load balance results, in the form of a dictionary, OR as a
                DataFrame if requested.
        """

        if not self._sql_file_exists():
            self._simulate()

        # obtain room energy results
        (
            cooling,
            heating,
            lighting,
            electric_equip,
            gas_equip,
            process,
            hot_water,
            fan_electric,
            pump_electric,
            people_gain,
            solar_gain,
            infiltration_load,
            mech_vent_load,
            nat_vent_load,
        ) = room_energy_result(self._sql_file.as_posix())

        # obtain face energy results
        face_indoor_temp, face_outdoor_temp, face_energy_flow = face_result(
            self._sql_file.as_posix()
        )

        # calculate load balance
        balance, balance_stor, norm_bal, norm_bal_stor = load_balance(
            [self._model()],
            cooling,
            heating,
            lighting,
            electric_equip,
            gas_equip,
            process,
            hot_water,
            people_gain,
            solar_gain,
            infiltration_load,
            mech_vent_load,
            nat_vent_load,
            face_energy_flow,
        )

        # create dict for load balance objects for easier referencing
        d = {}
        for collection in norm_bal_stor:
            d[collection.header.metadata["type"]] = collection
        for variable in [
            "Heating",
            "Solar",
            "Service Hot Water",
            "Gas Equipment",
            "Electric Equipment",
            "Lighting",
            "People",
            "Infiltration",
            "Mechanical Ventilation",
            "Opaque Conduction",
            "Window Conduction",
            "Cooling",
            "Storage",
        ]:
            if variable not in d:
                logger.info(
                    f"{variable} not found in load balance results for {self.building_type}."
                )
                d[variable] = list(d.values())[0].get_aligned_collection(0)

        if not normalised:
            for k, v in d.items():
                d[k] = v.aggregate_by_area(area=self.typical_gfa, area_unit="m2")

        if as_dataframe:
            df = pd.concat(
                [collection_to_series(v) for k, v in d.items()], axis=1, keys=d.keys()
            )
            return df

        return d

    # endregion: RESULTS_LOADING

    # region: RESULTS_PROCESSING

    def cooling_energy_demand(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Return the cooling energy demand of the simulation.

        Args:
            as_series (bool, optional):
                Whether to return the results as a pandas Series. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                The cooling energy demand of the building typology.
        """

        pth = self._simulation_directory / f"data_{inspect.stack()[0][3]}.csv"
        if pth.exists():
            cooling_demand = collection_from_series(
                pd.read_csv(pth, index_col=0, parse_dates=True, header=0).squeeze()
            )
        else:
            eui = self.eui_result(as_series=False)
            cooling_load_balance = -self.load_balance_result(
                as_dataframe=False, normalised=True
            )["Cooling"]
            cooling_demand = (cooling_load_balance / cooling_load_balance.total) * eui[
                "Cooling"
            ]
            collection_to_series(cooling_demand).to_csv(pth)

        if as_series:
            s = collection_to_series(cooling_demand)
            s.name = s.name.replace("Energy", "Cooling Energy")
            return s

        return cooling_demand

    def cooling_energy_consumption(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Get the cooling energy consumption of the simulation, including
        equipment efficiency.

        Args:
            as_series (bool, optional):
                Whether to return the results as a pandas Series. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                The cooling energy consumption of the building typology.
        """
        return self.cooling_energy_demand(as_series=as_series) / self.cooling_eer

    def heating_energy_demand(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Return the cooling energy demand of the simulation.

        Args:
            as_series (bool, optional):
                Whether to return the results as a pandas Series. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                The heating energy demand of the building typology.
        """

        pth = self._simulation_directory / f"data_{inspect.stack()[0][3]}.csv"
        if pth.exists():
            heating_demand = collection_from_series(
                pd.read_csv(pth, index_col=0, parse_dates=True, header=0).squeeze()
            )
        else:
            eui = self.eui_result(as_series=False)
            heating_load_balance = self.load_balance_result(
                as_dataframe=False, normalised=True
            )["Heating"]
            heating_demand = (heating_load_balance / heating_load_balance.total) * eui[
                "Heating"
            ]
            collection_to_series(heating_demand).to_csv(pth)

        if as_series:
            s = collection_to_series(heating_demand)
            s.name = s.name.replace("Energy", "Heating Energy")
            return s

        return heating_demand

    def heating_energy_consumption(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Get the heating energy consumption of the simulation, including
        equipment efficiency.

        Args:
            as_series (bool, optional):
                Whether to return the results as a pandas Series. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                The heating energy consumption of the building typology.
        """
        return self.heating_energy_demand(as_series=as_series) / self.heating_cop

    def lighting_energy_demand(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Return the lighting energy demand of the simulation.

        Args:
            as_series (bool, optional):
                Whether to return the results as a pandas Series. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                The lighting energy demand of the building typology.
        """

        pth = self._simulation_directory / f"data_{inspect.stack()[0][3]}.csv"
        if pth.exists():
            lighting_demand = collection_from_series(
                pd.read_csv(pth, index_col=0, parse_dates=True, header=0).squeeze()
            )
        else:
            eui = self.eui_result(as_series=False)
            lighting_load_balance = self.load_balance_result(
                as_dataframe=False, normalised=True
            )["Lighting"]
            try:
                lighting_demand = (
                    lighting_load_balance / lighting_load_balance.total
                ) * eui["Interior Lighting"]
            except ZeroDivisionError:
                lighting_demand = lighting_load_balance
            collection_to_series(lighting_demand).to_csv(pth)

        if as_series:
            s = collection_to_series(lighting_demand)
            s.name = s.name.replace("Energy", "Lighting Energy")
            return s

        return lighting_demand

    def lighting_energy_consumption(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Get the lighting energy consumption of the simulation, including
        equipment efficiency.

        Args:
            as_series (bool, optional):
                Whether to return the results as a pandas Series. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                The lighting energy consumption of the building typology.
        """
        return self.lighting_energy_demand(as_series=as_series)

    def electric_equipment_energy_demand(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Return the electric_equipment energy demand of the simulation.

        Args:
            as_series (bool, optional):
                Whether to return the results as a pandas Series. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                The electric_equipment energy demand of the building typology.
        """

        pth = self._simulation_directory / f"data_{inspect.stack()[0][3]}.csv"
        if pth.exists():
            electric_equipment_demand = collection_from_series(
                pd.read_csv(pth, index_col=0, parse_dates=True, header=0).squeeze()
            )
        else:
            eui = self.eui_result(as_series=False)
            electric_equipment_load_balance = self.load_balance_result(
                as_dataframe=False, normalised=True
            )["Electric Equipment"]
            try:
                electric_equipment_demand = (
                    electric_equipment_load_balance
                    / electric_equipment_load_balance.total
                ) * eui["Electric Equipment"]
            except ZeroDivisionError:
                electric_equipment_demand = electric_equipment_load_balance
            collection_to_series(electric_equipment_demand).to_csv(pth)

        if as_series:
            s = collection_to_series(electric_equipment_demand)
            s.name = s.name.replace("Energy", "Electric Equipment Energy")
            return s

        return electric_equipment_demand

    def electric_equipment_energy_consumption(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Get the electric_equipment energy consumption of the simulation, including
        equipment efficiency.

        Args:
            as_series (bool, optional):
                Whether to return the results as a pandas Series. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                The electric_equipment energy consumption of the building typology.
        """
        return self.electric_equipment_energy_demand(as_series=as_series)

    def hot_water_energy_demand(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Return the electric_equipment energy demand of the simulation.

        Args:
            as_series (bool, optional):
                Whether to return the results as a pandas Series. Defaults to False.
            normalised (bool, optional):
                Whether to return the results in kWh/m2. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                The electric_equipment energy demand of the building typology.
        """

        pth = self._simulation_directory / f"data_{inspect.stack()[0][3]}.csv"
        if pth.exists():
            hot_water_demand = collection_from_series(
                pd.read_csv(pth, index_col=0, parse_dates=True, header=0).squeeze()
            )
        else:
            eui = self.eui_result(as_series=False)
            hot_water_load_balance = self.load_balance_result(
                as_dataframe=False, normalised=True
            )["Service Hot Water"]
            try:
                hot_water_demand = (
                    hot_water_load_balance / hot_water_load_balance.total
                ) * eui["Water Systems"]
            except ZeroDivisionError:
                hot_water_demand = hot_water_load_balance
            collection_to_series(hot_water_demand).to_csv(pth)

        if as_series:
            s = collection_to_series(hot_water_demand)
            s.name = s.name.replace("Energy", "Hot Water Energy")
            return s

        return hot_water_demand

    def hot_water_consumption(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Get the hot water energy consumption of the simulation, including
        equipment efficiency.

        Args:
            as_series (bool, optional):
                Whether to return the results as a pandas Series. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                The hot water energy consumption of the building typology.
        """
        return self.hot_water_energy_demand(as_series=as_series) / self.heating_cop

    def pump_energy_consumption(
        self,
        as_series: bool = False,
    ) -> pd.Series:
        """Estimate the energy consumption of pumps in the building, based on
        simulation results. This includes efficiencies of the pumps.

        Note:
            This method is a bit of a hack, and assumes that the pump power is
            infinitely variable based on Q = m.Cp.dT. This is not true, and
            should be updated in the future.

            It also assumes the dT for both hot water and chilled water is constant.
            It also assumes the Cp for water is constant.

        References:
            These calcs come from a range of sources, most notably these files:
            - https://burohappold.sharepoint.com/:x:/r/sites/060941/Shared%20Documents/Sustainability%20and%20Microclimate/energy/ss/RC_ExcelEnergyLoadCalcs/Copy%20of%20160525%20RC%20DEWA%20Office%20Energy%20REVISION%20Sunpower%20345.xlsx?d=w18617de981194b4ba3f2ab8d7449a218&csf=1&web=1&e=tpJtfq
            - https://burohappold.sharepoint.com/sites/060941/Shared%20Documents/Sustainability%20and%20Microclimate/energy/ss/RC_ExcelEnergyLoadCalcs/Copy%20of%20160525%20RC%20DEWA%20Office%20Energy%20REVISION%20Sunpower%20345.xlsx?web=1

        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional):
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings

        Returns:
            pd.Series: Hourly pump energy demand
        """

        pth = self._simulation_directory / f"data_{inspect.stack()[0][3]}.csv"
        if pth.exists():
            pump_energy_consumption = pd.read_csv(
                pth, index_col=0, parse_dates=True, header=0
            ).squeeze()
        else:
            cooling_demand = self.cooling_energy_demand(as_series=True)
            heating_demand = self.heating_energy_demand(as_series=True)
            hot_water_demand = self.hot_water_energy_demand(as_series=True)

            # TODO - make this method dynamic, using timestep Cp and delta Ts ... maybe

            # HOT WATER FLOW #
            # NOTE: This doesn't account for pump minimum flow rates
            hot_water_delta_t = pd.Series(
                [25] * len(cooling_demand.index),
                index=cooling_demand.index,
                name="Hot Water Delta T (K)",
            )  # K
            hot_water_cp = pd.Series(
                [4.18] * len(cooling_demand.index),
                index=cooling_demand.index,
                name="Hot Water Specific Heat Capcity (J/g/K)",
            )  # J/g/K

            hot_water_flowrate = (
                (heating_demand + hot_water_demand) / (hot_water_delta_t * hot_water_cp)
            ).rename(
                "Hot Water Flow (l/s)"
            )  # l/s

            # CHILLED WATER FLOW #
            chilled_water_delta_t = pd.Series(
                [6] * len(cooling_demand.index),
                index=cooling_demand.index,
                name="Chilled Water Delta T (K)",
            )  # K
            chilled_water_cp = pd.Series(
                [4.18] * len(cooling_demand.index),
                index=cooling_demand.index,
                name="Chilled Water Specific Heat Capcity (J/g/K)",
            )  # J/g/K

            chilled_water_flowrate = (
                cooling_demand / (chilled_water_delta_t * chilled_water_cp)
            ).rename(
                "Chilled Water Flow (l/s)"
            )  # l/s

            # calculate pump energy demand, and convert back to kW
            hot_water_pumping = hot_water_flowrate * self.pump_power  # W/m2
            chilled_water_pumping = chilled_water_flowrate * self.pump_power  # W/m2
            pump_energy_consumption = (
                (hot_water_pumping + chilled_water_pumping) / 1000
            ).rename("Energy Intensity (kWh/m2)")
            pump_energy_consumption.to_csv(pth)

        if as_series:
            pump_energy_consumption.name = pump_energy_consumption.name.replace(
                "Energy", "Pump Energy"
            )
            return pump_energy_consumption

        return collection_from_series(pump_energy_consumption)

    def fan_energy_consumption(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Estimate the energy consumption of fans in the building, based on
        simulation results, including efficiency of the fans.

        Note:
            This method is a bit of a hack, and assumes that the fan power is
            infinitely variable based on Q = m.Cp.dT. This is not true, and
            should be updated in the future.
            It also assumes the dT for air is constant.
            It also assumes the Cp for air is constant.

        References:
            These calcs come from a range of sources, most notably these files:
            - https://burohappold.sharepoint.com/:x:/r/sites/060941/Shared%20Documents/Sustainability%20and%20Microclimate/energy/ss/RC_ExcelEnergyLoadCalcs/Copy%20of%20160525%20RC%20DEWA%20Office%20Energy%20REVISION%20Sunpower%20345.xlsx?d=w18617de981194b4ba3f2ab8d7449a218&csf=1&web=1&e=tpJtfq
            - https://burohappold.sharepoint.com/sites/060941/Shared%20Documents/Sustainability%20and%20Microclimate/energy/ss/RC_ExcelEnergyLoadCalcs/Copy%20of%20160525%20RC%20DEWA%20Office%20Energy%20REVISION%20Sunpower%20345.xlsx?web=1

        Args:
            as_series: bool, optional
                Whether to return the results as a pandas Series. Defaults to False.

        Returns:
            HourlyContinuousCollection | pd.Series:
                Hourly fan energy consumption
        """

        pth = self._simulation_directory / f"data_{inspect.stack()[0][3]}.csv"
        if pth.exists():
            fan_energy = pd.read_csv(
                pth, index_col=0, parse_dates=True, header=0
            ).squeeze()
        else:

            # get volume of air being supplied for a single building
            supply_air_ls = self.ventilation_flowrate(as_series=True) * self.typical_gfa

            # get the energy demand in kWh/m2 (sfp is in w/l/s, so the /1000 accounts for that)
            fan_energy = (
                supply_air_ls * self.fan_power / 1000
            ) / self.typical_gfa  # kWh/m2

            # THIS PART IS ADDED TO ACCOUNT FOR ENERGY FOR FANS TO PUMP COOLED/HEATED AIR INTO A SPACE, NOT JUST VENTILATION
            # get cooling/heating flow recirculation energy
            load_balance_kwh = self.load_balance_result(
                as_dataframe=True, normalised=False
            )
            # load_balance_w = self.load_balance(normalised=False, single_building=True)
            # max htg/clg load - to get peak fan energy
            max_load_w = (
                max(
                    load_balance_kwh["Cooling"].max(),
                    load_balance_kwh["Heating"].max(),
                )
                / 1000
            )
            delta_t = 12
            flow_ls = max_load_w / (1.2 * 1.02 * delta_t)
            flow_energy_kwh = (self.fan_power * flow_ls) / 1000
            # distribute over the htg/clg loads, by interpolating between highest and lowest total htg/clg, and making the peak wattage the flow energy the peak energy
            xx = load_balance_kwh["Cooling"] + load_balance_kwh["Heating"]
            xx = np.interp(xx, [xx.min(), xx.max()], [0, flow_energy_kwh])
            # add to the fan_energy_consumption
            fan_energy += xx
            fan_energy = fan_energy.rename("Energy Intensity (kWh/m2)")
            fan_energy.to_csv(pth)

        if as_series:
            fan_energy.name = fan_energy.name.replace("Energy", "Fan Energy")
            return fan_energy

        return collection_from_series(fan_energy)

    def lift_energy_consumption(
        self, as_series: bool = False
    ) -> HourlyContinuousCollection | pd.Series:
        """Estimate the annual energy consumption of a lift system in kWh/m2.

        References:
        - BRE. “NABERS UK: Guide to Design for Performance,” April 2021.

        Returns:
            pd.Series:
                The estimated hourly energy consumption of the lift system.
        """

        if self.average_num_floors < 2:
            # no lifts in buildings < 2 floors
            annual_energy_kwhm2 = 0
        else:
            alpha = (0.26 + 0.37) / 2
            annual_energy_kwhm2 = (
                ((528 * self.average_num_floors) + (5.5 * self.typical_gfa))
                * (1 - alpha)
            ) / self.typical_gfa

        # get the number of occupants to distribute the energy over
        occupants = self._population(per_building=True)

        # clip occupants so that the lowest level is at least 25% of the maximum level (to approximate standby power)
        occupants.clip(lower=(occupants.max() - occupants.min()) * 0.25, inplace=True)

        # distribute annual energy across year weighted by occupants
        lift_energy = ((occupants / occupants.sum()) * annual_energy_kwhm2).rename(
            "Energy Intensity (kWh/m2)"
        )

        if as_series:
            lift_energy.name = lift_energy.name.replace("Energy", "Lift Energy")
            return lift_energy

        return collection_from_series(lift_energy)

    def energy_consumption(
        self, normalised: bool = True, as_dataframe: bool = False
    ) -> pd.DataFrame:
        """Get the energy consumption of the typology, including effects from equipment performance.

        Args:
            normalised (bool, optional):
                Whether to return the results in kWh/m2. Defaults to True.

        Returns:
            pd.DataFrame:
                The annual hourly energy consumption of the typology.
        """

        # get each major metric
        cooling = self.cooling_energy_consumption(as_series=False)
        heating = self.heating_energy_consumption(as_series=False)
        lighting = self.lighting_energy_consumption(as_series=False)
        electric_equipment = self.electric_equipment_energy_consumption(as_series=False)
        hot_water = self.hot_water_consumption(as_series=False)
        lifts = self.lift_energy_consumption(as_series=False)
        pumps = self.pump_energy_consumption(as_series=False)
        fans = self.fan_energy_consumption(as_series=False)

        if not normalised:
            cooling = cooling.aggregate_by_area(area=self.total_area, area_unit="m2")
            heating = heating.aggregate_by_area(area=self.total_area, area_unit="m2")
            lighting = lighting.aggregate_by_area(area=self.total_area, area_unit="m2")
            electric_equipment = electric_equipment.aggregate_by_area(
                area=self.total_area, area_unit="m2"
            )
            hot_water = hot_water.aggregate_by_area(
                area=self.total_area, area_unit="m2"
            )
            lifts = lifts.aggregate_by_area(area=self.total_area, area_unit="m2")
            pumps = pumps.aggregate_by_area(area=self.total_area, area_unit="m2")
            fans = fans.aggregate_by_area(area=self.total_area, area_unit="m2")

        if as_dataframe:
            serieses = [
                collection_to_series(i)
                for i in [
                    cooling,
                    heating,
                    lighting,
                    electric_equipment,
                    hot_water,
                    lifts,
                    pumps,
                    fans,
                ]
            ]
            unit = get_unit(serieses[0].name)
            keys = [
                f"{i} ({unit})"
                for i in [
                    "Cooling",
                    "Heating",
                    "Lighting",
                    "Electric Equipment",
                    "Hot Water",
                    "Lifts",
                    "Pumps",
                    "Fans",
                ]
            ]
            df = pd.concat(
                serieses,
                axis=1,
                keys=keys,
            )
            return df

        return {
            "Cooling": cooling,
            "Heating": heating,
            "Lighting": lighting,
            "Electric Equipment": electric_equipment,
            "Hot Water": hot_water,
            "Lifts": lifts,
            "Pumps": pumps,
            "Fans": fans,
        }

    def run_all(self) -> None:
        """Run a combination of simulations and post-processing in one function."""
        self._simulate()
        self.energy_consumption()
        return None

    # endregion: RESULTS_PROCESSING

    # region: PLOTTING

    def plot_annual_monthly(
        self,
        ax: plt.Axes = None,
        rule: str = "MS",
        label: bool = True,
        legend: bool = True,
    ) -> plt.Axes:
        """Plot the monthly energy consumption of the typology.

        Args:
            ax (plt.Axes):
                The axes to plot on. Default is None.
            rule (str):
                The resampling rule to use. Default is "MS".
            label (bool):
                Label the axes. Default is True.
            legend (bool):
                Show the legend. Default is True.
        """

        # get the typology hourly energy consumption, hourly, in kWh
        df = self.energy_consumption(normalised=False, as_dataframe=True)

        # sort in order to most to least energy consumption
        df = df[df.sum(axis=0).sort_values(ascending=False).index]

        if ax is None:
            ax = plt.gca()

        ax = stacked_bar(df=df, ax=ax, rule=rule, label=label, legend=legend)

        _ = ax.set_title(f"{self.identifier} - Energy Consumption")

        return ax

    def plot_pie(
        self,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        ax: plt.Axes = None,
        label: bool = True,
        legend: bool = True,
        normalised: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """Plot a pie chart of the annual energy consumption of the typology.

        Args:
            ax (plt.Axes):
                The axes to plot on. Default is None.
            label (bool):
                Label the axes. Default is True.
            legend (bool):
                Show the legend. Default is True.

        Returns:
            plt.Axes:
                The axes object.
        """

        series = (
            self.energy_consumption(normalised=normalised, as_dataframe=True)
            .loc[pd.to_datetime(analysis_period.datetimes)]
            .sum(axis=0)
        )  # kWh/time-period
        # sort by largest first
        series = series.sort_values(ascending=False)
        # get unit
        unit = get_unit(series.index[0])

        if ax is None:
            ax = plt.gca()

        ax = pie(series=series, ax=ax, legend=legend, label=label, **kwargs)

        ti = f"{self.identifier} - Energy Consumption\n{describe_analysis_period(analysis_period)}\n{series.sum():,.0f}{unit.replace('m2', 'm$^{2}$')}"
        if not normalised:
            ti += f" (over {self.total_area:,.0f}m$^{2}$)"
        _ = ax.set_title(ti)

        return ax

    def plot_diurnal(
        self,
        ax: plt.Axes = None,
        legend: bool = True,
        logy: bool = False,
        normalised: bool = False,
    ) -> plt.Axes:
        """Plot a monthly diurnal profile for energy consumption of the typology.

        Args:
            epw (EPW):
                The EPW file to use for the simulation.
            ax (plt.Axes):
                The axes to plot on. Default is None.
            legend (bool):
                Show the legend. Default is True.
            logy (bool):
                Use a log scale. Default is False.

        Returns:
            plt.Axes:
                The axes object.
        """

        # logger.info(f"{self} - Plotting diurnal energy consumption")

        df = self.energy_consumption(normalised=normalised, as_dataframe=True)

        if ax is None:
            ax = plt.gca()

        ax = diurnal(df, ax=ax, legend=legend, logy=logy)

        _ = ax.set_title(f"{self.identifier} - Energy Consumption")

        return ax

    # def plot_duration_curve(
    #     self,
    #     ax: plt.Axes = None,
    #     remove_zero: bool = True,
    #     legend: bool = True,
    #     normalised: bool = False,
    #     single_building: bool = False,
    #     **kwargs,
    # ) -> plt.Axes:
    #     """Plot a duration curve for energy consumption of the typology.

    #     Args:
    #         ax (plt.Axes):
    #             The axes to plot on. Default is None.
    #         remove_zero (bool):
    #             Remove zero values. Default is True.
    #         legend (bool):
    #             Show the legend. Default is True.
    #         **kwargs:
    #             Additional keyword arguments to pass to the plt.hist function.

    #     Returns:
    #         plt.Axes:
    #             The axes object.
    #     """

    #     # logger.info(f"{self} - Plotting duration curve")

    #     df = self.energy_consumption(
    #         normalised=normalised, single_building=single_building
    #     )

    #     if ax is None:
    #         ax = plt.gca()

    #     ax = duration_curve(df, ax=ax, legend=legend, remove_zero=remove_zero, **kwargs)

    #     _ = ax.set_title(f"{self.identifier} - Energy Consumption - Duration Curve")

    #     return ax

    # endregion: PLOTTING

    # def run_everything(self):
    #     """Run all calculations for the typology."""

    #     _ = self._all_data()

    #     # create plots
    #     for legend in [True, False]:

    #         # DIURNAL #
    #         fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
    #         self.plot_diurnal(directory=directory, ax=ax, legend=legend)
    #         plt.tight_layout()
    #         plt.savefig(Path(directory) / self.identifier / f"fig_diurnal{'' if legend else '_nolegend'}.png", bbox_inches="tight", transparent=True)
    #         plt.close(fig)

    #         # PIE #
    #         fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SQUARE)
    #         self.plot_pie(directory=directory, ax=ax, legend=legend)
    #         plt.tight_layout()
    #         plt.savefig(Path(directory) / self.identifier / f"fig_pie{'' if legend else '_nolegend'}.png", bbox_inches="tight", transparent=True)
    #         plt.close(fig)

    #         # ANNUAL MONTHLY #
    #         fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
    #         self.plot_annual_monthly(directory=directory, ax=ax, legend=legend)
    #         plt.tight_layout()
    #         plt.savefig(Path(directory) / self.identifier / f"fig_annual_monthly{'' if legend else '_nolegend'}.png", bbox_inches="tight", transparent=True)
    #         plt.close(fig)
