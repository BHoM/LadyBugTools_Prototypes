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
from honeybee_energy.result.loadbalance import LoadBalance, SQLiteResult
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval
from honeybee_energy.shw import SHWSystem
from honeybee_energy.simulation.parameter import (RunPeriod, ShadowCalculation,
                                                  SimulationControl,
                                                  SimulationOutput,
                                                  SimulationParameter,
                                                  SizingParameter)
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug_geometry.geometry2d import Point2D, Polygon2D, Vector2D
from ladybug_geometry.geometry3d import (Face3D, LineSegment3D, Point3D,
                                         Vector3D)
from pydantic import BaseModel, Field, root_validator
from sklearn.linear_model import LinearRegression

from .config import DATA_PATH, FIGSIZE_RECTANGLE, FIGSIZE_SQUARE, INDEX, logger
from .enums import (EPW, BuildingType, ConstructionType, EconomizerType,
                    LiftEnergyEfficiency, LiftUsageIntensity, TerrainType,
                    Vintage, default_construction_type,
                    default_constructionset, default_context_shade_distance,
                    default_cooling_eer, default_daylight_dimming,
                    default_demand_controlled_ventilation,
                    default_economizer_type, default_fan_power,
                    default_floor_height, default_footprint_area,
                    default_glazing_ratio, default_heating_cop,
                    default_hr_effectiveness, default_number_of_floors,
                    default_program_type, default_pump_power,
                    default_skylight_ratio)
from .lifts import (al_sharif_1996, approximate_lift_energy_demand,
                    simple_estimate, simple_estimate_from_storeys)
from .plot import diurnal, duration_curve, pie, stacked_bar
from .util import (annual_eui, collection_to_series, construction_sri,
                   describe_analysis_period, estimate_sri_properties,
                   face_orientation, face_result, get_unit, load_balance,
                   random_id, room_comfort_result, room_energy_result)

# pylint: enable=E0401
# endregion: IMPORTS

ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ROOT_DIRECTORY = Path(hb_folders.default_simulation_folder)
RELOAD = True


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
    # endregion: ATTRIBUTES

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.identifier})"

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

    def basic_summary(self) -> str:
        """Return a basic summary of the typology."""
        return "\n".join(
            [
                f"{self.identifier} - {self.building_type} - {self.vintage}",
                f"Total GFA: {self.total_area} m2",
                f"Typical building GFA: {self.typical_building_gfa} m2",
                f"Number of buildings: {self._number_of_buildings()}",
                f"Typical storeys: {self.average_num_floors}",
            ]
        )

    # pylint: disable=no-self-argument
    @root_validator(pre=True)
    def validate_atts(cls, values):
        """Validate the attributes, ensuring that certain values work with each other."""

        identifier = values.get("identifier")
        valid_string(identifier)

        epw_file = Path(values.get("epw_file"))
        assert epw_file.exists(), f"{epw_file.absolute()} file does not exist."

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

    # pylint: enable=no-self-argument

    @classmethod
    def random(cls, epw_file: Path = None, seed: int = None) -> "Typology":
        """Generate a random typology."""

        np.random.seed(seed)

        if epw_file is None:
            # reference the test EPW here ... this is bad practice, but meh
            epw_file = Path(__file__).absolute().parent / "test" / "test.epw"

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

        Returns:
            Typology: The typology object.
        """

        epw = EPW(epw_file)

        if identifier is None:
            identifier = building_type.value
            logger.info(
                f"{__class__.__name__} - Using default identifier: {identifier}"
            )

        if isinstance(building_type, str):
            building_type = BuildingType(building_type)

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

        d = {
            "identifier": building_type.value if identifier is None else identifier,
            "total_area": total_area,
            "epw_file": Path(epw_file),
            "building_type": building_type,
            "vintage": vintage,
            "rotation": rotation,
            "terrain": terrain,
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

        # create the default typology
        default_typology = cls.from_building_type(
            building_type=d["building_type"],
            total_area=d["total_area"],
            epw_file=d["epw_file"],
            identifier=d["identifier"],
            vintage=d["vintage"],
            rotation=d["rotation"],
            terrain=d["terrain"],
        )

        # overwrite any nulls with default values
        for k, v in d.items():
            if pd.isnull(v):
                val = getattr(default_typology, k)
                logger.info(f"Using default value {val} for {k} in {d['identifier']}.")
                d[k] = val

        return cls.parse_obj(d)

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
        """The Solar Heat Gain Coefficients of the windows in each orientation."""
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
    def typical_building_gfa(self) -> float:
        """Get the typical gross floor area for a building of this typology."""
        return self.total_area / self._number_of_buildings()

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

    def _ideal_air(self) -> IdealAirSystem:
        """Return the ideal air system associated with the system."""

        return IdealAirSystem(
            identifier=self.identifier,
            economizer_type=self.economizer_type.value,
            demand_controlled_ventilation=self.demand_controlled_ventilation,
            sensible_heat_recovery=self.sensible_heat_recovery_effectiveness,
            latent_heat_recovery=self.latent_heat_recovery_effectiveness,
        )

    def _building_height(self) -> float:
        """Get the typical height for an individual building."""

        return self.average_num_floors * self.average_floor_height

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
            context_height = self._building_height() * 0.75
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

    def _number_of_buildings(self) -> float:
        """Return the number of buildings this Typology represents"""
        return self.total_area / (self.average_footprint_area * self.average_num_floors)

    def _occupancy_schedule(self) -> pd.Series:
        """Get the occupancy schedule for the building type."""
        program = self._program_type()

        if program.people is None:
            values = np.zeros(8760)
        else:
            values = program.people.occupancy_schedule.data_collection().values

        return pd.Series(
            values,
            index=pd.to_datetime(AnalysisPeriod().datetimes),
            name="Occupancy",
        )

    def _occupants(self, per_building: bool = False) -> pd.Series:
        """Get the number of occupants in the building type."""
        return (
            self._occupancy_schedule()
            * self.occupant_density
            * (self.typical_building_gfa if per_building else self.total_area)
        )

    def _target_dir(self, directory: Path = ROOT_DIRECTORY) -> Path:
        """Lightweight helper method to get the target directory for the typology simulation results."""
        return Path(directory) / self.identifier

    def _openstudio_dir(self, directory: Path = ROOT_DIRECTORY) -> Path:
        """Lightweight helper method to get the OpenStudio directory for the typology simulation results."""
        return self._target_dir(directory) / "openstudio"

    def _sql_file(self, directory: Path = ROOT_DIRECTORY) -> Path:
        """Lightweight helper method to get the SQL file for the typology simulation results."""
        return self._openstudio_dir(directory) / "run" / "eplusout.sql"

    def _osw_file(self, directory: Path = ROOT_DIRECTORY) -> Path:
        """Lightweight helper method to get the OSW file for the typology simulation results."""
        return self._openstudio_dir(directory) / "workflow.osw"

    def _config_file(self, directory: Path = ROOT_DIRECTORY) -> Path:
        """Lightweight helper method to get the config file for the typology simulation results."""
        return self._target_dir(directory) / "mped_config.json"

    def _results_exist(self, directory: Path = ROOT_DIRECTORY) -> bool:
        """Check if the results for the typology already exist.

        Args:
            directory (Path):
                The directory to check for results in. Defaults to the ladybug simulation directory.

        Returns:
            bool:
                True if results exist, False otherwise.
        """
        return self._sql_file_exists(directory=Path(directory))

    def _sql_file_exists(self, directory: Path = ROOT_DIRECTORY) -> bool:
        """Check if the results for the typology already exist.

        Args:
            directory (Path):
                The directory to check for results in. Defaults to the ladybug simulation directory.

        Returns:
            bool:
                True if results exist, False otherwise.
        """

        directory = Path(directory)

        config_file = self._config_file(directory=directory)
        if not config_file.exists():
            return False
        if Typology.parse_file(config_file) != self:
            return False
        osw_file = self._osw_file(directory=directory)
        if not osw_file.exists():
            return False
        with open(osw_file, "r", encoding="utf-8") as fp:
            old_epw_file = Path(json.load(fp)["weather_file"])
        if old_epw_file.name != self.epw_file.name:
            return False
        if self._sql_file(directory=directory).exists():
            return True

        return False

    def _simulate(self, directory: Path = ROOT_DIRECTORY) -> pd.DataFrame:
        """Simulate the typology for a given EPW file and return the results in a DataFrame.

        Args:
            directory (Path):
                The directory to save the results in.

        Returns:
            pd.DataFrame:
                All results of the simulation.
        """

        # TODO - replace with purpose specific simulate method, rather than having all the functionality within this function
        # validate inputs
        if Path(directory).is_file():
            raise ValueError("Target directory is a file and must be a directory.")

        # create model
        logger.disabled = True
        model = self._model()
        logger.disabled = False

        # run simulation if it hasn't already been run
        if not self._sql_file_exists(directory):

            # run simulation
            logger.info(f"{self} - Simulating results")

            # remove old files in target directory just in case
            for f in self._target_dir(directory=directory).glob("*"):
                if f.is_file():
                    f.unlink()

            self._openstudio_dir(directory=directory).mkdir(parents=True, exist_ok=True)

            # write config file
            with open(
                self._config_file(directory=directory), "w", encoding="utf-8"
            ) as fp:
                fp.write(self.json(indent=4))

            # write model to target directory to reference in simulation
            model.to_hbjson(folder=self._openstudio_dir(directory=directory))

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

            sim_par_file = (
                self._openstudio_dir(directory=directory) / "simulation_parameters.json"
            )
            with open(sim_par_file, "w", encoding="utf-8") as fp:
                json.dump(simulation_parameter.to_dict(), fp)

            osw = to_openstudio_osw(
                self._openstudio_dir(directory=directory).as_posix(),
                (
                    self._openstudio_dir(directory=directory)
                    / f"{self.identifier}.hbjson"
                ).as_posix(),
                sim_par_file.as_posix(),
                additional_measures=None,
                epw_file=self.epw_file.as_posix(),
            )
            _, idf = run_osw(osw, silent=True)

            _, _, _, _, _ = run_idf(
                idf_file_path=idf,
                epw_file_path=self.epw_file.as_posix(),
                expand_objects=True,
                silent=True,
            )

        return None

    def load_balance(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
    ) -> pd.DataFrame:
        """Load the load balance results from the simulation.

        Args:
            directory (Path):
                The directory to save the results in.

        Returns:
            pd.DataFrame:
                The load balance results in a pandas DataFrame.

        Notes:
            The file saved to disk will always be stored in kWh/m2, but the
            DataFrame will be returned in kWh if normalised is False.
            Also, if normalised is false and single building is True, then the
            results will be multiplied by the number of buildings.
        """

        _sql = self._sql_file(directory=directory)
        if not _sql.exists():
            raise FileNotFoundError(
                "No simulation results found. Try running it first :)"
            )

        pth = (
            Path(directory)
            / self.identifier
            / f"data_{inspect.stack()[0][3]}_normalised.csv"
        )
        if all([RELOAD, pth.exists()]):
            # reload existing results that were previously generated
            logger.info(f"{self} - Reloading load balance data")
            df = pd.read_csv(pth, index_col=0, parse_dates=True, header=0)
        else:
            # process results if they don't already exist
            logger.info(f"{self} - Creating load balance data")
            (
                cooling,
                heating,
                lighting,
                electric_equip,
                gas_equip,
                process,
                hot_water,
                _,
                _,
                people_gain,
                solar_gain,
                infiltration_load,
                mech_vent_load,
                nat_vent_load,
            ) = room_energy_result(_sql)
            _, _, face_energy_flow = face_result(_sql)
            _, _, _, norm_bal_stor = load_balance(
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
            d = []
            for collection in norm_bal_stor:
                collection: HourlyContinuousCollection
                s = collection_to_series(collection)
                s.name = f"Load Balance {collection.header.metadata['type']} ({collection.header.unit})"
                d.append(s)
            df = pd.concat(d, axis=1)
            df.to_csv(pth)

        # convert to de-normalised if requested
        if not normalised:
            df *= self.typical_building_gfa
            df.columns = [i.replace("(kWh/m2)", "(kWh)") for i in df.columns]

        # multiply by number of buildings if requested
        if not single_building:
            if not normalised:
                df *= self._number_of_buildings()

        return df

    def sql_results(self, directory: Path = ROOT_DIRECTORY) -> SQLiteResult:
        """Return the SQLite results of the simulation.
        
        Args:
            directory (Path):
                The directory where results can be found.
        
        Returns:
            SQLiteResult:
                The SQLite results of the simulation.
        """
        _sql = self._sql_file(directory=directory)
        if not _sql.exists():
            raise FileNotFoundError("No simulation results found. Try running the simulation first :)")
        
        return SQLiteResult(_sql.as_posix())

    def space_conditions(self, directory: Path = ROOT_DIRECTORY) -> pd.DataFrame:
        """Get the room conditions of the simulated typology.

        Args:
            epw (EPW):
                The EPW file to use for the simulation.
            directory (Path):
                The directory to save the results in.

        Returns:
            pd.DataFrame:
                The room conditions of the typology.
        """

        pth = Path(directory) / self.identifier / f"data_{inspect.stack()[0][3]}.csv"
        if all([RELOAD, pth.exists()]):
            logger.info(f"{self} - Reloading space condition data")
            return pd.read_csv(pth, index_col=0, header=0, parse_dates=True)
        
        sql_obj = self.sql_results(directory=directory)
        _, air_temp, rad_temp, rel_humidity, _, _ = room_comfort_result(sql_obj.file_path)

        # get averages
        dbt = (
            pd.concat([collection_to_series(i) for i in air_temp], axis=1)
            .mean(axis=1)
            .rename("Dry Bulb Temperature (C)")
        )
        mrt = (
            pd.concat([collection_to_series(i) for i in rad_temp], axis=1)
            .mean(axis=1)
            .rename("Mean Radiant Temperature (C)")
        )
        rh = (
            pd.concat([collection_to_series(i) for i in rel_humidity], axis=1)
            .mean(axis=1)
            .rename("Relative Humidity (%)")
        )
        htg_setpt = collection_to_series(
            sql_obj.data_collections_by_output_name(
                "Zone Thermostat Heating Setpoint Temperature"
            )[0]
        ).rename("Heating Setpoint Temperature (C)")
        clg_setpt = collection_to_series(
            sql_obj.data_collections_by_output_name(
                "Zone Thermostat Cooling Setpoint Temperature"
            )[0]
        ).rename("Cooling Setpoint Temperature (C)")
        ach = collection_to_series(
            sql_obj.data_collections_by_output_name(
                "Zone Mechanical Ventilation Air Changes per Hour"
            )[0]
        ).rename("Air Changes per Hour")
        df = pd.concat([rh, dbt, mrt, htg_setpt, clg_setpt, ach], axis=1)
        logger.info(f"{self} - Creating space condition data")
        df.to_csv(pth)
        return df

    def external_conditions(self) -> pd.DataFrame:
        """Get the external conditions of the simulated typology.

        Returns:
            pd.DataFrame:
                The external conditions of the typology.
        """

        df = pd.concat(
            [
                collection_to_series(self.epw.dry_bulb_temperature),
                collection_to_series(self.epw.relative_humidity),
            ],
            axis=1,
        )

        return df

    def _cooling_energy_demand(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
    ) -> pd.Series:
        """Get the cooling energy demand of the typology.

        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional):
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings.

        Returns:
            pd.Series: A pandas Series of the hourly cooling energy demand (prior to applying system efficiency) in kWh.
        """

        pth = Path(directory) / self.identifier / f"data_{inspect.stack()[0][3]}.csv"
        if all([RELOAD, pth.exists()]):
            # reload existing calculation
            logger.info(f"{self} - Reloading cooling energy demand")
            cooling_hourly = pd.read_csv(
                pth, index_col=0, header=0, parse_dates=True
            ).squeeze()
        else:
            # run calculation process
            logger.info(f"{self} - Calculating cooling energy demand")

            # get load balance outputs - normalised, in kWh/m2
            load_balance_df = self.load_balance(
                directory=directory, normalised=True, single_building=True
            )

            # get EUI outputs, which are also always normalised
            eui_df = self.annual_eui(directory=directory)

            # calculate and scale cooling load to eui to ensure consistency
            try:
                cooling_annual_kwhm2 = eui_df["Cooling"]
                cooling_hourly = -load_balance_df["Load Balance Cooling (kWh/m2)"]
                cooling_hourly = ((
                    cooling_hourly / cooling_hourly.sum()
                ) * cooling_annual_kwhm2)
            except KeyError:
                logger.warning(
                    f"{self} - No cooling demand found. Returning zeros."
                )
                cooling_hourly = pd.Series([0] * len(load_balance_df.index), index=load_balance_df.index)
            cooling_hourly.name = "Cooling (kWh/m2)"

            # save to file
            cooling_hourly.to_csv(pth)

        # convert to normalised if needed
        if not normalised:
            cooling_hourly *= self.typical_building_gfa
            cooling_hourly.name = cooling_hourly.name.replace("(kWh/m2)", "(kWh)")

        # multiply by number of buildings if needed
        if not single_building:
            if not normalised:
                cooling_hourly *= self._number_of_buildings()

        return cooling_hourly

    def _cooling_energy_consumption(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
    ) -> pd.Series:
        """Get the cooling energy consumption of the typology, including effects from equipment performance.

        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional):
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings.

        Returns:
            pd.Series:
                The cooling energy consumption of the typology.
        """

        return (
            self._cooling_energy_demand(
                directory=directory,
                normalised=normalised,
                single_building=single_building,
            )
            / self.cooling_eer
        )

    def _heating_energy_demand(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
    ) -> pd.Series:
        """Get the heating energy demand of the typology.

        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional):
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings.

        Returns:
            pd.Series: A pandas Series of the hourly heating energy demand (prior to applying system efficiency) in kWh.
        """

        pth = Path(directory) / self.identifier / f"data_{inspect.stack()[0][3]}.csv"
        if all([RELOAD, pth.exists()]):
            # reload existing calculation
            logger.info(f"{self} - Reloading heating energy demand")
            heating_hourly = pd.read_csv(
                pth, index_col=0, header=0, parse_dates=True
            ).squeeze()
        else:
            # run calculation process
            logger.info(f"{self} - Calculating heating energy demand")

            # get load balance outputs - normalised, in kWh/m2
            load_balance_df = self.load_balance(
                directory=directory, normalised=True, single_building=True
            )

            # get EUI outputs, which are also always normalised
            eui_df = self.annual_eui(directory=directory)

            # calculate and scale heating load to eui to ensure consistency
            try:
                heating_annual_kwhm2 = eui_df["Heating"]
                heating_hourly = -load_balance_df["Load Balance Heating (kWh/m2)"]
                heating_hourly = ((
                    heating_hourly / heating_hourly.sum()
                ) * heating_annual_kwhm2)
            except KeyError:
                logger.warning(
                    f"{self} - No heating demand found. Returning zeros."
                )
                heating_hourly = pd.Series([0] * len(load_balance_df.index), index=load_balance_df.index)
            heating_hourly.name = "Heating (kWh/m2)"

            # save to file
            heating_hourly.to_csv(pth)

        # convert to normalised if needed
        if not normalised:
            heating_hourly *= self.typical_building_gfa
            heating_hourly.name = heating_hourly.name.replace("(kWh/m2)", "(kWh)")

        # multiply by number of buildings if needed
        if not single_building:
            if not normalised:
                heating_hourly *= self._number_of_buildings()

        return heating_hourly

    def _heating_energy_consumption(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
    ) -> pd.Series:
        """Get the heating energy consumption of the typology, including effects from equipment performance.

        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional):
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings.

        Returns:
            pd.Series:
                The heating energy consumption of the typology.
        """

        return (
            self._heating_energy_demand(
                directory=directory,
                normalised=normalised,
                single_building=single_building,
            )
            / self.heating_cop
        )

    def _lighting_energy_demand(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
    ) -> pd.Series:
        """Get the lighting energy demand of the typology.

        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional):
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings.

        Returns:
            pd.Series: A pandas Series of the hourly lighting energy demand in kWh.
        """

        pth = Path(directory) / self.identifier / f"data_{inspect.stack()[0][3]}.csv"
        if all([RELOAD, pth.exists()]):
            # reload existing calculation
            logger.info(f"{self} - Reloading lighting energy demand")
            lighting_hourly = pd.read_csv(
                    pth, index_col=0, header=0, parse_dates=True
                ).squeeze()
        else:
            # run calculation process
            logger.info(f"{self} - Calculating lighting energy demand")

            # get load balance outputs - normalised, in kWh/m2
            load_balance_df = self.load_balance(
                directory=directory, normalised=True, single_building=True
            )

            # get EUI outputs, which are also always normalised
            eui_df = self.annual_eui(directory=directory)

            # calculate and scale heating load to eui to ensure consistency
            try:
                lighting_annual_kwhm2 = eui_df["Interior Lighting"]
                lighting_hourly = -load_balance_df["Load Balance Lighting (kWh/m2)"]
                lighting_hourly = ((
                    lighting_hourly / lighting_hourly.sum()
                ) * lighting_annual_kwhm2)
            except KeyError:
                logger.warning(
                    f"{self} - No lighting demand found. Returning zeros."
                )
                lighting_hourly = pd.Series([0] * len(load_balance_df.index), index=load_balance_df.index)
            lighting_hourly.name = "Lighting (kWh/m2)"
            
            # save to file
            lighting_hourly.to_csv(pth)

        # convert to normalised if needed
        if not normalised:
            lighting_hourly *= self.typical_building_gfa
            lighting_hourly.name = lighting_hourly.name.replace("(kWh/m2)", "(kWh)")

        # multiply by number of buildings if needed
        if not single_building:
            if not normalised:
                lighting_hourly *= self._number_of_buildings()

        return lighting_hourly

    def _lighting_energy_consumption(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
    ) -> pd.Series:
        """Get the lighting energy consumption of the typology.

        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional):
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings.

        Returns:
            pd.Series:
                The lighting energy consumption of the typology.
        """

        return self._lighting_energy_demand(
            directory=directory, normalised=normalised, single_building=single_building
        )

    def _electric_equipment_energy_demand(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
    ) -> pd.Series:
        """Get the electric equipment energy demand of the typology.

        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional):
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings.

        Returns:
            pd.Series: A pandas Series of the hourly electric equipment energy demand in kWh.
        """

        pth = Path(directory) / self.identifier / f"data_{inspect.stack()[0][3]}.csv"
        if all([RELOAD, pth.exists()]):
            # reload existing calculation
            logger.info(f"{self} - Reloading electric equipment energy demand")
            electric_equipment_hourly = pd.read_csv(
                    pth, index_col=0, header=0, parse_dates=True
                ).squeeze()
        else:
            # run calculation process
            logger.info(f"{self} - Calculating electric equipment energy demand")

            # get load balance outputs - normalised, in kWh/m2
            load_balance_df = self.load_balance(
                directory=directory, normalised=True, single_building=True
            )

            # get EUI outputs, which are also always normalised
            eui_df = self.annual_eui(directory=directory)

            # calculate and scale heating load to eui to ensure consistency
            try:
                electric_equipment_annual_kwhm2 = eui_df["Electric Equipment"]
                electric_equipment_hourly = -load_balance_df[
                    "Load Balance Electric Equipment (kWh/m2)"
                ]
                electric_equipment_hourly = ((
                    electric_equipment_hourly / electric_equipment_hourly.sum()
                ) * electric_equipment_annual_kwhm2)
            except KeyError:
                logger.warning(
                    f"{self} - No electric equipment demand found. Returning zeros."
                )
                electric_equipment_hourly = pd.Series([0] * len(load_balance_df.index), index=load_balance_df.index)
            electric_equipment_hourly.name = "Electric Equipment (kWh/m2)"

            # save to file
            electric_equipment_hourly.to_csv(pth)

        # convert to normalised if needed
        if not normalised:
            electric_equipment_hourly *= self.typical_building_gfa
            electric_equipment_hourly.name = electric_equipment_hourly.name.replace(
                "(kWh/m2)", "(kWh)"
            )

        # multiply by number of buildings if needed
        if not single_building:
            if not normalised:
                electric_equipment_hourly *= self._number_of_buildings()

        return electric_equipment_hourly

    def _electric_equipment_energy_consumption(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
    ) -> pd.Series:
        """Get the electric equipment energy consumption of the typology.

        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional):
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings.

        Returns:
            pd.Series:
                The electric equipment energy consumption of the typology.
        """

        return self._electric_equipment_energy_demand(
            directory=directory, normalised=normalised, single_building=single_building
        )

    def _hot_water_energy_demand(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
    ) -> pd.Series:
        """Get the hot water energy demand of the typology.

        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional):
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings.

        Returns:
            pd.Series: A pandas Series of the hourly hot water energy demand (prior to applying system efficiency) in kWh.
        """

        pth = Path(directory) / self.identifier / f"data_{inspect.stack()[0][3]}.csv"
        if all([RELOAD, pth.exists()]):
            # reload existing calculation
            logger.info(f"{self} - Reloading hot water energy demand")
            hot_water_hourly = pd.read_csv(
                    pth, index_col=0, header=0, parse_dates=True
                ).squeeze()
        else:
            # run calculation process
            logger.info(f"{self} - Calculating hot water energy demand")

            # get load balance outputs - normalised, in kWh/m2
            load_balance_df = self.load_balance(
                directory=directory, normalised=True, single_building=True
            )

            # get EUI outputs, which are also always normalised
            eui_df = self.annual_eui(directory=directory)

            try:
                # calculate and scale hot water load to eui to ensure consistency
                hot_water_annual_kwhm2 = eui_df["Water Systems"]
                hot_water_hourly = -load_balance_df["Load Balance Service Hot Water (kWh/m2)"]
                hot_water_hourly = ((
                    hot_water_hourly / hot_water_hourly.sum()
                ) * hot_water_annual_kwhm2)
            except KeyError:
                hot_water_hourly = pd.Series(
                    np.zeros(8760), index=load_balance_df.index
                )
            hot_water_hourly.name = "Hot Water (kWh/m2)"

            # save to file
            hot_water_hourly.to_csv(pth)

        # convert to normalised if needed
        if not normalised:
            hot_water_hourly *= self.typical_building_gfa
            hot_water_hourly.name = hot_water_hourly.name.replace("(kWh/m2)", "(kWh)")

        # multiply by number of buildings if needed
        if not single_building:
            if not normalised:
                hot_water_hourly *= self._number_of_buildings()

        return hot_water_hourly

    def _hot_water_energy_consumption(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
    ) -> pd.Series:
        """Get the hot water energy consumption of the typology, including effects from equipment performance.

        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional):
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings.

        Returns:
            pd.Series:
                The hot water energy consumption of the typology.
        """

        return (
            self._hot_water_energy_demand(
                directory=directory,
                normalised=normalised,
                single_building=single_building,
            )
            / self.heating_cop
        )

    def _lift_energy_demand(self, normalised: bool = False, single_building: bool = False) -> pd.Series:
        """Estimate the annual energy consumption of a lift system in kWh/m2,
        for a building of the given height.

        References:
        - BRE. “NABERS UK: Guide to Design for Performance,” April 2021.

        Returns:
            pd.Series:
                The estimated hourly energy consumption of the lift system.
        """

        logger.info(f"{self} - Calculating lifts demand")

        if self.average_num_floors < 2:
            # no lifts in buildings < 2 floors
            lift_energy_hourly = pd.Series([0] * 8760, index=INDEX)
        else:
            alpha = (0.26 + 0.37) / 2

            annual_energy_kwh = ((528 * self.average_num_floors) + (5.5 * self.typical_building_gfa)) * (1 - alpha)
            
            # get occupants to distribute annual energy over
            occupants = self._occupants(per_building=True)

            # clip occupants so that the lowest level is at least 50% of the maximum level (to approximate high standby power)
            occupants.clip(lower=occupants.quantile(0.5), inplace=True)

            lift_energy_hourly = (occupants / occupants.sum()) * annual_energy_kwh
            lift_energy_hourly = lift_energy_hourly / self.typical_building_gfa  # normalised
        lift_energy_hourly.name = "Lifts (kWh/m2)"

        # convert to normalised if needed
        if not normalised:
            lift_energy_hourly *= self.typical_building_gfa
            lift_energy_hourly.name = lift_energy_hourly.name.replace("(kWh/m2)", "(kWh)")

        # multiply by number of buildings if needed
        if not single_building:
            if not normalised:
                lift_energy_hourly *= self._number_of_buildings()

        return lift_energy_hourly

    def _lift_energy_consumption(self, normalised: bool = False, single_building: bool = False) -> pd.Series:
        """Estimate the annual energy consumption of a lift system in kWh,
        for a building of the given height.

        Returns:
            pd.Series:
                The estimated hourly energy consumption of the lift system in kWh.
        """
        return self._lift_energy_demand(normalised=normalised, single_building=single_building)

    def _pump_energy_consumption(
        self,
        directory: Path = ROOT_DIRECTORY,
        normalised: bool = False,
        single_building: bool = False,
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

        # run calculation process
        logger.info(f"{self} - Calculating pumps energy consumption")

        # TODO - make this method dynamic, using timestep Cp and delta Ts ... maybe
        
        # load datasets - in normalised form for conversion later
        load_balance_df = self.load_balance(directory=directory, normalised=True, single_building=True)

        # HOT WATER FLOW #
        # NOTE: This doesn't account for pump minimum flow rates
        hot_water_delta_t = pd.Series(
            [25] * len(load_balance_df.index),
            index=load_balance_df.index,
            name="Hot Water Delta T (K)",
        )  # K
        hot_water_cp = pd.Series(
            [4.18] * len(load_balance_df.index),
            index=load_balance_df.index,
            name="Hot Water Specific Heat Capcity (J/g/K)",
        )  # J/g/K
        dhw_energy_demand = (
            self._hot_water_energy_demand(
                directory=directory, normalised=True, single_building=True
            )
        ).rename("DHW Energy Demand (kWh/m2)")
        space_heating_energy_demand = (
            self._heating_energy_demand(
                directory=directory, normalised=True, single_building=True
            )
        ).rename(
            "Space Heating Energy Demand (kWh/m2)"
        )  # kW/m2
        heating_energy_demand = (
            dhw_energy_demand + space_heating_energy_demand
        ).rename("Heating Energy Demand (kWh/m2)")

        hot_water_flowrate = (
            heating_energy_demand / (hot_water_delta_t * hot_water_cp)
        ).rename(
            "Hot Water Flow (l/s)"
        )  # l/s

        # CHILLED WATER FLOW #
        chilled_water_delta_t = pd.Series(
            [6] * len(load_balance_df.index),
            index=load_balance_df.index,
            name="Chilled Water Delta T (K)",
        )  # K
        chilled_water_cp = pd.Series(
            [4.18] * len(load_balance_df.index),
            index=load_balance_df.index,
            name="Chilled Water Specific Heat Capcity (J/g/K)",
        )  # J/g/K
        cooling_energy_demand = (
            self._cooling_energy_demand(
                directory=directory, normalised=True, single_building=True
            )
        ).rename(
            "Cooling Energy Demand (kWh/m2)"
        )  # kW/m2

        chilled_water_flowrate = (
            cooling_energy_demand / (chilled_water_delta_t * chilled_water_cp)
        ).rename(
            "Chilled Water Flow (l/s)"
        )  # l/s

        # calculate pump energy demand, and convert back to kW
        hot_water_pumping = (hot_water_flowrate * self.pump_power).rename(
            "Hot Water Pumps (W/m2)"
        )
        chilled_water_pumping = (chilled_water_flowrate * self.pump_power).rename(
            "Chilled Water Pumps (W/m2)"
        )
        pump_energy_consumption = (
            (hot_water_pumping + chilled_water_pumping) / 1000
        ).rename("Pumps (kWh/m2)")

        # convert to normalised if needed
        if not normalised:
            pump_energy_consumption *= self.typical_building_gfa
            pump_energy_consumption.name = pump_energy_consumption.name.replace("(kWh/m2)", "(kWh)")

        # multiply by number of buildings if needed
        if not single_building:
            if not normalised:
                pump_energy_consumption *= self._number_of_buildings()

        # Combine and sum
        return pump_energy_consumption
    
    def _ventilation_flowrate(self, directory: Path = ROOT_DIRECTORY, single_building: bool = False) -> pd.Series:
        """Helper method to provide the ventilation flowrate for the typology.
        
        Args:
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings
        
        Returns:
            pd.Series: The ventilation flowrate in l/s
        """

        sql_obj = self.sql_results(directory=directory)

        # get total volume of air being delivered to zones
        supply_air_flowrate = pd.concat(
            [
                collection_to_series(i)
                for i in sql_obj.data_collections_by_output_name(
                    "Zone Mechanical Ventilation Current Density Volume Flow Rate"
                )
            ],
            axis=1,
        )
        supply_air_flowrate = supply_air_flowrate.sum(axis=1).rename(
            supply_air_flowrate.columns[0]
        )

        # convert m3/s to l/s
        supply_air_ls = supply_air_flowrate * 1000

        # rename
        supply_air_ls.name = "Volume Flow Rate (l/s)"

        # multiply by number of buildings if needed
        if not single_building:
            supply_air_ls *= self._number_of_buildings()
        
        return supply_air_ls

    def _fan_energy_consumption(self, directory: Path = ROOT_DIRECTORY, normalised: bool = False, single_building: bool = False) -> pd.Series:
        """Estimate the energy consumption of fans in the building, based on
        simulation results. This includes efficiencies of the fans.

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
            directory (Path, optional):
                The directory where simulations results are stored. Defaults
                to ROOT_DIRECTORY.
            normalised (bool, optional): 
                If True, then the results are normalised by the total area of the typology.
            single_building (bool, optional):
                If True, then the results are not multiplied by the number of buildings

        Returns:
            pd.Series: Hourly fan energy consumption
        """

        logger.info(f"{self} - Calculating fans energy consumption")

        # get volume of air being supplied
        supply_air_ls = self._ventilation_flowrate(directory=directory, single_building=True)

        # get the energy demand in kWh/m2
        fan_energy = (supply_air_ls * self.fan_power / 1000) / self.typical_building_gfa  # kWh/m2

        # THIS PART IS ADDED TO ACCOUNT FOR ENERGY FOR FANS TO PUMP COOLED/HEATED AIR INTO A SPACE, NOT JUST VENTILATION
        # get cooling/heating flow recirculation energy
        load_balance_w = self.load_balance(directory=directory, normalised=False, single_building=True)
        # max htg/clg load - to get peak fan energy
        max_load_w = max(load_balance_w["Load Balance Cooling (kWh)"].max(), load_balance_w["Load Balance Heating (kWh)"].max()) / 1000
        delta_t = 12
        flow_ls = max_load_w / (1.2 * 1.02 * delta_t)
        flow_energy_kwh = (self.fan_power * flow_ls) / 1000
        # distribute over the htg/clg loads, by interpolating between highest and lowest total htg/clg, and making the peak wattage the flow energy the peak energy
        xx = (load_balance_w["Load Balance Cooling (kWh)"] + load_balance_w["Load Balance Heating (kWh)"])
        xx = np.interp(xx, [xx.min(), xx.max()], [0, flow_energy_kwh])
        # add to the fan_energy_consumption
        fan_energy += xx

        fan_energy = fan_energy.rename(
            "Fans (kWh/m2)"
        )

        # convert to normalised if needed
        if not normalised:
            fan_energy *= self.typical_building_gfa
            fan_energy.name = fan_energy.name.replace("(kWh/m2)", "(kWh)")

        # multiply by number of buildings if needed
        if not single_building:
            if not normalised:
                fan_energy *= self._number_of_buildings()

        return fan_energy

    def energy_consumption(
        self, directory: Path = ROOT_DIRECTORY, normalised: bool = False, single_building: bool = False
    ) -> pd.DataFrame:
        """Get the energy consumption of the typology, including effects from equipment performance.

        Args:
            directory (Path):
                The directory to save the results in.
            normalised (bool):
                Normalise the results by area. Default is False.
            single_building (bool):
                If True, then the results are not multiplied by the number of buildings.

        Returns:
            pd.DataFrame:
                The annual hourly energy consumption of the typology.
        """

        pth = Path(directory) / self.identifier / f"data_{inspect.stack()[0][3]}.csv"
        if all((RELOAD, pth.exists())):
            logger.info(f"{self} - Reloading energy consumption")
            energy_consumption_df = pd.read_csv(pth, index_col=0, header=0, parse_dates=True)
        else:
            # ensure simulation has been run
            self._simulate(directory)

            # get each of the energy consumptions, including system efficiencies
            cooling = self._cooling_energy_consumption(
                directory=directory, normalised=True, single_building=False
            )
            heating = self._heating_energy_consumption(
                directory=directory, normalised=True, single_building=False
            )
            lighting = self._lighting_energy_consumption(
                directory=directory, normalised=True, single_building=False
            )
            electric_equipment = self._electric_equipment_energy_consumption(
                directory=directory, normalised=True, single_building=False
            )
            hot_water = self._hot_water_energy_consumption(
                directory=directory, normalised=True, single_building=False
            )
            lifts = self._lift_energy_consumption(normalised=True, single_building=False)
            pumps = self._pump_energy_consumption(
                directory=directory, normalised=True, single_building=False
            )
            fans = self._fan_energy_consumption(directory=directory, normalised=True, single_building=False)

            # combine results
            energy_consumption_df = pd.concat(
                [
                    cooling,
                    heating,
                    lighting,
                    electric_equipment,
                    hot_water,
                    lifts,
                    pumps,
                    fans,
                ],
                axis=1,
            )

            # save to file
            energy_consumption_df.to_csv(pth)
        
        # convert to normalised if needed
        if not normalised:
            energy_consumption_df *= self.typical_building_gfa
            energy_consumption_df.columns = [i.replace("(kWh/m2)", "(kWh)") for i in energy_consumption_df.columns]
        
        # multiply by number of buildings if needed
        if not single_building:
            if not normalised:
                energy_consumption_df *= self._number_of_buildings()

        return energy_consumption_df

    def _all_data(
        self, directory: Path = ROOT_DIRECTORY, normalised: bool = False, single_building: bool = False, include_external: bool = True
    ) -> pd.DataFrame:
        """Join all hourly data together in a single DataFrame. Useful for debugging."""
        
        objects = [
            self.energy_consumption(directory=directory, normalised=normalised, single_building=single_building),
            self._ventilation_flowrate(directory=directory, single_building=single_building),
            self.external_conditions(),
        ]
        keys = ["Energy Consumption", "Ventilation Flowrate", "External Conditions",]
        if include_external:
            objects.append(self.space_conditions(directory=directory))
            keys.append("Space Conditions")

        df = pd.concat(
            objects,
            axis=1,
            keys=keys,
        )

        return df

    def annual_eui(self, directory: Path = ROOT_DIRECTORY) -> pd.Series:
        """Get the annual energy use intensity of the typology in kWh.

        Args:
            directory (Path):
                The directory to save the results in.

        Returns:
            pd.DataFrame:
                The annual energy use intensity of the typology.
        """
        pth = Path(directory) / self.identifier / f"data_{inspect.stack()[0][3]}.csv"
        if all([RELOAD, pth.exists()]):
            logger.info(f"{self} - Reloading annual EUI")
            return pd.read_csv(pth, index_col=0, header=0).squeeze()

        s = annual_eui(self._sql_file(directory=directory).as_posix())
        s.to_csv(pth)
        return s

    def plot_annual_monthly(
        self,
        directory: Path = ROOT_DIRECTORY,
        ax: plt.Axes = None,
        rule: str = "MS",
        label: bool = True,
        legend: bool = True,
        single_building: bool = False,
    ) -> plt.Axes:
        """Plot the monthly energy consumption of the typology.

        Args:
            epw (EPW):
                The EPW file to use for the simulation.
            directory (Path):
                The directory to save the results in.
            ax (plt.Axes):
                The axes to plot on. Default is None.
            rule (str):
                The resampling rule to use. Default is "MS".
            label (bool):
                Label the axes. Default is True.
            legend (bool):
                Show the legend. Default is True.
        """

        # logger.info(f"{self} - Plotting annual monthly energy consumption")

        # get the typology hourly energy consumption, hourly, in kWh
        df = self.energy_consumption(directory, normalised=False, single_building=single_building)

        # sort in order to most to least energy consumption
        df = df[df.sum(axis=0).sort_values(ascending=False).index]

        if ax is None:
            ax = plt.gca()

        ax = stacked_bar(df=df, ax=ax, rule=rule, label=label, legend=legend)

        _ = ax.set_title(f"{self.identifier} - Energy Consumption")

        return ax

    def plot_pie(
        self,
        directory: Path = ROOT_DIRECTORY,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        ax: plt.Axes = None,
        label: bool = True,
        legend: bool = True,
        single_building: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Plot a pie chart of the annual energy consumption of the typology.

        Args:
            epw (EPW):
                The EPW file to use for the simulation.
            directory (Path):
                The directory to save the results in.
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

        # logger.info(f"{self} - Plotting annual energy consumption pie chart")

        series = (
            self.energy_consumption(directory, normalised=False, single_building=single_building)
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

        _ = ax.set_title(
            f"{self.identifier} - Energy Consumption\n{describe_analysis_period(analysis_period)}\n{series.sum():,.0f}{unit} (over {self.typical_building_gfa if single_building else self.total_area:,.0f}m$^{2}$)"
        )

        return ax

    def plot_diurnal(
        self,
        directory: Path = ROOT_DIRECTORY,
        ax: plt.Axes = None,
        legend: bool = True,
        logy: bool = False,
        normalised: bool = False,
        single_building: bool = False,
    ) -> plt.Axes:
        """Plot a monthly diurnal profile for energy consumption of the typology.

        Args:
            epw (EPW):
                The EPW file to use for the simulation.
            directory (Path):
                The directory to save the results in.
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

        df = self.energy_consumption(directory=directory, normalised=normalised, single_building=single_building)

        if ax is None:
            ax = plt.gca()

        ax = diurnal(df, ax=ax, legend=legend, logy=logy)

        _ = ax.set_title(f"{self.identifier} - Energy Consumption")

        return ax

    def plot_duration_curve(
        self,
        directory: Path = ROOT_DIRECTORY,
        ax: plt.Axes = None,
        remove_zero: bool = True,
        legend: bool = True,
        normalised: bool = False,
        single_building: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Plot a duration curve for energy consumption of the typology.

        Args:
            epw (EPW):
                The EPW file to use for the simulation.
            directory (Path):
                The directory to save the results in.
            ax (plt.Axes):
                The axes to plot on. Default is None.
            remove_zero (bool):
                Remove zero values. Default is True.
            legend (bool):
                Show the legend. Default is True.
            **kwargs:
                Additional keyword arguments to pass to the plt.hist function.

        Returns:
            plt.Axes:
                The axes object.
        """

        # logger.info(f"{self} - Plotting duration curve")

        df = self.energy_consumption(directory=directory, normalised=normalised, single_building=single_building)

        if ax is None:
            ax = plt.gca()

        ax = duration_curve(df, ax=ax, legend=legend, remove_zero=remove_zero, **kwargs)

        _ = ax.set_title(f"{self.identifier} - Energy Consumption - Duration Curve")

        return ax

    # def run_everything(self, directory: Path = ROOT_DIRECTORY):
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
