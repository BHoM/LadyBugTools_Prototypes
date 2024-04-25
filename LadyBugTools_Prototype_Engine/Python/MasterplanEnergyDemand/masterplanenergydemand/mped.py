"""Main module."""

# pylintXXX: disable=too-many-lines, import-error, logging-fstring-interpolation, unused-import, too-many-locals, too-many-statements, too-many-branches, no-name-in-module, line-too-long
# pylintXXX: disable=logging-fstring-interpolation

# pylint: disable=E0401
import concurrent
import concurrent.futures
import inspect
import json
import logging
import warnings
from enum import Enum
from pathlib import Path
from uuid import uuid4

# pylint: enable=E0401

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd
from honeybee.boundarycondition import Outdoors
from honeybee.config import folders as hb_folders
from honeybee.facetype import Floor, RoofCeiling, Wall
from honeybee.model import Face, Model, Shade
from honeybee.room import Room
from honeybee.typing import valid_string
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.internalmass import InternalMass
from honeybee_energy.lib.constructionsets import (
    ConstructionSet,
    construction_set_by_identifier,
)
from honeybee_energy.lib.scheduletypelimits import humidity, temperature
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from honeybee_energy.lib.scheduletypelimits import schedule_type_limit_by_identifier
from honeybee_energy.result.loadbalance import LoadBalance, SQLiteResult
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval
from honeybee_energy.simulation.parameter import (
    RunPeriod,
    ShadowCalculation,
    SimulationControl,
    SimulationOutput,
    SimulationParameter,
    SizingParameter,
)
from ladybug.wea import EPW, AnalysisPeriod, HourlyContinuousCollection
from ladybug_geometry.geometry2d import Point2D, Polygon2D, Vector2D
from ladybug_geometry.geometry3d import Face3D, Point3D, Vector3D, LineSegment3D

from . import FORMATTING, LOAD_BALANCE_TERMS, DPI, FIGSIZE_RECTANGLE, FIGSIZE_SQUARE
from .programtypes import ProgramType, _building_program_type_by_identifier
from .enum import (
    BuildingForm,
    BuildingType,
    ConstructionType,
    EconomizerType,
    TerrainType,
    Vintage,
    typical_construction_type,
    typical_floor_height,
    typical_footprint_area,
    typical_gfa,
    typical_glazing_ratio,
    typical_num_floors,
    typical_context_distance,
)
from .utilities import (
    angle_from_north,
    cardinality,
    contrasting_color,
    estimate_sri_properties,
    plot_diurnal,
    plot_monthly_stacked_bar,
    plot_pie,
    plot_heating_cooling_series,
    plot_duration_curve,
    typical_lift_energy,
    convert_dataframe,
)


class MPED:
    """The Masterplan Energy Demand (MPED) class."""

    def __init__(
        self,
        epw_file: Path,
        project_identifier: str = None,
        case_identifier: str = None,
        building_identifier: str = None,
        total_area: float = None,
        average_footprint_area: float = None,
        average_num_floors: int = None,
        average_floor_height: float = None,
        building_form: BuildingForm = None,
        aspect_ratio: float = None,
        rotation: float = None,
        building_type: BuildingType = None,
        construction_type: ConstructionType = None,
        vintage: Vintage = None,
        terrain: TerrainType = None,
        context_shading: float = None,
        glazing_ratio: list[float] = None,
        skylight_ratio: float = None,
        wall_u_value: list[float] = None,
        wall_sri: list[float] = None,
        floor_u_value: float = None,
        roof_u_value: float = None,
        window_u_value: list[float] = None,
        window_shgc: list[float] = None,
        skylight_u_value: float = None,
        skylight_shgc: float = None,
        roof_sri: float = None,
        occupant_density: float = None,
        lighting_power_density: float = None,
        equipment_power_density: float = None,
        infiltration_rate: float = None,
        ventilation_rate: float = None,
        heating_setpoint: float = None,
        heating_setback: float = None,
        cooling_setpoint: float = None,
        cooling_setback: float = None,
        humidifying_setpoint: float = None,
        humidifying_setback: float = None,
        dehumidifying_setpoint: float = None,
        dehumidifying_setback: float = None,
        economizer_type: EconomizerType = None,
        sensible_heat_recovery_effectiveness: float = None,
        latent_heat_recovery_effectiveness: float = None,
        demand_controlled_ventilation: bool = None,
        daylight_dimming: bool = None,
        heating_cop: float = None,
        cooling_eer: float = None,
        fan_power: float = None,
        pump_power: float = None,
    ) -> None:
        """_"""
        self.epw_file = epw_file
        self.project_identifier = project_identifier
        self.case_identifier = case_identifier
        self.building_identifier = building_identifier
        self.building_type = building_type
        self.total_area = total_area
        self.construction_type = construction_type
        self.vintage = vintage
        self.terrain = terrain
        self.average_footprint_area = average_footprint_area
        self.average_num_floors = average_num_floors
        self.average_floor_height = average_floor_height
        self.building_form = building_form
        self.aspect_ratio = aspect_ratio
        self.rotation = rotation
        self.context_shading = context_shading
        self.glazing_ratio = glazing_ratio
        self.skylight_ratio = skylight_ratio
        self.wall_u_value = wall_u_value
        self.wall_sri = wall_sri
        self.floor_u_value = floor_u_value
        self.roof_u_value = roof_u_value
        self.window_u_value = window_u_value
        self.window_shgc = window_shgc
        self.skylight_u_value = skylight_u_value
        self.skylight_shgc = skylight_shgc
        self.roof_sri = roof_sri
        self.occupant_density = occupant_density
        self.lighting_power_density = lighting_power_density
        self.equipment_power_density = equipment_power_density
        self.infiltration_rate = infiltration_rate
        self.ventilation_rate = ventilation_rate
        self.heating_setpoint = heating_setpoint
        self.heating_setback = heating_setback
        self.cooling_setpoint = cooling_setpoint
        self.cooling_setback = cooling_setback
        self.humidifying_setpoint = humidifying_setpoint
        self.humidifying_setback = humidifying_setback
        self.dehumidifying_setpoint = dehumidifying_setpoint
        self.dehumidifying_setback = dehumidifying_setback
        self.economizer_type = economizer_type
        self.sensible_heat_recovery_effectiveness = sensible_heat_recovery_effectiveness
        self.latent_heat_recovery_effectiveness = latent_heat_recovery_effectiveness
        self.demand_controlled_ventilation = demand_controlled_ventilation
        self.daylight_dimming = daylight_dimming
        self.heating_cop = heating_cop
        self.cooling_eer = cooling_eer
        self.fan_power = fan_power
        self.pump_power = pump_power

        # post-init validation
        if self.heating_setback > self.heating_setpoint:
            raise ValueError(
                f"heating_setback ({self.heating_setback}) must be less than or equal to heating_setpoint ({self.heating_setpoint})"
            )

        if self.cooling_setback < self.cooling_setpoint:
            raise ValueError(
                f"cooling_setback ({self.cooling_setback}) must be greater than or equal to cooling_setpoint ({self.cooling_setpoint})"
            )

        if self.humidifying_setback > self.humidifying_setpoint:
            raise ValueError(
                f"humidifying_setback ({self.humidifying_setback}) must be less than or equal to humidifying_setpoint ({self.humidifying_setpoint})"
            )

        if self.dehumidifying_setback < self.dehumidifying_setpoint:
            raise ValueError(
                f"dehumidifying_setback ({self.dehumidifying_setback}) must be greater than or equal to dehumidifying_setpoint ({self.dehumidifying_setpoint})"
            )

    def __str__(self) -> str:
        return f"{self.project_identifier}::{self.case_identifier}::{self.building_identifier}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: "MPED") -> bool:
        return self.to_dict() == other.to_dict()

    # region: Interoperables

    def to_dict(self) -> dict:
        """Convert this object to a dictionary."""
        return {k[1:]: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "MPED":
        """Create this object from a dictionary."""
        return cls(**data)

    def to_json(self) -> Path:
        """Save the object to a JSON file."""

        class CustomEncoder(json.JSONEncoder):
            """A custom JSON encoder to handle enumerables and paths."""

            def default(self, o):
                if isinstance(o, Enum):
                    return o.name
                if isinstance(o, Path):
                    return str(o)
                return super().default(o)

        with open(self._config_json, "w") as fp:
            fp.write(json.dumps(self.to_dict(), indent=4, cls=CustomEncoder))
        return self._config_json

    @classmethod
    def from_json(cls, json_file: Path) -> "MPED":
        """Create an object from a JSON file."""

        with open(json_file, "r") as f:
            data = json.load(f)

        keys_enums = {
            "building_type": BuildingType,
            "construction_type": ConstructionType,
            "vintage": Vintage,
            "terrain": TerrainType,
            "building_form": BuildingForm,
            "economizer_type": EconomizerType,
        }
        for k, v in keys_enums.items():
            if k in data:
                data[k] = v[data[k]]

        return cls.from_dict(data)

    def to_series(self) -> pd.Series:
        """Convert this object to a pandas Series."""
        return pd.Series(self.to_dict())

    @classmethod
    def from_series(cls, series: pd.Series) -> "MPED":
        """Create this object from a pandas Series."""

        keys = set(inspect.getfullargspec(MPED).args[1:])

        # remove keys that shouldnt exist
        for key in series.index:
            if key not in keys:
                series.drop(key, inplace=True)

        for key in keys:
            if key not in series.index:
                raise AttributeError(f"Series missing key: {key}")

        series.fillna(None, inplace=True)

        return cls(**series.to_dict())

    @classmethod
    def from_excel(
        cls,
        excel_file: str | Path,
        sheet_name: str,
    ) -> list["MPED"]:
        """Create a list of these objects from an Excel file.

        Args:
            excel_file (str | Path): The path to the Excel file.
            sheet_name (str): The name of the sheet in the Excel file, which contains the data.

        Returns:
            list[Mped]: A list of Mped objects.
        """

        excel_file = Path(excel_file).absolute()

        logging.info(f"Creating case/s from {excel_file}")  # pylint: disable=W1203

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            df = (
                pd.read_excel(
                    io=excel_file, sheet_name=sheet_name, index_col="Key", header=0
                )
                .fillna(np.nan)
                .replace([np.nan], [None])
            )
            df.drop(columns=["Unit"], inplace=True)

        objects = []
        for _, d in df.items():

            # construct combo properties
            d["glazing_ratio"] = [
                d["glazing_ratio_N"],
                d["glazing_ratio_NE"],
                d["glazing_ratio_E"],
                d["glazing_ratio_SE"],
                d["glazing_ratio_S"],
                d["glazing_ratio_SW"],
                d["glazing_ratio_W"],
                d["glazing_ratio_NW"],
            ]
            d["wall_u_value"] = [
                d["wall_u_value_N"],
                d["wall_u_value_NE"],
                d["wall_u_value_E"],
                d["wall_u_value_SE"],
                d["wall_u_value_S"],
                d["wall_u_value_SW"],
                d["wall_u_value_W"],
                d["wall_u_value_NW"],
            ]
            d["window_u_value"] = [
                d["window_u_value_N"],
                d["window_u_value_NE"],
                d["window_u_value_E"],
                d["window_u_value_SE"],
                d["window_u_value_S"],
                d["window_u_value_SW"],
                d["window_u_value_W"],
                d["window_u_value_NW"],
            ]
            d["window_shgc"] = [
                d["window_shgc_N"],
                d["window_shgc_NE"],
                d["window_shgc_E"],
                d["window_shgc_SE"],
                d["window_shgc_S"],
                d["window_shgc_SW"],
                d["window_shgc_W"],
                d["window_shgc_NW"],
            ]
            d["wall_sri"] = [
                d["wall_sri_N"],
                d["wall_sri_NE"],
                d["wall_sri_E"],
                d["wall_sri_SE"],
                d["wall_sri_S"],
                d["wall_sri_SW"],
                d["wall_sri_W"],
                d["wall_sri_NW"],
            ]

            # remove any not in object argspec
            args = inspect.getfullargspec(MPED).args[1:]
            for k in list(d.keys()):
                if k not in args:
                    d.pop(k)

            # create object and append
            objects.append(cls(**d))

        return objects

    # endregion

    # region: Getters, Setters and Validation

    @property
    def project_identifier(self):
        """Getter for the project_identifier property."""
        return self._project_identifier

    @project_identifier.setter
    def project_identifier(self, value):
        """Setter for the project_identifier property."""
        if value is None:
            value = uuid4().hex[:8]
        valid_string(value, "project_identifier")
        self._project_identifier = value

    @property
    def case_identifier(self):
        """Getter for the case_identifier property."""
        return self._case_identifier

    @case_identifier.setter
    def case_identifier(self, value):
        """Setter for the case_identifier property."""
        if value is None:
            value = uuid4().hex[:8]
        valid_string(value, "case_identifier")
        self._case_identifier = value

    @property
    def building_identifier(self):
        """Getter for the building_identifier property."""
        return self._building_identifier

    @building_identifier.setter
    def building_identifier(self, value):
        """Setter for the building_identifier property."""
        if value is None:
            value = uuid4().hex[:8]
        valid_string(value, "building_identifier")
        self._building_identifier = value

    @property
    def epw_file(self):
        """Getter for the epw_file property."""
        return self._epw_file

    @epw_file.setter
    def epw_file(self, value):
        """Setter for the epw_file property."""
        prop_name = inspect.currentframe().f_code.co_name
        if not isinstance(value, Path | str):
            raise ValueError(f"{self} - {prop_name} must be a Path or str")
        value = Path(value)
        if not value.exists():
            raise FileNotFoundError(f"{self} - {prop_name}, {value} does not exist")
        if value.suffix != ".epw":
            raise ValueError(f"{self} - {prop_name} must have a .epw extension")
        self._epw_file = value

    @property
    def building_type(self):
        """Getter for the building_type property."""
        return self._building_type

    @building_type.setter
    def building_type(self, value):
        """Setter for the building_type property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = BuildingType.OFFICE_MEDIUM
            logging.info(
                "%s - no %s provided, using default value of %s", self, prop_name, value
            )
        else:
            try:
                value = BuildingType(value)
            except ValueError:
                value = BuildingType[value]
        self._building_type = value

    @property
    def construction_type(self):
        """Getter for the construction_type property."""
        return self._construction_type

    @construction_type.setter
    def construction_type(self, value):
        """Setter for the construction_type property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = typical_construction_type(building_type=self.building_type)
            logging.info(
                "%s - no %s provided, using default value of %s", self, prop_name, value
            )
        else:
            try:
                value = ConstructionType(value)
            except ValueError:
                value = ConstructionType[value]
        self._construction_type = value

    @property
    def vintage(self):
        """Getter for the vintage property."""
        return self._vintage

    @vintage.setter
    def vintage(self, value):
        """Setter for the vintage property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = Vintage.ASHRAE_901_2019
            logging.info(
                "%s - no %s provided, using default value of %s", self, prop_name, value
            )
        else:
            try:
                value = Vintage(value)
            except ValueError:
                value = Vintage[value]
        self._vintage = value

    @property
    def terrain(self):
        """Getter for the terrain property."""
        return self._terrain

    @terrain.setter
    def terrain(self, value):
        """Setter for the terrain property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = TerrainType.URBAN
            logging.info(
                "%s - no %s provided, using default value of %s", self, prop_name, value
            )
        else:
            try:
                value = TerrainType(value)
            except ValueError:
                value = TerrainType[value]
        self._terrain = value

    @property
    def building_form(self):
        """Getter for the building_form property."""
        return self._building_form

    @building_form.setter
    def building_form(self, value):
        """Setter for the building_form property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = BuildingForm.CUBOID
            logging.info(
                "%s - no %s provided, using default value of %s", self, prop_name, value
            )
        else:
            try:
                value = BuildingForm(value)
            except:
                value = BuildingForm[value]
        self._building_form = value

    @property
    def economizer_type(self):
        """Getter for the economizer_type property."""
        return self._economizer_type

    @economizer_type.setter
    def economizer_type(self, value):
        """Settter for the economizer_type property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = EconomizerType.NO_ECONOMIZER
            logging.info(
                "%s - no %s provided, using default value of %s",
                self,
                prop_name,
                value,
            )
        else:
            try:
                value = EconomizerType(value)
            except:
                value = EconomizerType[value]
        self._economizer_type = value

    @property
    def context_shading(self):
        """Getter for the context_shading property."""
        return self._context_shading

    @context_shading.setter
    def context_shading(self, value):
        """Setter for the context_shading property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 0.75
            logging.info(
                "%s - no %s provided, using default value of %.2f",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0 or value > 1:
            raise ValueError(f"{self} - {prop_name} must be within the range 0 to 1")
        self._context_shading = value

    @property
    def total_area(self):
        """Getter for the total_area property."""
        return self._total_area

    @total_area.setter
    def total_area(self, value):
        """Setter for the total_area property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = typical_gfa(building_type=self.building_type)
            logging.info(
                "%s - no %s provided, using default value of %.0fm2",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value <= 0:
            raise ValueError(f"{self} - {prop_name} must be greater than 0")
        try:
            if value < self.average_footprint_area:
                raise ValueError(
                    f"{self} - {prop_name} ({value}) must be greater than or equal to average_footprint_area ({self.average_footprint_area})"
                )
        except AttributeError:
            pass
        self._total_area = value

    @property
    def average_footprint_area(self):
        """Getter for the average_footprint_area property."""
        return self._average_footprint_area

    @average_footprint_area.setter
    def average_footprint_area(self, value):
        """Setter for the average_footprint_area property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = typical_footprint_area(building_type=self.building_type)
            logging.info(
                "%s - no %s provided, using default value of %.0fm2",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value <= 0:
            raise ValueError(f"{self} - {prop_name} must be greater than 0")
        try:
            if value > self.total_area:
                raise ValueError(
                    f"{self} - {prop_name} ({value}) must be less than or equal to total_area ({self.total_area})"
                )
        except AttributeError:
            pass
        self._average_footprint_area = value

    @property
    def average_num_floors(self):
        """Getter for the average_num_floors property."""
        return self._average_num_floors

    @average_num_floors.setter
    def average_num_floors(self, value):
        """Setter for the average_num_floors property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = typical_num_floors(building_type=self.building_type)
            logging.info(
                "%s - no %s provided, using default value of %.0f",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, int):
            raise ValueError(f"{self} - {prop_name} must be an integer")
        if value < 1:
            raise ValueError(f"{self} - {prop_name} must be greater than 0")
        self._average_num_floors = value

    @property
    def average_floor_height(self):
        """Getter for the average_floor_height property."""
        return self._average_floor_height

    @average_floor_height.setter
    def average_floor_height(self, value):
        """Setter for the average_floor_height property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = typical_floor_height(building_type=self.building_type)
            logging.info(
                "%s - no %s provided, using default value of %.0fm",
                self,
                prop_name,
                value,
            )
        if value <= 0:
            raise ValueError(f"{self} - {prop_name} must be greater than 0")
        self._average_floor_height = value

    @property
    def aspect_ratio(self):
        """Getter for the aspect_ratio property."""
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, value):
        """Setter for the aspect_ratio property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 1
            logging.info(
                "%s - no %s provided, using default value of %.1f",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 1:
            raise ValueError(f"{self} - {prop_name} must be greater than 1")
        self._aspect_ratio = value

    @property
    def rotation(self):
        """Getter for the rotation property."""
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        """Setter for the rotation property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 0
            logging.info(
                "%s - no %s provided, using default value of %.0f°",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0 or value > 360:
            raise ValueError(f"{self} - {prop_name} must be between 0 and 360")
        self._rotation = value

    @property
    def glazing_ratio(self):
        """Getter for the glazing_ratio property."""
        return self._glazing_ratio

    @glazing_ratio.setter
    def glazing_ratio(self, value):
        """Setter for the glazing_ratio property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = [typical_glazing_ratio(building_type=self.building_type)] * 8
            logging.info(
                "%s - no %s provided, using default value of %.2f for all facade orientations",
                self,
                prop_name,
                value[0],
            )
        if not isinstance(value, list):
            raise ValueError(f"{self} - {prop_name} must be a list")
        if len(value) != 8:
            raise ValueError(f"{self} - {prop_name} must have 8 items")
        for i in value:
            if not isinstance(i, (int, float)):
                raise ValueError(f"{self} - {prop_name} items must be numbers")
            if i < 0 or i > 0.95:
                raise ValueError(
                    f"{self} - {prop_name} items must be between 0 and 0.95"
                )
        self._glazing_ratio = value

    @property
    def skylight_ratio(self):
        """Getter for the skylight_ratio property."""
        return self._skylight_ratio

    @skylight_ratio.setter
    def skylight_ratio(self, value):
        """Setter for the skylight_ratio property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 0
            logging.info(
                "%s - no %s provided, using default value of %.2f",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0 or value > 0.95:
            raise ValueError(f"{self} - {prop_name} must be between 0 and 0.95")
        self._skylight_ratio = value

    @property
    def wall_u_value(self):
        """Getter for the wall_u_value property."""
        return self._wall_u_value

    @wall_u_value.setter
    def wall_u_value(self, value):
        """Setter for the wall_u_value property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = [0.5] * 8
            logging.info(
                "%s - no %s provided, using default value of %.2fW/m2K for all facade orientations",
                self,
                prop_name,
                value[0],
            )
        if not isinstance(value, list):
            raise ValueError(f"{self} - {prop_name} must be a list")
        if len(value) != 8:
            raise ValueError(f"{self} - {prop_name} must have 8 items")
        for i in value:
            if not isinstance(i, (int, float)):
                raise ValueError(f"{self} - {prop_name} items must be numbers")
            if i < 0.05 or i > 6:
                raise ValueError(
                    f"{self} - {prop_name} items must be between 0.05 and 6"
                )
        self._wall_u_value = value

    @property
    def floor_u_value(self):
        """Getter for the floor_u_value property."""
        return self._floor_u_value

    @floor_u_value.setter
    def floor_u_value(self, value):
        """Setter for the floor_u_value property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 0.5
            logging.info(
                "%s - no %s provided, using default value of %.3fW/m2K",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - prop_name must be a number")
        if value < 0.05 or value > 6:
            raise ValueError(f"{self} - prop_name must be between 0.05 and 6")
        self._floor_u_value = value

    @property
    def roof_u_value(self):
        """Getter for the roof_u_value property."""
        return self._roof_u_value

    @roof_u_value.setter
    def roof_u_value(self, value):
        """Setter for the roof_u_value property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 0.3
            logging.info(
                "%s - no %s provided, using default value of %.3fW/m2K",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0.05 or value > 6:
            raise ValueError(f"{self} - {prop_name} must be between 0.05 and 6")
        self._roof_u_value = value

    @property
    def window_u_value(self):
        """Getter for the window_u_value property."""
        return self._window_u_value

    @window_u_value.setter
    def window_u_value(self, value):
        """Setter for the window_u_value property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = [2] * 8
            logging.info(
                "%s - no %s provided, using default value of %.3fW/m2K for all facade orientations",
                self,
                prop_name,
                value[0],
            )
        if not isinstance(value, list):
            raise ValueError(f"{self} - {prop_name} must be a list")
        if len(value) != 8:
            raise ValueError(f"{self} - {prop_name} must have 8 items")
        for i in value:
            if not isinstance(i, (int, float)):
                raise ValueError(f"{self} - {prop_name} items must be numbers")
            if i < 0.05 or i > 6:
                raise ValueError(
                    f"{self} - {prop_name} items must be between 0.05 and 6"
                )
        self._window_u_value = value

    @property
    def window_shgc(self):
        """Getter for the window_shgc property."""
        return self._window_shgc

    @window_shgc.setter
    def window_shgc(self, value):
        """Setter for the window_shgc property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = [0.4] * 8
            logging.info(
                "%s - no %s provided, using default value of %.3f for all facade orientations",
                self,
                prop_name,
                value[0],
            )
        if not isinstance(value, list):
            raise ValueError(f"{self} - {prop_name} must be a list")
        if len(value) != 8:
            raise ValueError(f"{self} - {prop_name} must have 8 items")
        for i in value:
            if not isinstance(i, (int, float)):
                raise ValueError(f"{self} - {prop_name} items must be numbers")
            if i < 0.01 or i > 0.99:
                raise ValueError(
                    f"{self} - {prop_name} items must be between 0.01 and 0.99"
                )
        self._window_shgc = value

    @property
    def skylight_u_value(self):
        """Getter for the skylight_u_value property."""
        return self._skylight_u_value

    @skylight_u_value.setter
    def skylight_u_value(self, value):
        """Setter for the skylight_u_value property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 2
            logging.info(
                "%s - no %s provided, using default value of %.3fW/m2K",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0.05 or value > 6:
            raise ValueError(f"{self} - {prop_name} must be between 0.05 and 6")
        self._skylight_u_value = value

    @property
    def skylight_shgc(self):
        """Getter for the skylight_shgc property."""
        return self._skylight_shgc

    @skylight_shgc.setter
    def skylight_shgc(self, value):
        """Setter for the skylight_shgc property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 0.4
            logging.info(
                "%s - no %s provided, using default value of %.3f",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0.01 or value > 0.99:
            raise ValueError(f"{self} - {prop_name} must be between 0.01 and 0.99")
        self._skylight_shgc = value

    @property
    def roof_sri(self):
        """Getter for the roof_sri property."""
        return self._roof_sri

    @roof_sri.setter
    def roof_sri(self, value):
        """Setter for the roof_sri property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 35
            logging.info(
                "%s - no %s provided, using default value of %.0f",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 1 or value > 122:
            raise ValueError(f"{self} - {prop_name} must be between 1 and 122")
        self._roof_sri = value

    @property
    def wall_sri(self):
        """Getter for the wall_sri property."""
        return self._wall_sri

    @wall_sri.setter
    def wall_sri(self, value):
        """Setter for the wall_sri property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = [35] * 8
            logging.info(
                "%s - no %s provided, using default value of %0.0f for all facade orientations",
                self,
                prop_name,
                value[0],
            )
        if not isinstance(value, list):
            raise ValueError(f"{self} - {prop_name} must be a list")
        if len(value) != 8:
            raise ValueError(f"{self} - {prop_name} must have 8 items")
        for i in value:
            if not isinstance(i, (int, float)):
                raise ValueError(f"{self} - {prop_name} items must be numberic")
            if i < 1 or i > 122:
                raise ValueError(
                    f"{self} - {prop_name} items must be between 1 and 122"
                )
        self._wall_sri = value

    @property
    def occupant_density(self):
        """Getter for the occupant_density property."""
        return self._occupant_density

    @occupant_density.setter
    def occupant_density(self, value):
        """Setter for the occupant_density property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            if program.people is None:
                value = 0
                logging.info(
                    "%s - no default %s is available for this building type, occupant_density set to %.3f person/m2",
                    self,
                    prop_name,
                    value,
                )
            else:
                value = program.people.people_per_area
                logging.info(
                    "%s - no %s provided, using default value of %.3f person/m2",
                    self,
                    prop_name,
                    value,
                )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0:
            raise ValueError(f"{self} - {prop_name} must be greater than or equal to 0")
        self._occupant_density = value

    @property
    def lighting_power_density(self):
        """Getter for the lighting_power_density property."""
        return self._lighting_power_density

    @lighting_power_density.setter
    def lighting_power_density(self, value):
        """Setter for the lighting_power_density property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            if program.lighting is None:
                value = 0
                logging.info(
                    "%s - no default %s is available for this building type, set to %.3fW/m2",
                    self,
                    prop_name,
                    value,
                )
            else:
                value = program.lighting.watts_per_area
                logging.info(
                    "%s - no %s is provided, using default value of %.3fW/m2",
                    self,
                    prop_name,
                    value,
                )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0:
            raise ValueError(f"{self} - {prop_name} must be greater than or equal to 0")

        self._lighting_power_density = value

    @property
    def equipment_power_density(self):
        """Getter for the equipment_power_density property."""
        return self._equipment_power_density

    @equipment_power_density.setter
    def equipment_power_density(self, value):
        """Setter for the equipment_power_density property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            if program.electric_equipment is None:
                value = 0
                logging.info(
                    "%s - no default %s is available for this building type, set to %.3fW/m2",
                    self,
                    prop_name,
                    value,
                )
            else:
                value = program.electric_equipment.watts_per_area
                logging.info(
                    "%s - no %s is provided, using default value of %.3fW/m2",
                    self,
                    prop_name,
                    value,
                )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0:
            raise ValueError(f"{self} - {prop_name} must be greater than or equal to 0")

        self._equipment_power_density = value

    @property
    def infiltration_rate(self):
        """Getter for the infiltration_rate property."""
        return self._infiltration_rate

    @infiltration_rate.setter
    def infiltration_rate(self, value):
        """Setter for the infiltration_rate property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            if program.infiltration is None:
                value = 0.0006
                logging.info(
                    "%s - no default %s is available for this building type, set to %.6fm3/s/m2",
                    self,
                    prop_name,
                    value,
                )
            else:
                value = program.infiltration.flow_per_exterior_area
                logging.info(
                    "%s - no %s is provided, using default value of %.6fm3/s/m2",
                    self,
                    prop_name,
                    value,
                )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0:
            raise ValueError(f"{self} - {prop_name} must be greater than or equal to 0")

        self._infiltration_rate = value

    @property
    def ventilation_rate(self):
        """Getter for the ventilation_rate property."""
        return self._ventilation_rate

    @ventilation_rate.setter
    def ventilation_rate(self, value):
        """Setter for the ventilation_rate property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            if program.ventilation is None:
                value = 0
                logging.info(
                    "%s - no default %s is available for this building type, set to %.6fm3/s/person",
                    self,
                    prop_name,
                    value,
                )
            else:
                value = program.ventilation.flow_per_person
                logging.info(
                    "%s - no %s is provided, using default value of %.6fm3/s/m2",
                    self,
                    prop_name,
                    value,
                )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0:
            raise ValueError(f"{self} - {prop_name} must be greater than or equal to 0")

        self._ventilation_rate = value

    @property
    def heating_setpoint(self):
        """Getter for the heating_setpoint property."""
        return self._heating_setpoint

    @heating_setpoint.setter
    def heating_setpoint(self, value):
        """Setter for the heating_setpoint property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            value = program.setpoint.heating_setpoint
            logging.info(
                "%s - no %s is provided, using default value of %.1f°C",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        self._heating_setpoint = value

    @property
    def heating_setback(self):
        """Getter for the heating_setback property."""
        return self._heating_setback

    @heating_setback.setter
    def heating_setback(self, value):
        """Setter for the heating_setback property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            value = program.setpoint.heating_setback
            logging.info(
                "%s - no %s is provided, using default value of %.1f°C",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        self._heating_setback = value

    @property
    def cooling_setpoint(self):
        """Getter for the cooling_setpoint property."""
        return self._cooling_setpoint

    @cooling_setpoint.setter
    def cooling_setpoint(self, value):
        """Setter for the cooling_setpoint property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            value = program.setpoint.cooling_setpoint
            logging.info(
                "%s - no %s is provided, using default value of %.1f°C",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        self._cooling_setpoint = value

    @property
    def cooling_setback(self):
        """Getter for the cooling_setback property."""
        return self._cooling_setback

    @cooling_setback.setter
    def cooling_setback(self, value):
        """Setter for the cooling_setback property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            value = program.setpoint.cooling_setback
            logging.info(
                "%s - no %s is provided, using default value of %.1f°C",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        self._cooling_setback = value

    @property
    def humidifying_setpoint(self):
        """Getter for the humidifying_setpoint property."""
        return self._humidifying_setpoint

    @humidifying_setpoint.setter
    def humidifying_setpoint(self, value):
        """Setter for the humidifying_setpoint property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            value = program.setpoint.humidifying_setpoint
            if value is None:
                value = 40
            logging.info(
                "%s - no %s is provided, using default value of %.1f%%",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")

        if value < 0 or value > 100:
            raise ValueError(f"{self} - humidifying_setpoint must be between 0 and 100")

        self._humidifying_setpoint = value

    @property
    def humidifying_setback(self):
        """Getter for the humidifying_setback property."""
        return self._humidifying_setback

    @humidifying_setback.setter
    def humidifying_setback(self, value):
        """Setter for the humidifying_setback property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            value = program.setpoint.humidifying_setback
            if value is None:
                value = 0
            logging.info(
                "%s - no %s is provided, using default value of %.1f%%",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")

        if value < 0 or value > 100:
            raise ValueError(f"{self} - humidifying_setpoint must be between 0 and 100")

        self._humidifying_setback = value

    @property
    def dehumidifying_setpoint(self):
        """Getter for the dehumidifying_setpoint property."""
        return self._dehumidifying_setpoint

    @dehumidifying_setpoint.setter
    def dehumidifying_setpoint(self, value):
        """Setter for the dehumidifying_setpoint property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            value = program.setpoint.dehumidifying_setpoint
            if value is None:
                value = 60
            logging.info(
                "%s - no %s is provided, using default value of %.1f%%",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")

        if value < 0 or value > 100:
            raise ValueError(f"{self} - humidifying_setpoint must be between 0 and 100")

        self._dehumidifying_setpoint = value

    @property
    def dehumidifying_setback(self):
        """Getter for the dehumidifying_setback property."""
        return self._dehumidifying_setback

    @dehumidifying_setback.setter
    def dehumidifying_setback(self, value):
        """Setter for the dehumidifying_setback property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            program = _building_program_type_by_identifier(
                building_type=self.building_type.value
            ).duplicate()
            value = program.setpoint.dehumidifying_setback
            if value is None:
                value = 100
            logging.info(
                "%s - no %s is provided, using default value of %.1f%%",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")

        if value < 0 or value > 100:
            raise ValueError(f"{self} - humidifying_setpoint must be between 0 and 100")

        self._dehumidifying_setback = value

    @property
    def sensible_heat_recovery_effectiveness(self):
        """Getter for the sensible_heat_recovery_effectiveness property."""
        return self._sensible_heat_recovery_effectiveness

    @sensible_heat_recovery_effectiveness.setter
    def sensible_heat_recovery_effectiveness(self, value):
        """Setter for the sensible_heat_recovery_effectiveness property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 0
            logging.info(
                "%s - no %s provided, using default value of %.2f",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0 or value > 1:
            raise ValueError(f"{self} - {prop_name} must be between 0 and 1")
        self._sensible_heat_recovery_effectiveness = value

    @property
    def latent_heat_recovery_effectiveness(self):
        """Getter for the latent_heat_recovery_effectiveness property."""
        return self._latent_heat_recovery_effectiveness

    @latent_heat_recovery_effectiveness.setter
    def latent_heat_recovery_effectiveness(self, value):
        """Setter for the latent_heat_recovery_effectiveness property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 0
            logging.info(
                "%s - no %s provided, using default value of %.2f",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0 or value > 1:
            raise ValueError(f"{self} - {prop_name} must be between 0 and 1")
        self._latent_heat_recovery_effectiveness = value

    @property
    def demand_controlled_ventilation(self):
        """Getter for the demand_controlled_ventilation property."""
        return self._demand_controlled_ventilation

    @demand_controlled_ventilation.setter
    def demand_controlled_ventilation(self, value):
        """Setter for the demand_controlled_ventilation property."""
        if value is None:
            value = False
        else:
            value = bool(value)
        self._demand_controlled_ventilation = value

    @property
    def daylight_dimming(self):
        """Getter for the daylight_dimming property."""
        return self._daylight_dimming

    @daylight_dimming.setter
    def daylight_dimming(self, value):
        """Setter for the daylight_dimming property."""
        if value is None:
            value = False
        else:
            value = bool(value)
        self._daylight_dimming = value

    @property
    def heating_cop(self):
        """Getter for the heating_cop property."""
        return self._heating_cop

    @heating_cop.setter
    def heating_cop(self, value):
        """Setter for the heating_cop property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 1
            logging.info(
                "%s - no %s provided, using default value of %.2f",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0:
            raise ValueError(f"{self} - {prop_name} must be greater than 0")
        self._heating_cop = value

    @property
    def cooling_eer(self):
        """Getter for the cooling_eer property."""
        return self._cooling_eer

    @cooling_eer.setter
    def cooling_eer(self, value):
        """Setter for the cooling_eer property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 1
            logging.info(
                "%s - no %s provided, using default value of %.2f",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0:
            raise ValueError(f"{self} - {prop_name} must be greater than 0")
        self._cooling_eer = value

    @property
    def fan_power(self):
        """Getter for the fan_power property."""
        return self._fan_power

    @fan_power.setter
    def fan_power(self, value):
        """Setter for the fan_power property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 1.8
            logging.info(
                "%s - no %s provided, using default value of %.2fW/l/s",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0:
            raise ValueError(f"{self} - {prop_name} must be greater than 0")
        self._fan_power = value

    @property
    def pump_power(self):
        """Getter for the pump_power property."""
        return self._pump_power

    @pump_power.setter
    def pump_power(self, value):
        """Setter for the pump_power property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 0.35
            logging.info(
                "%s - no %s provided, using default value of %.2fW/l/s",
                self,
                prop_name,
                value,
            )
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self} - {prop_name} must be a number")
        if value < 0:
            raise ValueError(f"{self} - {prop_name} must be greater than 0")
        self._pump_power = value

    # endregion

    # region: Process filepaths

    @property
    def project_directory(self) -> Path:
        """Get the project directory for the simulation."""
        project_dir = (
            Path(hb_folders.default_simulation_folder) / self.project_identifier
        )
        project_dir.mkdir(exist_ok=True, parents=True)
        return project_dir

    @property
    def case_directory(self) -> Path:
        """Get the case directory for the simulation."""
        case_dir = self.project_directory / self.case_identifier
        case_dir.mkdir(exist_ok=True, parents=True)
        return case_dir

    @property
    def simulation_directory(self) -> Path:
        """Get the simulation directory for the simulation."""
        simulation_dir = self.case_directory / self.building_identifier
        simulation_dir.mkdir(exist_ok=True, parents=True)
        return simulation_dir

    @property
    def _model_json(self) -> Path:
        """Get the path to the model JSON file for the simulation."""
        return self.simulation_directory / f"{self.building_identifier}.hbjson"

    @property
    def _config_json(self) -> Path:
        """Get the path to the config JSON file for the current object."""
        return self.simulation_directory / f"{self.__class__.__name__}.json"

    @property
    def _simulation_parameters_json(self) -> Path:
        """Get the path to the simulation parameters JSON file for the simulation."""
        return self.simulation_directory / "simulation_parameter.json"

    @property
    def _idf_file(self) -> Path:
        """Get the path to the IDF file for the simulation."""
        return self.simulation_directory / "run" / "in.idf"

    @property
    def _sql_file(self) -> Path:
        """Get the path to the SQL results file from the simulation."""
        return self.simulation_directory / "run" / "eplusout.sql"

    # endregion

    # region: Computed properties

    @property
    def number_of_buildings(self) -> int:
        """_"""
        return self.total_area / (self.average_footprint_area * self.average_num_floors)

    @property
    def average_building_gfa(self) -> float:
        """Get the typical GFA for an individual building."""
        return self.total_area / self.number_of_buildings

    @property
    def average_building_height(self) -> float:
        """Get the typical height for an individual building."""
        return self.average_num_floors * self.average_floor_height

    @property
    def epw(self) -> EPW:
        """Get the EPW object for the simulation."""
        return EPW(self.epw_file)

    def _footprint(self) -> Polygon2D:
        """Create the footprint for the building."""

        base_poly = Polygon2D.from_rectangle(
            base=3,
            height=3 / float(self.aspect_ratio),
            base_point=Point2D(),
            height_vector=Vector2D(0, 1),
        )

        match self.building_form:
            case BuildingForm.CUBOID:
                footprint = base_poly
            case BuildingForm.L_SHAPED:
                cutting_shape: Polygon2D = Polygon2D.from_rectangle(
                    base=3,
                    height=3 / float(self.aspect_ratio),
                    base_point=base_poly.center,
                    height_vector=Vector2D(0, 1),
                )
                footprint = base_poly.boolean_difference(
                    polygon=cutting_shape, tolerance=0.1
                )[0]
            case BuildingForm.U_SHAPED:
                cutting_shape: Polygon2D = Polygon2D.from_rectangle(
                    base=1,
                    height=4 / float(self.aspect_ratio),
                    base_point=base_poly.center.move(
                        -Vector2D(
                            (base_poly.center - Point2D()).x / 3,
                            (base_poly.center - Point2D()).y / 4,
                        )
                    ),
                    height_vector=Vector2D(0, 1),
                )
                footprint = base_poly.boolean_difference(
                    polygon=cutting_shape, tolerance=0.1
                )[0]
            case _:
                raise ValueError(f"{self} - How did you get here?")

        # rotate based on input
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
        footprint = footprint.scale(n)

        return footprint

    def visualise_footprint(self, ax: plt.Axes = None) -> plt.Axes:
        """_"""
        if ax is None:
            ax = plt.gca()

        ax.set_aspect("equal")
        ax.autoscale()

        ax.add_patch(
            mpatches.Polygon(
                xy=np.array(self._footprint().to_array()),
                closed=True,
                fill=False,
                edgecolor="black",
                linewidth=2,
            )
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        return ax

    def _base_room(
        self,
    ) -> Room:
        """Create single ground-floor room."""

        floor_face = Face3D(
            boundary=[Point3D(*i.to_array()) for i in self._footprint().vertices]
        )
        ceiling_face = floor_face.move(
            moving_vec=Vector3D(0, 0, self.average_floor_height)
        ).flip()

        wall_faces = [
            Face3D.from_extrusion(
                line_segment=i,
                extrusion_vector=Vector3D(0, 0, self.average_floor_height),
            )
            for i in floor_face.boundary_segments
        ]

        hb_faces = [
            Face(identifier="level_00_floor", geometry=floor_face, type=Floor()),
            Face(
                identifier="level_00_roofceiling",
                geometry=ceiling_face,
                type=RoofCeiling(),
            ),
        ]
        for n, i in enumerate(wall_faces):
            hb_faces.append(
                Face(
                    identifier=f"level_00_wall_{n:02d}",
                    geometry=i,
                    type=Wall(),
                )
            )

        room = Room(identifier="level_00", faces=hb_faces, tolerance=0.01)

        lk = self._directional_lookup()

        for face in room.walls:
            face: Face
            if isinstance(face.boundary_condition, Outdoors):
                # get face cardinal direction
                _dir = cardinality(angle_from_north(face.normal), directions=8)
                face.apertures_by_ratio(lk[_dir]["glazing_ratio"], 0.1)

        return room

    def _wall_constructions(self) -> list[OpaqueConstruction]:
        """Create the directional wall constructions for the simulation."""
        # create directional constructions
        wall_constructions = []
        for wall_u, _dir in zip(
            *[self.wall_u_value, ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]]
        ):
            wall_constructions.append(
                OpaqueConstruction.from_simple_parameters(
                    identifier=f"U {wall_u:0.2f} Wall {_dir}",
                    r_value=1 / wall_u,
                    roughness="MediumRough",
                    thermal_absorptance=0.9,
                    solar_absorptance=0.7,
                )
            )
        return wall_constructions

    def _window_constructions(self) -> list[WindowConstruction]:
        """Create the directional window constructions for the simulation."""
        # create directional constructions
        window_constructions = []
        for window_u, window_shgc, _dir in zip(
            *[
                self.window_u_value,
                self.window_shgc,
                ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            ]
        ):
            window_constructions.append(
                WindowConstruction.from_simple_parameters(
                    identifier=f"U {window_u:0.2f} SHGC {window_shgc:0.2f} Window {_dir}",
                    u_factor=window_u,
                    shgc=window_shgc,
                    vt=0.6,
                )
            )
        return window_constructions

    def _directional_lookup(self) -> to_dict:
        """_"""
        wall_constructions = self._wall_constructions()
        window_constructions = self._window_constructions()
        d = {}
        for n, direction in enumerate(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]):
            d[direction] = {
                "glazing_ratio": self.glazing_ratio[n],
                "window_u_value": self.window_u_value[n],
                "window_shgc": self.window_shgc[n],
                "wall_u_value": self.wall_u_value[n],
                "wall_construction": wall_constructions[n],
                "window_construction": window_constructions[n],
            }

        return d

    def _ground_construction(self) -> OpaqueConstruction:
        """Create the ground floor construction for the simulation."""
        return OpaqueConstruction.from_simple_parameters(
            identifier=f"U {self.floor_u_value:0.2f} Ground Floor",
            r_value=1 / self.floor_u_value,
            roughness="MediumRough",
            thermal_absorptance=0.9,
            solar_absorptance=0.7,
        )

    def _roof_construction(self) -> OpaqueConstruction:
        """Create the roof construction for the simulation."""
        roof_solar_absorptance, roof_thermal_emittance = estimate_sri_properties(
            self.roof_sri
        )
        return OpaqueConstruction.from_simple_parameters(
            identifier=f"U {self.roof_u_value:0.2f} SRI {self.roof_sri} Roof",
            r_value=1 / self.roof_u_value,
            roughness="MediumRough",
            thermal_absorptance=roof_thermal_emittance,
            solar_absorptance=roof_solar_absorptance,
        )

    def _skylight_construction(self) -> WindowConstruction:
        """Create the skylight construction for the simulation."""
        return WindowConstruction.from_simple_parameters(
            identifier=f"U {self.skylight_u_value:0.2f} SHGC {self.skylight_shgc:0.2f} Skylight",
            u_factor=self.skylight_u_value,
            shgc=self.skylight_shgc,
            vt=0.6,
        )

    def _base_model(self) -> Model:
        """Create the base model (prior to assignment of program and fabric properties) for the simulation."""

        base_room = self._base_room()

        rooms = [base_room]
        for i in range(self.average_num_floors):
            if i == 0:
                continue
            new_room = base_room.duplicate()
            new_room.move(moving_vec=Vector3D(0, 0, self.average_floor_height * i))
            new_room.identifier = f"level_{i:02d}"
            for face in new_room.faces:
                face.identifier = face.identifier.replace("level_00", f"level_{i:02d}")
                for aperture in face.apertures:
                    aperture.identifier = aperture.identifier.replace(
                        "level_00", f"level_{i:02d}"
                    )
            rooms.append(new_room)

        # add context shades based on terrain
        context_distance = typical_context_distance(self.terrain)
        context_height = self.average_building_height * 0.75
        shades = []
        if context_distance < 1000 and context_height > 0:
            floor_face = base_room.floors[0].geometry
            points = np.array([i.to_array()[:2] for i in floor_face.vertices])
            hull = ConvexHull(points, qhull_options="QJ")
            hull_pts = np.stack([points[hull.vertices, 0], points[hull.vertices, 1]]).T
            context_base = Polygon2D.from_array(
                point_array=[Point2D(*i) for i in hull_pts]
            ).offset(-context_distance)
            # create context shades and add transmissivity shcedule also
            shd_transmissivity = ScheduleRuleset.from_constant_value(
                "context_shade_transmissivity",
                1 - self.context_shading,
                schedule_type_limit_by_identifier("Fractional"),
            )
            shades = []
            for segment in context_base.segments:
                _shd = Shade(
                    identifier="context_shade",
                    geometry=Face3D.from_extrusion(
                        LineSegment3D.from_line_segment2d(segment),
                        extrusion_vector=Vector3D(0, 0, context_height),
                    ),
                )
                _shd.properties.energy.transmittance_schedule = shd_transmissivity
                shades.append(_shd)

        # create model from rooms, and solve adjacencies
        base_model = Model.from_objects(
            identifier=self.building_identifier, objects=rooms + shades
        )
        base_model.solve_adjacency(
            merge_coplanar=False,
            intersect=False,
            overwrite=False,
            air_boundary=False,
            adiabatic=False,
            tolerance=None,
        )

        # add skylights if required
        if self.skylight_ratio > 0:
            for room in base_model.rooms:
                for face in room.faces:
                    face: Face
                    if isinstance(face.type, RoofCeiling) and isinstance(
                        face.boundary_condition, Outdoors
                    ):
                        face.apertures_by_ratio(self.skylight_ratio, tolerance=0.1)

        return base_model

    def model(self) -> Model:
        """Create the model for the simulation."""

        # TODO - based on shadedness/terrain type, make an assumption for overshading ans create a rough shade structure around teh mdoel to account for this

        base_model = self._base_model()

        # apply constructions
        constr_set = self.default_construction_set.duplicate()
        constr_set.floor_set.ground_construction = self._ground_construction()
        constr_set.roof_ceiling_set.exterior_construction = self._roof_construction()
        constr_set.aperture_set.skylight_construction = self._skylight_construction()

        # apply constructions to model
        for room in base_model.rooms:
            room.properties.energy.construction_set = constr_set

            # add internal mass
            internal_mass = InternalMass(
                identifier=f"{room.identifier}_internal_mass",
                area=room.exposed_area,
                construction=constr_set.wall_set.exterior_construction,
            )
            room.properties.energy.add_internal_mass(internal_mass)

        # apply directional constructions
        lk = self._directional_lookup()
        for face in base_model.faces:
            face: Face
            if isinstance(face.boundary_condition, Outdoors) and isinstance(
                face.type, Wall
            ):
                # get face cardinal direction
                _dir = cardinality(angle_from_north(face.normal), directions=8)
                face.properties.energy.construction = lk[_dir]["wall_construction"]
                for aperture in face.apertures:
                    aperture.properties.energy.construction = lk[_dir][
                        "window_construction"
                    ]

        # apply program
        program = self.default_program.duplicate()

        if program.people is not None:
            program.people.people_per_area = self.occupant_density

        if program.lighting is not None:
            program.lighting.watts_per_area = self.lighting_power_density

        if program.electric_equipment is not None:
            program.electric_equipment.watts_per_area = self.equipment_power_density

        if program.ventilation is not None:
            program.ventilation.flow_per_person = self.ventilation_rate

        if program.infiltration is not None:
            program.infiltration.flow_per_exterior_area = self.infiltration_rate

        if program.setpoint is None:
            raise NotImplementedError(
                "No setpoint exists in the program, not sure what to do here. Panic?"
            )

        # get the heating schedule and modify it to use the new heating setpoint and setback
        old_heating_schedule = np.array(
            program.setpoint.heating_schedule.data_collection().values
        )
        new_heating_schedule = np.interp(
            old_heating_schedule,
            [old_heating_schedule.min(), old_heating_schedule.max()],
            [self.heating_setback, self.heating_setpoint],
        )
        program.setpoint.heating_schedule = ScheduleFixedInterval(
            "heating_schedule",
            new_heating_schedule,
            temperature,
        )
        # get the cooling schedule and modify it to use the new cooling setpoint and setback
        old_cooling_schedule = np.array(
            program.setpoint.cooling_schedule.data_collection().values
        )
        new_cooling_schedule = np.interp(
            old_cooling_schedule,
            [old_cooling_schedule.max(), old_cooling_schedule.min()],
            [self.cooling_setback, self.cooling_setpoint],
        )
        program.setpoint.cooling_schedule = ScheduleFixedInterval(
            "cooling_schedule",
            new_cooling_schedule,
            temperature,
        )

        # check for humidifying/dehumidifying setpoints, and if not found, then use the schedule from heating/cooling and create new humidifying/dehumidifying schedules
        if program.setpoint.humidifying_setpoint is None:
            program.setpoint.humidifying_schedule = ScheduleFixedInterval(
                "humidifying_schedule",
                np.where(
                    np.array(program.setpoint.heating_schedule.data_collection.values)
                    == program.setpoint.heating_setpoint,
                    self.humidifying_setpoint,
                    self.humidifying_setback,
                ),
                humidity,
            )
            program.setpoint.dehumidifying_schedule = ScheduleFixedInterval(
                "dehumidifying_schedule",
                np.where(
                    np.array(program.setpoint.heating_schedule.data_collection.values)
                    == program.setpoint.heating_setpoint,
                    self.dehumidifying_setpoint,
                    self.dehumidifying_setback,
                ),
                humidity,
            )
        else:
            program.setpoint.humidifying_setpoint = self.humidifying_setpoint
            program.setpoint.humidifying_setback = self.humidifying_setback
            program.setpoint.dehumidifying_setpoint = self.dehumidifying_setpoint
            program.setpoint.dehumidifying_setback = self.dehumidifying_setback

        ideal_air = IdealAirSystem(
            identifier="ideal_air_system",
            economizer_type=self.economizer_type.value,
            demand_controlled_ventilation=self.demand_controlled_ventilation,
            sensible_heat_recovery=self.sensible_heat_recovery_effectiveness,
            latent_heat_recovery=self.latent_heat_recovery_effectiveness,
        )

        for room in base_model.rooms:
            room.properties.energy.program_type = program
            room.properties.energy.hvac = ideal_air

            if self.daylight_dimming:
                room.properties.energy.add_daylight_control_to_center(
                    distance_from_floor=0.8, control_fraction=0.5
                )

        # save model to disk
        base_model.to_hbjson(
            folder=self._model_json.parent.as_posix(), triangulate_sub_faces=True
        )

        return base_model

    @property
    def occupancy_schedule(self) -> pd.Series:
        """Create the occupancy schedule for the simulation."""
        _model = self.model()
        if _model.rooms[0].properties.energy.program_type.people is None:
            values = np.zeros(8760)
        else:
            values = (
                _model.rooms[0]
                .properties.energy.program_type.people.occupancy_schedule.data_collection()
                .values
            )

        return pd.Series(
            values,
            index=pd.to_datetime(AnalysisPeriod().datetimes),
            name="Occupancy",
        )

    # endregion

    # region: Useful defaults

    @property
    def default_index(self) -> pd.DatetimeIndex:
        """Create a default index for the hourly data."""
        return pd.to_datetime(AnalysisPeriod().datetimes)

    @property
    def default_program(self) -> ProgramType:
        """Get the default program for the building."""
        return _building_program_type_by_identifier(
            building_type=self.building_type.value
        )

    @property
    def default_construction_set(self) -> ConstructionSet:
        """Get the default construction set for the current simulation."""

        return construction_set_by_identifier(
            construction_set_identifier=f"{self.vintage.value}::ClimateZone{int(self.epw.ashrae_climate_zone[0])}::{self.construction_type.value}"
        )

    # endregion

    # region: Queries

    @property
    def _config_exists(self) -> bool:
        """Check if the config matches the current object."""
        if self._config_json.exists():
            other = MPED.from_json(self._config_json)
            if other == self:
                return True
        return False

    @property
    def _results_exist(self) -> bool:
        """Check if the results exist for the simulation."""
        if self._config_exists:
            if self._sql_file.exists():
                return True
        return False

    # endregion

    # region: Simulation
    def sql(self) -> Path:
        """Simulate the results and return the SQL file."""

        if self._results_exist:
            logging.info("%s - Reloading existing results", self)
            return self._sql_file

        for fp in self.simulation_directory.glob("**/*"):
            if fp.is_file():
                fp.unlink()

        logging.info("%s - Simulating results", self)

        # create model and save json to disk
        self.model()

        simulation_control = SimulationControl(
            do_zone_sizing=True,
            do_system_sizing=True,
            do_plant_sizing=True,
            run_for_sizing_periods=False,
            run_for_run_periods=True,
        )
        epw = self.epw
        design_days = [
            epw.approximate_design_day("WinterDesignDay"),
            epw.approximate_design_day("SummerDesignDay"),
        ]
        sizing_parameter = SizingParameter(design_days=design_days)
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
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature",
            "Zone Mechanical Ventilation Current Density Volume Flow Rate",
            "Water Use Equipment Heating Rate",
            "Water Use Equipment Hot Water Volume",
            "Site Outdoor Air Drybulb Temperature",
            "Zone Mean Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Air Relative Humidity",
            "Site Outdoor Air Relative Humidity",
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

        with open(self._simulation_parameters_json, "w", encoding="utf-8") as fp:
            json.dump(simulation_parameter.to_dict(), fp)

        osw = to_openstudio_osw(
            self.simulation_directory.as_posix(),
            self._model_json.as_posix(),
            self._simulation_parameters_json.as_posix(),
            additional_measures=None,
            epw_file=self.epw_file.as_posix(),
        )
        _, idf = run_osw(osw, silent=True)

        sql, _, _, _, _ = run_idf(
            idf_file_path=idf,
            epw_file_path=self.epw_file.as_posix(),
            expand_objects=True,
            silent=True,
        )

        # save information to disk
        self.to_json()
        epw.save(file_path=self.simulation_directory / Path(self.epw_file).name)

        return Path(sql).absolute()

    # endregion

    # region: Outputs

    def thermal_load_balance(self) -> pd.DataFrame:
        """The timestep-wise thermal load balance for the simulation for all
        heat flow pathways in/around the building."""

        if not self._results_exist:
            self.sql()

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.csv"
        )

        if filename.exists():
            logging.info(
                "%s - Reloading %s", self, inspect.currentframe().f_code.co_name
            )
            return pd.read_csv(
                filename,
                header=0,
                parse_dates=True,
                index_col=0,
            )

        logging.info("%s - Processing %s", self, inspect.currentframe().f_code.co_name)
        lb = LoadBalance.from_sql_file(
            model=self.model(), sql_path=self._sql_file.as_posix()
        )

        xx = []
        for i in lb.load_balance_terms(floor_normalized=False, include_storage=True):
            i.convert_to_unit("Wh")
            xx.append(
                pd.Series(
                    i.values,
                    index=self.default_index,
                    name=f"{i.header.metadata['type']} ({i.header.unit})",
                )
            )
        lb_df = pd.concat(xx, axis=1)
        for col in LOAD_BALANCE_TERMS:
            if col not in lb_df:
                lb_df[col] = 0
        lb_df = (
            lb_df[abs(lb_df.mean()).sort_values(ascending=False).index]
            * self.number_of_buildings
        )
        lb_df.to_csv(filename)

        return lb_df

    def thermal_load_balance_normalised(self) -> pd.DataFrame:
        """The timestep-wise thermal load balance for the simulation for all
        heat flow pathways in/around the building, normalised by floor area."""

        if not self._results_exist:
            self.sql()

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.csv"
        )

        if filename.exists():
            logging.info("%s - Reloading thermal load balance normalised", self)
            return pd.read_csv(
                filename,
                header=0,
                parse_dates=True,
                index_col=0,
            )

        logging.info("%s - Processing thermal load balance normalised", self)
        lb_df_normalised = self.thermal_load_balance() / (
            self.model().floor_area * self.number_of_buildings
        )
        lb_df_normalised.columns = [
            i.replace("Wh", "Wh/m2") for i in lb_df_normalised.columns
        ]
        lb_df_normalised.to_csv(filename)

        return lb_df_normalised

    def energy_consumption(self) -> pd.DataFrame:
        """The hourly energy consumed by the building typology simulated."""

        if not self._results_exist:
            self.sql()

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.csv"
        )

        if filename.exists():
            logging.info("%s - Reloading energy consumption results", self)
            return pd.read_csv(
                filename,
                header=0,
                parse_dates=True,
                index_col=0,
            )

        logging.info("%s - Processing energy consumption results", self)
        sqlr = SQLiteResult(file_path=self._sql_file.as_posix())
        idx = self.default_index

        # COOLING
        cooling_energy = []
        for i in sqlr.data_collections_by_output_name(
            output_name="Zone Ideal Loads Supply Air Total Cooling Energy"
        ):
            i.convert_to_unit("Wh")
            cooling_energy.append(
                pd.Series(
                    i.values,
                    index=idx,
                    name=f"{i.header.metadata['type']} ({i.header.unit})",
                )
            )
        if len(cooling_energy) == 0:
            logging.warning("%s - No cooling energy available!", self)
            cooling_energy = pd.Series(np.zeros(8760), index=idx, name="Cooling (Wh)")
        else:
            cooling_energy = (
                pd.concat(cooling_energy, axis=1).sum(axis=1).rename("Cooling (Wh)")
            )

        # HEATING
        heating_energy = []
        for i in sqlr.data_collections_by_output_name(
            output_name="Zone Ideal Loads Supply Air Total Heating Energy"
        ):
            i.convert_to_unit("Wh")
            heating_energy.append(
                pd.Series(
                    i.values,
                    index=idx,
                    name=f"{i.header.metadata['type']} ({i.header.unit})",
                )
            )
        if len(heating_energy) == 0:
            logging.warning(f"{self} - No heating energy available!")
            heating_energy = pd.Series(np.zeros(8760), index=idx, name="Heating (Wh)")
        else:
            heating_energy = (
                pd.concat(heating_energy, axis=1).sum(axis=1).rename("Heating (Wh)")
            )

        # HOT WATER
        service_hot_water_energy = []
        for i, j in zip(
            *[
                sqlr.data_collections_by_output_name(
                    output_name="Water Use Equipment Zone Sensible Heat Gain Energy"
                ),
                sqlr.data_collections_by_output_name(
                    output_name="Water Use Equipment Zone Latent Gain Energy"
                ),
            ]
        ):
            i.convert_to_unit("Wh")
            j.convert_to_unit("Wh")
            service_hot_water_energy.append(
                (
                    pd.Series(
                        i.values,
                        index=idx,
                    )
                    + pd.Series(
                        j.values,
                        index=idx,
                    )
                )
            )
        if len(service_hot_water_energy) == 0:
            logging.warning(f"{self} - No service hot water energy available!")
            service_hot_water_energy = pd.Series(
                np.zeros(8760), index=idx, name="Service Hot Water (Wh)"
            )
        else:
            service_hot_water_energy = (
                pd.concat(service_hot_water_energy, axis=1)
                .sum(axis=1)
                .rename("Service Hot Water (Wh)")
            )

        # FANS
        ventilation_rate = []
        for i in sqlr.data_collections_by_output_name(
            output_name="Zone Mechanical Ventilation Current Density Volume Flow Rate"
        ):
            # TODO - check air used is supplied from external
            i.convert_to_unit("L/s")
            ventilation_rate.append(
                pd.Series(
                    i.values,
                    index=idx,
                    name=f"{i.header.metadata['type']} ({i.header.unit})",
                )
            )
        if len(ventilation_rate) == 0:
            logging.warning("%s - No ventilation rate available!", self)
            ventilation_rate = pd.Series(
                np.zeros(8760), index=idx, name="Ventilation (L/s)"
            )
        else:
            ventilation_rate = pd.concat(ventilation_rate, axis=1).sum(axis=1)
        fan_energy = (ventilation_rate * self.fan_power).rename("Fans (Wh)")

        # LIGHTING
        lighting_energy = []
        for i in sqlr.data_collections_by_output_name(
            output_name="Zone Lights Electricity Energy"
        ):
            i.convert_to_unit("Wh")
            lighting_energy.append(
                pd.Series(
                    i.values,
                    index=idx,
                    name=f"{i.header.metadata['type']} ({i.header.unit})",
                )
            )
        if len(lighting_energy) == 0:
            logging.warning("%s - No lighting energy available!", self)
            lighting_energy = pd.Series(np.zeros(8760), index=idx, name="Lighting (Wh)")
        else:
            lighting_energy = (
                pd.concat(lighting_energy, axis=1).sum(axis=1).rename("Lighting (Wh)")
            )

        # EQUIPMENT
        equipment_energy = []
        for i in sqlr.data_collections_by_output_name(
            output_name="Zone Electric Equipment Electricity Energy"
        ):
            i.convert_to_unit("Wh")
            equipment_energy.append(
                pd.Series(
                    i.values,
                    index=idx,
                    name=f"{i.header.metadata['type']} ({i.header.unit})",
                )
            )
        if len(equipment_energy) == 0:
            logging.warning("%s- No equipment energy available!", self)
            equipment_energy = pd.Series(
                np.zeros(8760), index=idx, name="Electric Equipment (Wh)"
            )
        else:
            equipment_energy = (
                pd.concat(equipment_energy, axis=1)
                .sum(axis=1)
                .rename("Electric Equipment (Wh)")
            )

        # LIFTS
        lift_energy = typical_lift_energy(
            building_type=self.building_type,
            occupancy_schedule=self.occupancy_schedule,
            target_building_height=self.average_building_height,
            target_n_floors=self.average_num_floors,
        )

        # PUMPS
        # TODO - check that the Pump power calc is correct
        chw_delta_t = 6  # K
        chw_cp = 4.18  # kJ/kgK
        chw_flow_ls = cooling_energy / chw_cp / chw_delta_t
        pump_energy = (chw_flow_ls * self.pump_power).rename("Pumps (Wh)")

        df_energy = pd.concat(
            [
                cooling_energy,
                heating_energy,
                fan_energy,
                service_hot_water_energy,
                lighting_energy,
                equipment_energy,
                pump_energy,
                lift_energy,
            ],
            axis=1,
        )

        df_energy = (
            df_energy[abs(df_energy.mean()).sort_values(ascending=False).index]
            * self.number_of_buildings
        )

        df_energy.to_csv(filename)

        return df_energy

    def energy_consumption_normalised(self) -> pd.DataFrame:
        """The hourly energy consumed by the building typology simulated,
        normalised by floor area."""

        if not self._results_exist:
            self.sql()

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.csv"
        )

        if filename.exists():
            logging.info("%s - Reloading energy consumption normalised", self)
            return pd.read_csv(
                filename,
                header=0,
                parse_dates=True,
                index_col=0,
            )

        logging.info("%s - Processing energy consumption normalised", self)
        df_energy_normalised = self.energy_consumption() / (
            self.number_of_buildings * self.model().floor_area
        )
        df_energy_normalised.columns = [
            i.replace("Wh", "Wh/m2") for i in df_energy_normalised.columns
        ]
        df_energy_normalised.to_csv(filename)

        return df_energy_normalised

    def energy_demand(self) -> pd.DataFrame:
        """The hourly energy demanded by the building typology simulated -
        including system efficiencies giving resultant DEMAND for energy."""

        if not self._results_exist:
            self.sql()

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.csv"
        )

        if filename.exists():
            logging.info("%s - Reloading energy demand results", self)
            return pd.read_csv(
                filename,
                header=0,
                parse_dates=True,
                index_col=0,
            )

        logging.info("%s - Processing energy demand results", self)
        energy_consumption = self.energy_consumption()
        energy_demand = energy_consumption.copy()

        # apply cop/eer to different energy types
        energy_demand["Heating (Wh)"] = (
            energy_consumption["Heating (Wh)"] / self.heating_cop
        )
        energy_demand["Service Hot Water (Wh)"] = (
            energy_consumption["Service Hot Water (Wh)"] / self.heating_cop
        )
        energy_demand["Cooling (Wh)"] = (
            energy_consumption["Cooling (Wh)"] / self.cooling_eer
        )

        # sort again by highest energy demand
        energy_demand = energy_demand[
            abs(energy_demand.mean()).sort_values(ascending=False).index
        ]

        # write to disk for reload later
        energy_demand.to_csv(filename)

        return energy_demand

    def energy_demand_normalised(self) -> pd.DataFrame:
        """The hourly energy demanded by the building typology simulated -
        including system efficiencies giving resultant DEMAND for energy,
        normalised by floor area."""

        if not self._results_exist:
            self.sql()

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.csv"
        )

        if filename.exists():
            logging.info("%s - Reloading energy demand normalised", self)
            return pd.read_csv(
                filename,
                header=0,
                parse_dates=True,
                index_col=0,
            )

        logging.info("%s - Processing energy demand normalised", self)
        df_energy_normalised = self.energy_demand() / (
            self.number_of_buildings * self.model().floor_area
        )
        df_energy_normalised.columns = [
            i.replace("Wh", "Wh/m2") for i in df_energy_normalised.columns
        ]
        df_energy_normalised.to_csv(filename)

        return df_energy_normalised

    def room_conditions(self) -> pd.DataFrame:
        """The typical room conditions for the building typology simulated."""

        if not self._results_exist:
            self.sql()

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.csv"
        )

        if filename.exists():
            logging.info("%s - Reloading room conditions", self)
            return pd.read_csv(filename, header=0, parse_dates=True, index_col=0)

        logging.info("%s - Processing room conditions", self)
        sqlr = SQLiteResult(file_path=self._sql_file.as_posix())
        idx = self.default_index

        # ROOM CONDITIONS
        all_conditions = []
        keys = []
        i: HourlyContinuousCollection = None
        for varr in [
            "Zone Mean Air Temperature",
            "Zone Air Relative Humidity",
            "Zone Mean Radiant Temperature",
        ]:
            avg_conditions = []
            for i in sqlr.data_collections_by_output_name(output_name=varr):
                avg_conditions.append(
                    pd.Series(
                        i.values,
                        index=idx,
                    )
                )
            all_conditions.append(pd.concat(avg_conditions, axis=1).mean(axis=1))
            keys.append(f"{i.header.metadata['type']} ({i.header.unit})")

        df_conditions = pd.concat(all_conditions, axis=1, keys=keys)

        # get weather conditions also
        epw = self.epw
        df_conditions["Outdoor Air Dry Bulb Temperature (C)"] = (
            epw.dry_bulb_temperature.values
        )
        df_conditions["Outdoor Air Relative Humidity (%)"] = (
            epw.relative_humidity.values
        )

        # get setpoints also
        setpoint = self.model().properties.energy.program_types[0].setpoint
        df_conditions["Humidifying Setpoint (%)"] = (
            setpoint.humidifying_schedule.data_collection.values
        )
        df_conditions["Dehumidifying Setpoint (%)"] = (
            setpoint.dehumidifying_schedule.data_collection.values
        )
        df_conditions["Heating Setpoint (C)"] = (
            setpoint.heating_schedule.data_collection.values
        )
        df_conditions["Cooling Setpoint (C)"] = (
            setpoint.cooling_schedule.data_collection.values
        )

        df_conditions.to_csv(filename)

        return df_conditions

    def eui(self, unit="Wh/m2", datatype: str = "demand") -> pd.Series:
        """Get the energy use intensity for the building typology simulated."""

        if datatype not in ["demand", "consumption"]:
            raise ValueError(
                f"Datatype must be 'demand' or 'consumption', not {datatype}"
            )

        if unit not in ["Wh/m2", "kWh/m2", "MWh/m2"]:
            raise ValueError(f"Unit must be 'Wh/m2', 'kWh/m2' or 'MWh/m2', not {unit}")

        if not self._results_exist:
            self.sql()

        data = convert_dataframe(
            getattr(self, f"energy_{datatype}_normalised")().sum(axis=0).T.to_frame().T,
            source_unit="Wh/m2",
            target_unit=unit,
            remove_unit=False,
        ).T.squeeze()

        data.index = [i[:-1] + "/year)" for i in data.index]

        return data

    # endregion

    # region: Figures

    # region: Energy consumption

    def energy_consumption_monthly(self, unit: str = "kWh") -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        # get data and convert to requisite unit
        data = convert_dataframe(
            self.energy_consumption().resample("MS").sum(), "Wh", unit, remove_unit=True
        )

        # PLOT!!!
        plot_monthly_stacked_bar(df=data, ax=ax)

        # format
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel(f"Energy consumption ({unit})")

        ax.set_title(f"{self}\nEnergy consumption")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def energy_consumption_monthly_normalised(self) -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        data = convert_dataframe(
            self.energy_consumption_normalised().resample("MS").sum(),
            "Wh/m2",
            "Wh/m2",
            remove_unit=True,
        )

        # PLOT!!!
        plot_monthly_stacked_bar(df=data, ax=ax)

        # format
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Energy use intensity (Wh/m$^2$)")

        ax.set_title(f"{self}\nEnergy consumption (area normalised)")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def energy_consumption_diurnal(self, unit: str = "kWh") -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        data = convert_dataframe(
            self.energy_consumption(), "Wh", unit, remove_unit=True
        )

        plot_diurnal(data=data, ax=ax)

        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel(f"Energy ({unit})")
        ax.set_title(f"{self}\nTypical diurnal energy consumption")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def energy_consumption_diurnal_normalised(self) -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        data = convert_dataframe(
            self.energy_consumption_normalised(), "Wh/m2", "Wh/m2", remove_unit=True
        )

        plot_diurnal(data=data, ax=ax)

        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Energy use intensity (Wh/m$^2$)")
        ax.set_title(f"{self}\nTypical diurnal energy consumption (area normalised)")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def energy_consumption_pie(self, unit: str = "kWh") -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SQUARE)

        data = convert_dataframe(
            self.energy_consumption().sum().to_frame().T, "Wh", unit, remove_unit=True
        ).T.squeeze()

        plot_pie(series=data, ax=ax)

        ax.set_title(f"{self}\nAnnual energy consumption ({unit})")

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    # endregion

    # region: Energy demand

    def energy_demand_monthly(self, unit: str = "kWh") -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
        # get data and convert to requisite unit
        data = convert_dataframe(
            self.energy_demand().resample("MS").sum(), "Wh", unit, remove_unit=True
        )

        # PLOT!!!
        plot_monthly_stacked_bar(df=data, ax=ax)

        # format
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel(f"Energy demand ({unit})")

        ax.set_title(f"{self}\nEnergy demand")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def energy_demand_monthly_normalised(self) -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        data = convert_dataframe(
            self.energy_demand_normalised().resample("MS").sum(),
            "Wh/m2",
            "Wh/m2",
            remove_unit=True,
        )

        plot_monthly_stacked_bar(df=data, ax=ax)

        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Energy use intensity (Wh/m$^2$)")

        ax.set_title(f"{self}\nEnergy demand (area normalised)")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def energy_demand_diurnal(self, unit: str = "kWh") -> Figure:
        """_"""

        data = convert_dataframe(self.energy_demand(), "Wh", unit, remove_unit=True)

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        plot_diurnal(data=data, ax=ax)

        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel(f"Energy ({unit})")
        ax.set_title(f"{self}\nTypical diurnal energy demand")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def energy_demand_diurnal_normalised(self) -> Figure:
        """_"""

        data = convert_dataframe(
            self.energy_demand_normalised(), "Wh/m2", "Wh/m2", remove_unit=True
        )

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        plot_diurnal(data=data, ax=ax)

        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Energy use intensity (Wh/m$^2$)")
        ax.set_title(f"{self}\nTypical diurnal energy demand (area normalised)")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def energy_demand_pie(self, unit: str = "kWh") -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SQUARE)

        data = convert_dataframe(
            self.energy_demand().sum().to_frame().T, "Wh", unit, remove_unit=True
        ).T.squeeze()

        plot_pie(series=data, ax=ax)

        ax.set_title(f"{self}\nAnnual energy demand ({unit})")

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    # endregion

    # region: Thermal load balance

    def thermal_load_balance_monthly(self, unit: str = "MWh") -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        # get data and convert to requisite unit
        data = convert_dataframe(
            self.thermal_load_balance().resample("MS").sum(),
            "Wh",
            unit,
            remove_unit=True,
        )

        # PLOT!!!
        plot_monthly_stacked_bar(df=data, ax=ax)

        # format
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel(f"Thermal load ({unit})")

        ax.set_title(f"{self}\nThermal load balance")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def thermal_load_balance_monthly_normalised(self) -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        data = convert_dataframe(
            self.thermal_load_balance_normalised().resample("MS").mean(),
            "Wh/m2",
            "Wh/m2",
            remove_unit=True,
        )

        # PLOT!!!
        plot_monthly_stacked_bar(df=data, ax=ax)

        # format
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Thermal load (Wh/m$^2$)")

        ax.set_title(f"{self}\nThermal load balance (area normalised)")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def plot_thermal_load_balance_diurnal(self, unit: str = "kWh") -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        data = self.thermal_load_balance()
        data.columns = [i.split(" (")[0] for i in data.columns]

        if unit not in ["Wh", "kWh", "MWh"]:
            raise ValueError(f"Unit must be 'Wh', 'kWh' or 'MWh' not {unit}")

        if unit == "kWh":
            data = data / 1000
            y_formatter = "{x:,.0f}"
        elif unit == "MWh":
            data = data / 1000000
            y_formatter = "{x:,.1f}"
        else:
            y_formatter = "{x:,.0f}"

        plot_diurnal(data=data, ax=ax)

        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(y_formatter))
        ax.set_ylabel(f"Thermal load ({unit})")
        ax.set_title(f"{self}\nTypical diurnal thermal load balance")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def plot_thermal_load_balance_diurnal_normalised(self) -> Figure:
        """_"""

        data = convert_dataframe(
            self.thermal_load_balance_normalised(), "Wh/m2", "Wh/m2", remove_unit=True
        )

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        plot_diurnal(data=data, ax=ax)

        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Thermal load (Wh/m$^2$)")
        ax.set_title(f"{self}\nTypical diurnal thermal load balance (area normalised)")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    # endregion

    # region: Room conditions

    def plot_temperature_diurnal(self) -> Figure:
        """_"""
        data = self.room_conditions()[
            [
                "Zone Mean Air Temperature (C)",
                "Zone Mean Radiant Temperature (C)",
                "Outdoor Air Dry Bulb Temperature (C)",
                "Heating Setpoint (C)",
                "Cooling Setpoint (C)",
            ]
        ]
        data.columns = [i.split(" (")[0] for i in data.columns]

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        plot_diurnal(data=data, ax=ax)

        ax.set_ylabel("Temperature (C)")
        ax.set_title(f"{self}\nTypical diurnal temperature")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def plot_humidity_diurnal(self) -> Figure:
        """_"""
        data = self.room_conditions()[
            [
                "Zone Air Relative Humidity (%)",
                "Outdoor Air Relative Humidity (%)",
                "Humidifying Setpoint (%)",
                "Dehumidifying Setpoint (%)",
            ]
        ]
        data.columns = [i.split(" (")[0] for i in data.columns]

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        plot_diurnal(data=data, ax=ax)

        ax.set_ylabel("Relative Humidity (%)")
        ax.set_title(f"{self}\nTypical diurnal relative humidity")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def heating_cooling_diurnal(self) -> Figure:
        """_"""
        data = self.energy_consumption()[
            [
                "Cooling (Wh)",
                "Heating (Wh)",
            ]
        ]
        data.columns = [i.split(" (")[0] for i in data.columns]

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        plot_diurnal(data=data, ax=ax)

        ax.set_ylabel("Energy (%)")
        ax.set_title(f"{self}\nTypical diurnal heating/cooling energy demand")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    # endregion

    # region: Load duration curves

    def load_duration_curve(self, unit: str = "kWh") -> Figure:
        """_"""
        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        data = convert_dataframe(
            self.energy_demand()[
                ["Cooling (Wh)", "Heating (Wh)", "Service Hot Water (Wh)"]
            ],
            "Wh",
            unit,
            remove_unit=True,
        )

        plot_duration_curve(data=data, ax=ax, bins=101)
        ax.set_ylabel(unit)
        ax.set_title(f"{self}\nLoad duration curve")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    def load_duration_curve_normalised(self, unit: str = "kWh/m2") -> Figure:
        """_"""

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        data = convert_dataframe(
            self.energy_demand_normalised()[
                ["Cooling (Wh/m2)", "Heating (Wh/m2)", "Service Hot Water (Wh/m2)"]
            ],
            "Wh/m2",
            unit,
            remove_unit=True,
        )

        plot_duration_curve(data=data, ax=ax, bins=101)
        ax.set_ylabel(unit)
        ax.set_title(f"{self}\nLoad duration curve (area normalised)")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    # endregion

    # region: timeseries

    def heating_cooling_timeseries(self, unit: str = "kWh") -> Figure:

        filename = (
            self.simulation_directory
            / f"{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)

        data = convert_dataframe(
            self.energy_demand()[["Cooling (Wh)", "Heating (Wh)"]],
            "Wh",
            unit,
            remove_unit=True,
        )
        plot_heating_cooling_series(
            heating=data["Heating"], cooling=data["Cooling"], ax=ax, kind="line"
        )
        ax.set_ylabel(f"Energy demand ({unit})")
        ax.set_title(f"{self}\nAnnual hourly energy demand")
        plt.tight_layout()

        fig.savefig(filename, dpi=DPI, transparent=True)

        return fig

    # endregion

    # endregion

    def run_all(self) -> None:
        """_"""

        try:
            if not self._results_exist:
                self.sql()

            self.thermal_load_balance_monthly()
            self.thermal_load_balance_monthly_normalised()
            self.plot_thermal_load_balance_diurnal()
            self.plot_thermal_load_balance_diurnal_normalised()
            self.energy_consumption_monthly()
            self.energy_consumption_monthly_normalised()
            self.energy_consumption_diurnal()
            self.energy_consumption_diurnal_normalised()
            self.energy_consumption_pie()
            self.energy_demand_monthly()
            self.energy_demand_monthly_normalised()
            self.energy_demand_diurnal()
            self.energy_demand_diurnal_normalised()
            self.energy_demand_pie()
            self.plot_temperature_diurnal()
            self.plot_humidity_diurnal()
            self.load_duration_curve()
            self.load_duration_curve_normalised()

        except Exception as e:
            logging.error("%s - %s", self, e)

        plt.close("all")


def run_multiple(objects: list[MPED]) -> list[MPED]:
    """Run a set of MPED objects in parallel.

    Args:
        objects (list[MPED]): A list of MPED objects.

    Returns:
        list[MPED]: A list of MPED objects.
    """

    # validation
    if not isinstance(objects, (list, tuple)):
        raise TypeError(f"objects must be a list or tuple not {type(objects)}")

    if len(objects) == 0:
        raise ValueError(f"objects must contain at least one {MPED.__name__} object")

    # TODO - type validation for all objects! For some reason this isn't straightforward

    if len(set([str(obj) for obj in objects])) != len(objects):
        raise ValueError("All objects must be uniquely named")

    # check that all objects have the same project identifier
    if len(set(obj.project_identifier for obj in objects)) != 1:
        raise ValueError(
            "All objects must have the same project identifier ... for now"
        )

    # run simulations
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                obj.run_all,
            )
            for obj in objects
        ]
        _ = [future.result() for future in concurrent.futures.as_completed(futures)]

    # create case summary/ies
    cases = {}
    for obj in objects:
        if f"{obj.project_identifier}::{obj.case_identifier}" not in cases:
            cases[f"{obj.project_identifier}::{obj.case_identifier}"] = [obj]
        else:
            cases[f"{obj.project_identifier}::{obj.case_identifier}"].append(obj)

    for case, objs in cases.items():
        logging.info("Combining results and summarising %s", case)
        summarise_case(objs)

    return objects


def summarise_case(objects: list[MPED], unit: str = "MWh") -> None:
    """Given a list of MPED objects, combine and summarise the results.

    Args:
        objects (list[MPED]): A list of MPED objects.
        unit (str, optional): The unit to use. Defaults to "MWh".

    Raises:
        ValueError: If the unit is not "Wh", "kWh" or "MWh".
    """

    # set constants
    epw_file = objects[0].epw_file
    case_identifier = objects[0].case_identifier
    project_identifier = objects[0].project_identifier
    case_directory = objects[0].case_directory
    class_name = objects[0].__class__.__name__
    full_name = f"{project_identifier}::{case_identifier}"

    # validate inputs
    for obj in objects:
        if obj.epw_file != epw_file:
            raise ValueError(
                f"All objects to summarise must share the same epw_file. ({obj.epw_file} != {epw_file})"
            )
        if obj.case_identifier != case_identifier:
            raise ValueError(
                f"All objects to summarise must share the same case_identifier. ({obj.case_identifier} != {case_identifier})"
            )
        if obj.project_identifier != project_identifier:
            raise ValueError(
                f"All objects to summarise must share the same project_identifier. ({obj.project_identifier} != {project_identifier})"
            )

    # region: Filepaths
    energy_consumption_file = case_directory / f"{class_name}_energy_consumption.csv"
    energy_consumption_file_normalised = (
        case_directory / f"{class_name}_energy_consumption_normalised.csv"
    )
    thermal_load_balance_file = (
        case_directory / f"{class_name}_thermal_load_balance.csv"
    )
    thermal_load_balance_normalised_file = (
        case_directory / f"{class_name}_thermal_load_balance_normalised.csv"
    )
    energy_consumption_pie_figure = (
        case_directory / f"{class_name}_energy_consumption_pie.png"
    )

    energy_consumption_distributed_pie_figure = (
        case_directory / f"{class_name}_energy_consumption_distributed_pie.png"
    )
    # endregion

    # region: Combine results sets

    areas = []
    for obj in objects:
        areas.append(obj.total_area)
    total_area = sum(areas)

    # load results from each of the objects
    if energy_consumption_file.exists():
        energy_consumption = pd.read_csv(
            energy_consumption_file,
            header=0,
            index_col=0,
            parse_dates=True,
        )
    else:
        energy_consumptions = []
        for obj in objects:
            energy_consumptions.append(obj.energy_consumption())
        energy_consumption = pd.concat(energy_consumptions, axis=1)
        energy_consumption = (
            energy_consumption.T.groupby(energy_consumption.columns).sum().T
        )
        energy_consumption = energy_consumption[
            energy_consumption.mean().sort_values(ascending=False).index
        ]
        energy_consumption.to_csv(energy_consumption_file)

    if energy_consumption_file_normalised.exists():
        energy_consumption_normalised = pd.read_csv(
            energy_consumption_file_normalised,
            header=0,
            index_col=0,
            parse_dates=True,
        )
    else:
        energy_consumption_normalised = energy_consumption / total_area
        energy_consumption_normalised.columns = [
            i.replace("Wh", "Wh/m2") for i in energy_consumption_normalised.columns
        ]
        energy_consumption_normalised.to_csv(energy_consumption_file_normalised)

    if thermal_load_balance_file.exists():
        thermal_load_balance = pd.read_csv(
            thermal_load_balance_file,
            header=0,
            index_col=0,
            parse_dates=True,
        )
    else:
        thermal_load_balances = []
        for obj in objects:
            thermal_load_balances.append(obj.thermal_load_balance())

        thermal_load_balance = pd.concat(thermal_load_balances, axis=1)
        thermal_load_balance = (
            thermal_load_balance.T.groupby(thermal_load_balance.columns).sum().T
        )
        thermal_load_balance = thermal_load_balance[
            abs(thermal_load_balance).mean().sort_values(ascending=False).index
        ]
        thermal_load_balance.to_csv(thermal_load_balance_file)

    if thermal_load_balance_normalised_file.exists():
        thermal_load_balance_normalised = pd.read_csv(
            thermal_load_balance_normalised_file,
            header=0,
            index_col=0,
            parse_dates=True,
        )
    else:
        thermal_load_balance_normalised = thermal_load_balance / total_area
        thermal_load_balance_normalised.columns = [
            i.replace("Wh", "Wh/m2") for i in thermal_load_balance_normalised.columns
        ]
        thermal_load_balance_normalised.to_csv(thermal_load_balance_normalised_file)

    # endregion

    # region convert to target unit
    energy_consumption = convert_dataframe(
        energy_consumption, "Wh", unit, remove_unit=True
    )
    energy_consumption_normalised = convert_dataframe(
        energy_consumption_normalised, "Wh/m2", "Wh/m2", remove_unit=True
    )
    thermal_load_balance = convert_dataframe(
        thermal_load_balance, "Wh", unit, remove_unit=True
    )
    thermal_load_balance_normalised = convert_dataframe(
        thermal_load_balance_normalised, "Wh/m2", "Wh/m2", remove_unit=True
    )

    # region: Visualise

    # energy consumption monthly
    energy_consumption_monthly_figure = (
        case_directory / f"{class_name}_energy_consumption_monthly.png"
    )
    if not energy_consumption_monthly_figure.exists():
        data = energy_consumption.resample("MS").sum()
        data.columns = [i.split(" (")[0] for i in data.columns]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
        plot_monthly_stacked_bar(df=data, ax=ax)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel(f"Energy ({unit})")
        ax.set_title(f"{full_name}\nEnergy consumption")
        plt.tight_layout()
        fig.savefig(energy_consumption_monthly_figure, dpi=DPI, transparent=True)
        plt.close(fig)

    # energy consumption normalised monthly
    energy_consumption_monthly_figure_normalised = (
        case_directory / f"{class_name}_energy_consumption_monthly_normalised.png"
    )
    if not energy_consumption_monthly_figure_normalised.exists():
        data = energy_consumption_normalised.resample("MS").mean()
        data.columns = [i.split(" (")[0] for i in data.columns]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
        plot_monthly_stacked_bar(df=data, ax=ax)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Energy use intensity (Wh/m$^2$)")
        ax.set_title(f"{full_name}\nEnergy consumption (area normalised)")
        plt.tight_layout()
        fig.savefig(
            energy_consumption_monthly_figure_normalised, dpi=DPI, transparent=True
        )
        plt.close(fig)

    # energy consumption diurnal
    energy_consumption_diurnal_figure = (
        case_directory / f"{class_name}_energy_consumption_diurnal.png"
    )
    if not energy_consumption_diurnal_figure.exists():
        data = energy_consumption.copy()
        data.columns = [i.split(" (")[0] for i in data.columns]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
        plot_diurnal(data=data, ax=ax)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel(f"Energy ({unit})")
        ax.set_title(f"{full_name}\nTypical diurnal energy consumption")
        plt.tight_layout()
        fig.savefig(energy_consumption_diurnal_figure, dpi=DPI, transparent=True)
        plt.close(fig)

    # energy consumption normalised diurnal
    energy_consumption_diurnal_figure_normalised = (
        case_directory / f"{class_name}_energy_consumption_diurnal_normalised.png"
    )
    if not energy_consumption_diurnal_figure_normalised.exists():
        data = energy_consumption_normalised.copy()
        data.columns = [i.split(" (")[0] for i in data.columns]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
        plot_diurnal(data=data, ax=ax)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Energy use intensity (Wh/m$^2$)")
        ax.set_title(
            f"{full_name}\nTypical diurnal energy consumption (area normalised)"
        )
        plt.tight_layout()
        fig.savefig(
            energy_consumption_diurnal_figure_normalised, dpi=DPI, transparent=True
        )
        plt.close(fig)

    # thermal load balance monthly
    thermal_load_balance_monthly_figure = (
        case_directory / f"{class_name}_thermal_load_balance_monthly.png"
    )
    if not thermal_load_balance_monthly_figure.exists():
        data = thermal_load_balance.resample("MS").sum()
        data.columns = [i.split(" (")[0] for i in data.columns]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
        plot_monthly_stacked_bar(df=data, ax=ax)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel(f"Thermal load ({unit})")
        ax.set_title(f"{full_name}\nThermal load balance")
        plt.tight_layout()
        fig.savefig(thermal_load_balance_monthly_figure, dpi=DPI, transparent=True)
        plt.close(fig)

    # thermal load balance normalised monthly
    thermal_load_balance_monthly_figure_normalised = (
        case_directory / f"{class_name}_thermal_load_balance_monthly_normalised.png"
    )
    if not thermal_load_balance_monthly_figure_normalised.exists():
        data = thermal_load_balance_normalised.resample("MS").mean()
        data.columns = [i.split(" (")[0] for i in data.columns]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
        plot_monthly_stacked_bar(df=data, ax=ax)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Thermal load (Wh/m$^2$)")
        ax.set_title(f"{full_name}\nThermal load balance (area normalised)")
        plt.tight_layout()
        fig.savefig(
            thermal_load_balance_monthly_figure_normalised, dpi=DPI, transparent=True
        )
        plt.close(fig)

    # thermal load balance diurnal
    thermal_load_balance_diurnal_figure = (
        case_directory / f"{class_name}_thermal_load_balance_diurnal_monthly.png"
    )
    if not thermal_load_balance_diurnal_figure.exists():
        data = thermal_load_balance.copy()
        data.columns = [i.split(" (")[0] for i in data.columns]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
        plot_diurnal(data=data, ax=ax)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel(f"Thermal load ({unit})")
        ax.set_title(f"{full_name}\nTypical diurnal thermal load balance")
        plt.tight_layout()
        fig.savefig(thermal_load_balance_diurnal_figure, dpi=DPI, transparent=True)
        plt.close(fig)

    # thermal load balance normalised diurnal
    thermal_load_balance_diurnal_figure_normalised = (
        case_directory
        / f"{class_name}_thermal_load_balance_diurnal_monthly_normalised.png"
    )
    if not thermal_load_balance_diurnal_figure_normalised.exists():

        data = thermal_load_balance_normalised.copy()
        data.columns = [i.split(" (")[0] for i in data.columns]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
        plot_diurnal(data=data, ax=ax)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Thermal load (Wh/m$^2$)")
        ax.set_title(
            f"{full_name}\nTypical diurnal thermal load balance (area normalised)"
        )
        plt.tight_layout()
        fig.savefig(
            thermal_load_balance_diurnal_figure_normalised, dpi=DPI, transparent=True
        )
        plt.close(fig)

    # energy consumption pie
    if not energy_consumption_distributed_pie_figure.exists():
        data = energy_consumption.sum()
        data.index = [i.split(" (")[0] for i in data.index]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SQUARE)
        plot_pie(series=data, ax=ax)
        ax.set_title(f"{full_name}\nAnnual energy consumption ({unit})")
        plt.tight_layout()
        fig.savefig(energy_consumption_pie_figure, dpi=DPI, transparent=True)
        plt.close(fig)

        # energy consumption distributed pie
        cases = []
        keys = []
        for obj in objects:
            cases.append(obj.energy_consumption())
            bdg = obj.building_identifier
            keys.append(bdg)
        ap = AnalysisPeriod(st_hour=0, end_hour=23)
        temp = pd.concat(cases, axis=1, keys=keys)

        # apply time filter
        temp = temp.loc[pd.to_datetime(ap.datetimes)]
        # get sum
        temp = temp.sum(axis=0).unstack()
        temp = temp.loc[temp.sum(axis=1).sort_values(ascending=False).index]
        size = 0.33
        vals = temp.values
        outer_colors = ["grey"]
        inner_colors = [FORMATTING.color[i.split(" (")[0]] for i in temp.columns]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        inner_wedge, _ = ax.pie(
            vals.flatten(),
            radius=1 - size,
            wedgeprops=dict(width=size, edgecolor="w", linewidth=0),
            startangle=90,
            counterclock=False,
            colors=inner_colors,
        )
        _, outer_txt = ax.pie(
            vals.sum(axis=1),
            radius=1,
            wedgeprops=dict(width=size, edgecolor="w", linewidth=0.5),
            startangle=90,
            counterclock=False,
            colors=["grey"],
        )

        for n, (name, vals) in enumerate(temp.iterrows()):
            total_prop = vals.sum() / temp.sum().sum()
            if total_prop > 0.02:
                outer_txt[n].set_text(f"{name}\n{total_prop:0.1%}")
                outer_txt[n].set_fontsize("xx-small")
        ax.legend(
            inner_wedge[: len(temp.columns)],
            [i.split(" (")[0] for i in temp.columns],
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncols=3,
        )
        ax.set_title(
            f"Masterplan energy consumption\n{ap}\n{temp.sum().sum() / 1000000:,.0f}MWh"
        )
        plt.tight_layout()
        fig.savefig(
            energy_consumption_distributed_pie_figure, dpi=DPI, transparent=True
        )
        plt.close(fig)

    # load duration
    load_duration_curve_figure = (
        case_directory / f"{class_name}_load_duration_curve.png"
    )
    if not load_duration_curve_figure.exists():
        data = energy_consumption[["Cooling", "Heating", "Service Hot Water"]]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
        plot_duration_curve(data=data, ax=ax, bins=101)
        ax.set_title(f"{full_name}\nLoad duration curve")
        plt.tight_layout()
        fig.savefig(load_duration_curve_figure, dpi=DPI, transparent=True)
        plt.close(fig)
        return ax

    # load duration normalised
    load_duration_curve_figure_normalised = (
        case_directory / f"{class_name}_load_duration_curve_normalised.png"
    )
    if not load_duration_curve_figure_normalised.exists():
        # print(load_duration_curve_figure_normalised)
        data = energy_consumption_normalised[
            ["Cooling", "Heating", "Service Hot Water"]
        ]
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
        plot_duration_curve(data=data, ax=ax, bins=101)
        ax.set_title(f"{full_name}\nLoad duration curve (area normalised)")
        plt.tight_layout()
        fig.savefig(load_duration_curve_figure_normalised, dpi=DPI, transparent=True)
        plt.close(fig)
        return ax

    # endregion

    return None


def summarise_project(objects: list[MPED], unit: str = "MWh") -> None:
    """Given a list of MPED objects, summarise the results for the project.

    This method uses the MPED.project identifier and MPED.case_identifier to
    group the results.

    Args:
        objects (list[MPED]): A list of MPED objects.
        unit (str, optional): The unit to use for the results. Defaults to "MWh".

    Raises:
        TypeError: If objects is not a list or tuple.
        ValueError: If the unit is not 'Wh', 'kWh' or 'MWh'.
    """

    if not isinstance(objects, (list, tuple)):
        raise TypeError(f"objects must be a list or tuple not {type(objects)}")

    # set constants
    project_identifier = objects[0].project_identifier
    project_directory = objects[0].project_directory
    class_name = objects[0].__class__.__name__
    epw_file = objects[0].epw_file

    for obj in objects:
        if obj.epw_file != epw_file:
            raise ValueError("All objects to summarise must share the same epw_file")
        if obj.project_identifier != project_identifier:
            raise ValueError(
                "All objects to summarise must share the same project_identifier"
            )

    cases: dict[str, list[MPED]] = {}
    for obj in objects:
        if obj.case_identifier not in cases:
            cases[obj.case_identifier] = [obj]
        else:
            cases[obj.case_identifier].append(obj)

    # set filepaths

    energy_consumption_comparison_figure = (
        project_directory / f"{class_name}_energy_consumption_comparison.png"
    )
    peak_load_comparison_figure = (
        project_directory / f"{class_name}_peak_load_comparison.png"
    )

    # load results from disk
    energy_consumption = {}
    for case, objs in cases.items():
        total_area = 0
        for obj in objs:
            if case not in energy_consumption:
                energy_consumption[case] = [
                    convert_dataframe(
                        obj.energy_consumption(), "Wh", unit, remove_unit=True
                    )
                ]
            else:
                energy_consumption[case].append(
                    convert_dataframe(
                        obj.energy_consumption(), "Wh", unit, remove_unit=True
                    )
                )
        total_area += obj.total_area

    energy_consumption = pd.concat(
        {k: sum(v) for k, v in energy_consumption.items()}, axis=1
    )

    if not energy_consumption_comparison_figure.exists():
        data = energy_consumption.sum(axis=0).unstack()
        data = data[data.sum(axis=0).sort_values(ascending=False).index]
        data.columns = [i.split(" (")[0] for i in data.columns]
        colors = [FORMATTING.color[i] for i in data.columns]

        fig, ax = plt.subplots(1, 1, figsize=(3 * len(data.index), 4))
        data.plot(ax=ax, kind="bar", stacked=True, color=colors)
        ylims = ax.get_ylim()
        for rect in ax.patches:
            y_value = rect.get_y() + rect.get_height() / 2
            x_value = rect.get_x() + rect.get_width() / 2
            fc = rect.get_facecolor()
            actual_value = rect.get_height()
            if abs(actual_value) / (max(ylims) - min(ylims)) > 0.025:
                if abs(actual_value) < 100:
                    val = f"{actual_value:,.1f}"
                else:
                    val = f"{actual_value:,.0f}"
                # Create annotation
                ax.text(
                    x_value,
                    y_value,
                    val,
                    ha="center",
                    va="center",
                    fontsize="xx-small",
                    color=contrasting_color(fc),
                    alpha=0.75,
                    zorder=8,
                )
        for n, col in enumerate(data.T):
            perc = 1 - (data.T[col].sum() / data.T.sum(axis=0).max())
            if perc != 0:
                ax.text(
                    n,
                    data.T[col].sum(),
                    f"{perc:0.1%} reduction",
                    ha="center",
                    va="bottom",
                )
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        ax.tick_params(axis="x", labelrotation=0)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel(f"Energy ({unit})")
        ax.set_title(f"{project_identifier}\nAnnual energy consumption")
        plt.tight_layout()
        fig.savefig(energy_consumption_comparison_figure, transparent=True, dpi=DPI)
        plt.close(fig)
