# region: IMPORTS
# pylint: disable=E0401

import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from honeybee.boundarycondition import Ground, Outdoors
from honeybee.config import folders as hb_folders
from honeybee.model import Face, Model, Room
from honeybee.typing import valid_string
from honeybee_energy.programtype import ProgramType
from honeybee_energy.result.loadbalance import LoadBalance, SQLiteResult
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.simulation.parameter import (RunPeriod, ShadowCalculation,
                                                  SimulationControl,
                                                  SimulationOutput,
                                                  SimulationParameter,
                                                  SizingParameter)
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from pydantic import BaseModel, Field, root_validator
from sklearn.linear_model import LinearRegression

from .config import DATA_PATH, INDEX, logger
from .enums import (EPW, BuildingType, TerrainType, Vintage,
                    default_construction_type)
from .fabric import Fabric
from .form import Form
from .program import Program
from .system import System
from .util import face_orientation, random_id

# pylint: enable=E0401
# endregion: IMPORTS


class Typology(BaseModel):
    """The building typology to be simulated as part of a masterplan energy model."""

    identifier: str = Field(
        description="A unique identifier for the typology.",
        repr=True,
    )
    total_area: float = Field(
        description="The total area of this building typology (m2).",
        gt=0,
        repr=False,
    )
    form: Form = Field(description="The form of the building.", repr=False)
    fabric: Fabric = Field(description="The fabric of the building.", repr=False)
    program: Program = Field(description="The program of the building.", repr=False)
    system: System = Field(description="The system of the building.", repr=False)
    building_type: BuildingType = Field(
        description="The type of building.",
        repr=False,
    )
    vintage: Vintage = Field(
        description="The vintage of the building.",
        repr=False,
    )

    @root_validator(pre=True)
    def validate_atts(cls, values):
        """Validate the attributes, ensuring that systems do not override each other."""

        identifier = values.get("identifier")
        valid_string(identifier)

        return values

    @classmethod
    def random(cls, seed: int = None) -> "Typology":
        """Create a random typology."""

        np.random.seed(seed)

        return Typology(
            identifier=random_id(seed),
            total_area=np.random.uniform(0.1, 1000),
            building_type=np.random.choice(list(BuildingType)),
            vintage=np.random.choice(list(Vintage)),
            form=Form.random(seed),
            fabric=Fabric.random(seed),
            program=Program.random(seed),
            system=System.random(seed),
        )

    @classmethod
    def from_building_type(
        cls,
        building_type: BuildingType,
        total_area: float,
        epw: EPW,
        vintage: Vintage = Vintage.ASHRAE_901_2019,
        rotation: float = 0,
        terrain: TerrainType = TerrainType.URBAN,
    ) -> "Typology":
        """Create a Typology object from default values for a given BuildingType."""

        return cls(
            identifier=building_type.name,
            total_area=total_area,
            building_type=building_type,
            vintage=vintage,
            form=Form.from_building_type(
                building_type=building_type, rotation=rotation, terrain=terrain
            ),
            fabric=Fabric.from_building_type(
                building_type=building_type, epw=epw, vintage=vintage
            ),
            program=Program.from_building_type(building_type=building_type),
            system=System.from_building_type(
                building_type=building_type, epw=epw, vintage=vintage
            ),
        )

    @classmethod
    def parse_obj_extended(
        cls,
        d: dict,
        replace_null_with_defaults: bool = False,
        building_type: BuildingType = None,
    ) -> "Fabric":
        """Create a Fabric object from an extended dictionary, where directional glazing_ratio is present."""

        # copy to prevent mutation
        d = deepcopy(d)

        if replace_null_with_defaults and (building_type is None):
            raise ValueError(
                "Building type must be provided if populate_with_defaults is True."
            )

        # process
        return cls(
            identifier=d["identifier"],
            building_type=BuildingType(d["building_type"]),
            total_area=d["total_area"],
            vintage=Vintage(str(d["vintage"])),
            form=Form.parse_obj_extended(d),
            fabric=Fabric.parse_obj_extended(d),
            program=Program.parse_obj_extended(
                d=d,
                replace_null_with_defaults=replace_null_with_defaults,
                building_type=building_type,
            ),
            system=System.parse_obj(d),
        )

    @property
    def program_type(self) -> ProgramType:
        """Quick accessor for the program type of the building."""
        return self.program.program_type(self.building_type)

    def profile_table(self) -> pd.DataFrame:
        """Generate a table containing the annual hourly profiles used in the model."""

        return pd.concat(
            [
                self.program.profile_lighting(self.building_type),
                self.program.profile_equipment(self.building_type),
                self.program.profile_infiltration(self.building_type),
                self.program.profile_ventilation(self.building_type),
                self.program.profile_people(self.building_type),
                self.program.profile_heating(self.building_type),
                self.program.profile_cooling(self.building_type),
                self.program.profile_humidifying(self.building_type),
                self.program.profile_dehumidifying(self.building_type),
                self.program.profile_shw(self.building_type),
            ],
            axis=1,
        )

    @property
    def number_of_buildings(self) -> float:
        """_"""
        return self.total_area / (
            self.form.average_footprint_area * self.form.average_num_floors
        )

    def occupancy_schedule(self) -> pd.Series:
        """Get the occupancy schedule for the building type."""

        program = self.program_type

        if program.people is None:
            values = np.zeros(8760)
        else:
            values = program.people.occupancy_schedule.data_collection().values

        return pd.Series(
            values,
            index=pd.to_datetime(AnalysisPeriod().datetimes),
            name="Occupancy",
        )

    def estimate_lift_energy(self) -> pd.Series:
        """Estimate the annual energy consumption of a lift system in kWh,
        for a building of the given height.

        Source:
        For energy demand per year:
            Ang, Jia Hui, et al. 'Comprehensive Energy Consumption of Elevator
            Systems Based on Hybrid Approach of Measurement and Calculation in Low-
            and High-Rise Buildings of Tropical Climate towards Energy Efficiency'.
            Sustainability, vol. 14, no. 8, Apr. 2022, p. 4779. DOI.org (Crossref),
            https://doi.org/10.3390/su14084779.
        For lift usage profile during day:
            Tukia, Toni, et al. 'Modeling the Aggregated Power Consumption of
            Elevators - the New York City Case Study'. Applied Energy, vol. 251,
            Oct. 2019, p. 113356. DOI.org (Crossref),
            https://doi.org/10.1016/j.apenergy.2019.113356.

        Args:
            building_type (BuildingType):
                The type of building to calculate for.
            occupancy_schedule (pd.Series):
                The occupancy schedule for the building.
            target_n_floors (int):
                The number of floors in the building.
            target_building_height (float):
                The height of the building in meters.

        Returns:
            pd.Series:
                The estimated hourly energy consumption of the lift system in kWh.
        """

        building_type = self.building_type
        occupancy_schedule = self.occupancy_schedule()
        target_n_floors = self.form.average_num_floors
        target_building_height = self.form.building_height()

        if not isinstance(occupancy_schedule, pd.Series):
            raise TypeError("Occupancy schedule must be a pandas Series.")

        if not isinstance(target_n_floors, int):
            raise TypeError("Number of floors must be an integer.")

        if not isinstance(target_building_height, (int, float)):
            raise TypeError("Building height must be a number.")

        if target_n_floors < 2:
            return pd.Series(
                np.zeros(8760), index=occupancy_schedule.index, name="Lifts (Wh)"
            )

        if len(occupancy_schedule) != 8760:
            raise ValueError("Occupancy schedule must have 8760 values.")

        if not isinstance(occupancy_schedule.index, pd.DatetimeIndex):
            raise ValueError("Occupancy schedule must have a datetime index.")

        # load the datasets
        usage_profile = pd.read_csv(
            DATA_PATH / "lift_profile.csv",
            header=0,
            index_col=0,
        )
        annual_energy = pd.read_csv(
            DATA_PATH / "lift_energy.csv",
            header=0,
        )

        usage = []
        for wkday in occupancy_schedule.resample("D").mean().index.weekday:
            if wkday in [5, 6]:
                usage += usage_profile.weekend.values.tolist()
            else:
                usage += usage_profile.weekday.values.tolist()
        usage_profile = pd.Series(usage, index=occupancy_schedule.index)

        # get "Office" or "Residential" based on building type
        lift_bdg_type = (
            "Residential" if "ACCOMODATION" in building_type.name else "Office"
        )
        data = annual_energy[annual_energy.BuildingUse == lift_bdg_type][
            ["BuildingHeight_m", "Floors", "AnnualEnergyConsumption_Wh"]
        ]
        model = LinearRegression()
        model.fit(
            data[["BuildingHeight_m", "Floors"]].values,
            data["AnnualEnergyConsumption_Wh"].values,
        )
        # get annual total energy demand value
        annual_energy_kwh = (
            model.predict([[target_building_height, target_n_floors]])[0] / 1000
        )

        # apportion energy across year, based on occupancy level and daily usage profile
        temp = (usage_profile / usage_profile.sum()) * (
            occupancy_schedule / occupancy_schedule.sum()
        )
        temp = temp / temp.sum()
        temp.name = "Lifts (kWh)"
        return temp * annual_energy_kwh

    @property
    def occupants(self) -> pd.Series:
        """_"""
        program = self.program_type

        if program.people is None:
            people_per_area = 0
        else:
            people_per_area = program.people.people_per_area

        return people_per_area * self.occupancy_schedule() * self.total_area

    def model(self, epw: EPW) -> Model:
        """Create a honeybee model for the typology."""

        model = self.form.base_model().duplicate()

        program = self.program.program_type(self.building_type)
        construction_type = default_construction_type(self.building_type)

        constructions = self.fabric.constructions()

        ideal_air = self.system.ideal_air()

        shw = self.system.shw()

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
            for mass in self.fabric.internal_mass(
                room, epw, self.vintage, construction_type
            ):
                room.properties.energy.add_internal_mass(mass)

            # assign programs to rooms
            room.properties.energy.program_type = program

            # assign SHW to rooms
            if room.properties.energy.service_hot_water is not None:
                room.properties.energy.shw = shw

            # add daylight dimming
            if self.system.daylight_dimming:
                room.properties.energy.add_daylight_control_to_center(
                    distance_from_floor=0.8, control_fraction=0.5
                )

            # add system
            room.properties.energy.hvac = ideal_air

        # rename
        model.identifier = self.identifier

        return model

    def simulate(self, epw: EPW, directory: Path) -> Path:
        """Simulate the typology for a given EPW file."""

        # create model to simulate
        model = self.model(epw)

        # validate inputs
        if Path(directory).is_file():
            raise ValueError("Target directory is a file and must be a directory.")

        # create paths for reference
        target_dir = Path(directory) / model.identifier
        openstudio_dir = target_dir / "openstudio"
        sql_file = openstudio_dir / "run" / "eplusout.sql"
        hbjson_file = openstudio_dir / f"{model.identifier}.hbjson"
        osw_file = openstudio_dir / "workflow.osw"
        epw_file = Path(epw.file_path).absolute()
        sim_par_file = openstudio_dir / "simulation_parameters.json"
        config_file = target_dir / "mped_config.json"

        # reload old results if they exist
        if config_file.exists():
            if Typology.parse_file(config_file) == self:
                if sql_file.exists():
                    if osw_file.exists():
                        with open(osw_file, "r", encoding="utf-8") as fp:
                            old_epw_file = Path(json.load(fp)["weather_file"])
                        if old_epw_file.name == epw_file.name:
                            return sql_file
        logger.info(f"{self.identifier} - Simulating results")

        # remove old files in target directory just in case
        for f in target_dir.glob("*"):
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                for ff in f.glob("*"):
                    ff.unlink()
                f.rmdir()

        openstudio_dir.mkdir(parents=True, exist_ok=True)

        # write config file
        with open(config_file, "w", encoding="utf-8") as fp:
            fp.write(self.json())

        # write model to target directory to reference in simulation
        model.to_hbjson(folder=openstudio_dir)

        simulation_control = SimulationControl(
            do_zone_sizing=True,
            do_system_sizing=True,
            do_plant_sizing=True,
            run_for_sizing_periods=False,
            run_for_run_periods=True,
        )

        sizing_parameter = SizingParameter(
            design_days=[
                epw.approximate_design_day("WinterDesignDay"),
                epw.approximate_design_day("SummerDesignDay"),
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
            terrain_type=self.form.terrain.value,
            sizing_parameter=sizing_parameter,
        )

        with open(sim_par_file, "w", encoding="utf-8") as fp:
            json.dump(simulation_parameter.to_dict(), fp)

        osw = to_openstudio_osw(
            openstudio_dir.as_posix(),
            hbjson_file.as_posix(),
            sim_par_file.as_posix(),
            additional_measures=None,
            epw_file=epw_file.as_posix(),
        )
        _, idf = run_osw(osw, silent=True)

        sql, _, _, _, _ = run_idf(
            idf_file_path=idf,
            epw_file_path=epw_file.as_posix(),
            expand_objects=True,
            silent=True,
        )

        return sql

    def load_results(self, epw: EPW, directory: Path) -> pd.DataFrame:
        """Get the results of the simulation, normalised per area."""

        model = self.model(epw=epw)
        sql = self.simulate(epw, directory)

        logger.info(f"{self.identifier} - Loading results")

        lb = LoadBalance.from_sql_file(model=model, sql_path=Path(sql).as_posix())
        sqlr = SQLiteResult(file_path=Path(sql).as_posix())

        d = []
        for i in lb.load_balance_terms(floor_normalized=True, include_storage=True):
            d.append(i)

        for i in sqlr.available_outputs:
            temp: list[HourlyContinuousCollection] = []
            for col in sqlr.data_collections_by_output_name(i):
                col: HourlyContinuousCollection
                try:
                    temp.append(
                        col.normalize_by_area(area=model.floor_area, area_unit="m2")
                    )
                except (AssertionError, ValueError):
                    temp.append(col)
            if len(temp) == 0:
                continue
            if temp[0].header.unit in ["C", "%"]:
                d.append(
                    temp[0].get_aligned_collection(
                        np.mean([j.values for j in temp], axis=0)
                    )
                )
            else:
                d.append(
                    temp[0].get_aligned_collection(
                        np.sum([j.values for j in temp], axis=0)
                    )
                )

        x = []
        for v in d:
            x.append(
                pd.Series(
                    v.values,
                    index=INDEX,
                    name=f"{v.header.metadata['type']} ({v.header.unit})",
                )
            )

        df = pd.concat(x, axis=1)

        return df

    def energy_consumption(
        self,
        epw: EPW,
        directory: Path,
        existing_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Get the energy consumption of the typology, including effects from equipment performance."""

        # load from existing dataframe if provided
        if existing_df is not None:
            df = existing_df
        else:
            df = self.results(epw, directory)

        # COOLING
        cooling = (
            df["Zone Ideal Loads Zone Total Cooling Energy Intensity (kWh)"].rename(
                "Cooling (kWh)"
            )
            / self.system.cooling_eer
        )

        # HEATING
        heating = (
            df["Zone Ideal Loads Zone Total Heating Energy Intensity (kWh)"].rename(
                "Heating (kWh)"
            )
            / self.system.heating_cop
        )

        # LIGHTING
        lighting = df["Lighting (kWh)"].rename("Lighting (kWh)")

        # EQUIPMENT
        equipment = df["Electric Equipment (kWh)"].rename("Electric Equipment (kWh)")

        # HOT WATER
        # try:
        #     hot_water_from_eplus = (
        #         df["Water Heater Electricity Energy Intensity (kWh)"].rename(
        #             "Hot Water (kWh) eplus"
        #         )
        #         / self.system.heating_cop
        #     )
        #     hot_water_from_calc = (
        #         df["Water Use Equipment Zone Sensible Heat Gain Energy Intensity (kWh)"]
        #         + df["Water Use Equipment Zone Latent Gain Energy Intensity (kWh) calc"]
        #     ).rename("Hot Water (kWh) calc") / self.system.heating_cop
        # except KeyError:
        #     hot_water_from_eplus = pd.Series(
        #         np.zeros(8760), index=INDEX, name="Hot Water (kWh) eplus"
        #     )
        #     hot_water_from_calc = pd.Series(
        #         np.zeros(8760), index=INDEX, name="Hot Water (kWh) calc"
        #     )
        try:
            hot_water = df["Service Hot Water (kWh)"].rename("Hot Water (kWh)") / self.system.heating_cop
        except KeyError:
            hot_water = pd.Series(
                np.zeros(8760), index=INDEX, name="Hot Water (kWh)"
            )

        # LIFTS
        lifts = self.estimate_lift_energy()

        # PUMPS
        try:
            pumps_from_eplus = (
                df["Pump Electricity Energy Intensity (kWh)"].rename(
                    "Pumps (kWh) eplus"
                )
                * self.system.pump_power
            )
        except KeyError:
            pumps_from_eplus = pd.Series(
                np.zeros(8760), index=INDEX, name="Pumps (kWh) eplus"
            )

        chw_delta_t = 6  # K
        chw_cp = 4.18  # kJ/kgK
        chw_flow_ls = (cooling / 1000) / chw_cp / chw_delta_t
        pumps_from_calc = (chw_flow_ls * self.system.pump_power).rename(
            "Pumps (kWh) calc"
        ) / 1000

        # FANS - convert m3/s to kWh using Wh/L/s fan_power
        vent_flowrate_l = (
            df["Zone Mechanical Ventilation Current Density Volume Flow Rate (m3/s)"]
            * 1000
        )
        fans = (vent_flowrate_l * self.system.fan_power / 1000).rename("Fans (kWh)")

        # combine results
        return pd.concat(
            [
                cooling,
                heating,
                lighting,
                equipment,
                hot_water,
                lifts,
                pumps_from_eplus,
                pumps_from_calc,
                fans,
            ],
            axis=1,
        )
