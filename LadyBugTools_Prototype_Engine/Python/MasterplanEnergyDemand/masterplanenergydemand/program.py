# region: IMPORTS
# pylint: disable=E0401

from copy import deepcopy

import numpy as np
import pandas as pd
from honeybee_energy.lib.scheduletypelimits import humidity, temperature
from honeybee_energy.programtype import ProgramType
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval
from pydantic import BaseModel, Field, root_validator

from .config import INDEX, logger
from .enums import BuildingType, default_program

# pylint: enable=E0401
# endregion: IMPORTS


class Program(BaseModel):
    """The program and conditioning of a building."""

    occupant_density: float = Field(
        description="The number of occupants per area (people/m2).",
        ge=0,
        repr=False,
    )
    lighting_power_density: float = Field(
        description="The lighting power per area (W/m2).",
        ge=0,
        repr=False,
    )
    equipment_power_density: float = Field(
        description="The equipment power per area (W/m2).",
        ge=0,
        repr=False,
    )
    infiltration_rate: float = Field(
        description="The infiltration rate (m3/s/m2).",
        ge=0,
        repr=False,
    )
    ventilation_rate: float = Field(
        description="The ventilation rate (m3/s/person).",
        ge=0,
        repr=False,
    )
    heating_setpoint: float = Field(
        description="The heating setpoint temperature (C).",
        ge=0,
        repr=False,
    )
    heating_setback: float = Field(
        description="The heating setback temperature (C).",
        ge=0,
        repr=False,
    )
    cooling_setpoint: float = Field(
        description="The cooling setpoint temperature (C).",
        ge=0,
        repr=False,
    )
    cooling_setback: float = Field(
        description="The cooling setback temperature (C).",
        ge=0,
        repr=False,
    )
    humidifying_setpoint: float = Field(
        description="The humidifying setpoint (%RH).",
        ge=0,
        le=100,
        repr=False,
    )
    humidifying_setback: float = Field(
        description="The humidifying setback (%RH).",
        ge=0,
        le=100,
        repr=False,
    )
    dehumidifying_setpoint: float = Field(
        description="The dehumidifying setpoint (%RH).",
        ge=0,
        le=100,
        repr=False,
    )
    dehumidifying_setback: float = Field(
        description="The dehumidifying setback (%RH).",
        ge=0,
        le=100,
        repr=False,
    )

    @root_validator(pre=False)
    def validate_atts(cls, values):
        """Validate the attributes, ensuring that systems do not override each other."""
        
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

    @classmethod
    def parse_obj_extended(cls, d: dict, replace_null_with_defaults: bool = False, building_type: BuildingType = None) -> "Program":
        """Process an extended dictionary, augmenting the parse_obj method."""

        d = deepcopy(d)

        if replace_null_with_defaults and (building_type is None):
            raise ValueError("Building type must be provided if populate_with_defaults is True.")
        if replace_null_with_defaults:
            default_prog = cls.from_building_type(building_type)
        
        if pd.isnull(d.get("occupant_density", np.nan)):
            if replace_null_with_defaults:
                d["occupant_density"] = default_prog.occupant_density
            else:
                d["occupant_density"] = 0
            logger.info(f"occupant_density is null. Setting to {d['occupant_density']}.")
        
        if pd.isnull(d.get("lighting_power_density", np.nan)):
            if replace_null_with_defaults:
                d["lighting_power_density"] = default_prog.lighting_power_density
            else:
                d["lighting_power_density"] = 0
            logger.info(f"lighting_power_density is null. Setting to {d['lighting_power_density']}.")

        if pd.isnull(d.get("equipment_power_density", np.nan)):
            if replace_null_with_defaults:
                d["equipment_power_density"] = default_prog.equipment_power_density
            else:
                d["equipment_power_density"] = 0
            logger.info(f"equipment_power_density is null. Setting to {d['equipment_power_density']}.")
        
        if pd.isnull(d.get("infiltration_rate", np.nan)):
            if replace_null_with_defaults:
                d["infiltration_rate"] = default_prog.infiltration_rate
            else:
                d["infiltration_rate"] = 0.0006
            logger.info(f"infiltration_rate is null. Setting to {d['infiltration_rate']}.")
        
        if pd.isnull(d.get("ventilation_rate", np.nan)):
            if replace_null_with_defaults:
                d["ventilation_rate"] = default_prog.ventilation_rate
            else:
                d["ventilation_rate"] = 0
            logger.info(f"ventilation_rate is null. Setting to {d['ventilation_rate']}.")
        
        if pd.isnull(d.get("heating_setpoint", np.nan)):
            if replace_null_with_defaults:
                d["heating_setpoint"] = default_prog.heating_setpoint
            else:
                d["heating_setpoint"] = 20
            logger.info(f"heating_setpoint is null. Setting to {d['heating_setpoint']}.")
        
        if pd.isnull(d.get("heating_setback", np.nan)):
            if replace_null_with_defaults:
                d["heating_setback"] = default_prog.heating_setback
            else:
                d["heating_setback"] = 15
            logger.info(f"heating_setback is null. Setting to {d['heating_setback']}.")
        
        if pd.isnull(d.get("cooling_setpoint", np.nan)):
            if replace_null_with_defaults:
                d["cooling_setpoint"] = default_prog.cooling_setpoint
            else:
                d["cooling_setpoint"] = 24
            logger.info(f"cooling_setpoint is null. Setting to {d['cooling_setpoint']}.")
        
        if pd.isnull(d.get("cooling_setback", np.nan)):
            if replace_null_with_defaults:
                d["cooling_setback"] = default_prog.cooling_setback
            else:
                d["cooling_setback"] = 27
            logger.info(f"cooling_setback is null. Setting to {d['cooling_setback']}.")
        
        if pd.isnull(d.get("humidifying_setpoint", np.nan)):
            if replace_null_with_defaults:
                d["humidifying_setpoint"] = default_prog.humidifying_setpoint
            else:
                d["humidifying_setpoint"] = 40
            logger.info(f"humidifying_setpoint is null. Setting to {d['humidifying_setpoint']}.")
        
        if pd.isnull(d.get("humidifying_setback", np.nan)):
            if replace_null_with_defaults:
                d["humidifying_setback"] = default_prog.humidifying_setback
            else:
                d["humidifying_setback"] = 0
            logger.info(f"humidifying_setback is null. Setting to {d['humidifying_setback']}.")
        
        if pd.isnull(d.get("dehumidifying_setpoint", np.nan)):
            if replace_null_with_defaults:
                d["dehumidifying_setpoint"] = default_prog.dehumidifying_setpoint
            else:
                d["dehumidifying_setpoint"] = 60
            logger.info(f"dehumidifying_setpoint is null. Setting to {d['dehumidifying_setpoint']}.")

        if pd.isnull(d.get("dehumidifying_setback", np.nan)):
            if replace_null_with_defaults:
                d["dehumidifying_setback"] = default_prog.dehumidifying_setback
            else:
                d["dehumidifying_setback"] = 100
            logger.info(f"dehumidifying_setback is null. Setting to {d['dehumidifying_setback']}.")

        return cls.parse_obj(d)
    
    @classmethod
    def from_building_type(
        cls,
        building_type: BuildingType,
    ) -> "Program":
        """Create a program based on a building type."""

        base_program = default_program(building_type)

        if base_program is None:
            raise ValueError(f"No default program found for {building_type}.")

        if base_program.people is not None:
            occupant_density = base_program.people.people_per_area
        else:
            occupant_density = 0

        if base_program.lighting is not None:
            lighting_power_density = base_program.lighting.watts_per_area
        else:
            lighting_power_density = 0

        if base_program.electric_equipment is not None:
            equipment_power_density = base_program.electric_equipment.watts_per_area
        else:
            equipment_power_density = 0

        if base_program.infiltration is not None:
            infiltration_rate = base_program.infiltration.flow_per_exterior_area
        else:
            infiltration_rate = 0

        if base_program.ventilation is not None:
            ventilation_rate = base_program.ventilation.flow_per_person
        else:
            ventilation_rate = 0

        if base_program.setpoint is not None:
            if base_program.setpoint.heating_setpoint is None:
                heating_setpoint = 20
            else:
                heating_setpoint = base_program.setpoint.heating_setpoint
            if base_program.setpoint.heating_setback is None:
                heating_setback = 15
            else:
                heating_setback = base_program.setpoint.heating_setback
            if base_program.setpoint.cooling_setpoint is None:
                cooling_setpoint = 24
            else:
                cooling_setpoint = base_program.setpoint.cooling_setpoint
            if base_program.setpoint.cooling_setback is None:
                cooling_setback = 27
            else:
                cooling_setback = base_program.setpoint.cooling_setback

            if base_program.setpoint.humidifying_setpoint is None:
                humidifying_setpoint = 40
            else:
                humidifying_setpoint = base_program.setpoint.humidifying_setpoint
            if base_program.setpoint.humidifying_setback is None:
                humidifying_setback = 0
            else:
                humidifying_setback = base_program.setpoint.humidifying_setback
            if base_program.setpoint.dehumidifying_setpoint is None:
                dehumidifying_setpoint = 60
            else:
                dehumidifying_setpoint = base_program.setpoint.dehumidifying_setpoint
            if base_program.setpoint.dehumidifying_setback is None:
                dehumidifying_setback = 100
            else:
                dehumidifying_setback = base_program.setpoint.dehumidifying_setback

        return cls(
            occupant_density=occupant_density,
            lighting_power_density=lighting_power_density,
            equipment_power_density=equipment_power_density,
            infiltration_rate=infiltration_rate,
            ventilation_rate=ventilation_rate,
            heating_setpoint=heating_setpoint,
            heating_setback=heating_setback,
            cooling_setpoint=cooling_setpoint,
            cooling_setback=cooling_setback,
            humidifying_setpoint=humidifying_setpoint,
            humidifying_setback=humidifying_setback,
            dehumidifying_setpoint=dehumidifying_setpoint,
            dehumidifying_setback=dehumidifying_setback,
        )

    @classmethod
    def random(cls, seed: int = None) -> "Program":
        """Create a random program."""
        
        np.random.seed(seed)
        
        return cls(
            occupant_density=np.random.uniform(0.1, 0.5),
            lighting_power_density=np.random.uniform(5, 10),
            equipment_power_density=np.random.uniform(5, 10),
            infiltration_rate=np.random.uniform(0.0001, 0.0005),
            ventilation_rate=np.random.uniform(0.0001, 0.0005),
            heating_setpoint=np.random.uniform(20, 22),
            heating_setback=np.random.uniform(15, 17),
            cooling_setpoint=np.random.uniform(24, 26),
            cooling_setback=np.random.uniform(27, 29),
            humidifying_setpoint=np.random.uniform(40, 59),
            humidifying_setback=np.random.uniform(0, 20),
            dehumidifying_setpoint=np.random.uniform(60, 79),
            dehumidifying_setback=np.random.uniform(80, 100),
        )

    def program_type(self, building_type: BuildingType) -> ProgramType:
        """Return a programtype program based on building type."""

        base_program = default_program(building_type)
        program = base_program.duplicate()
        program.unlock()
        program.identifier = f"Custom program based on {building_type}"

        if program.people is not None:
            program.people.identifier = "Custom People"
            program.people.people_per_area = self.occupant_density

        if program.lighting is not None:
            program.lighting.identifier = "Custom Lighting"
            program.lighting.watts_per_area = self.lighting_power_density

        if program.electric_equipment is not None:
            program.electric_equipment.identifier = "Custom Equipment"
            program.electric_equipment.watts_per_area = self.equipment_power_density

        if program.infiltration is not None:
            program.infiltration.identifier = "Custom Infiltration"
            program.infiltration.flow_per_exterior_area = self.infiltration_rate

        if program.ventilation is not None:
            program.ventilation.identifier = "Custom Ventilation"
            program.ventilation.flow_per_person = self.ventilation_rate

        if program.setpoint is not None:
            program.setpoint.identifier = "Custom Setpoint"
            old_heating_schedule = np.array(
                program.setpoint.heating_schedule.data_collection().values
            )
            new_heating_schedule = np.interp(
                old_heating_schedule,
                [old_heating_schedule.min(), old_heating_schedule.max()],
                [self.heating_setback, self.heating_setpoint],
            )
            program.setpoint.heating_schedule = ScheduleFixedInterval(
                "Custom Heating Schedule",
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
                "Custom Cooling Schedule",
                new_cooling_schedule,
                temperature,
            )

            if program.setpoint.humidifying_setpoint is None:
                program.setpoint.humidifying_schedule = ScheduleFixedInterval(
                    "Custom Humidifying Schedule",
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
                    "Custom Dehumidifying Schedule",
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
                program.setpoint.humidifying_setpoint = self.humidifying_setpoint
                program.setpoint.humidifying_setback = self.humidifying_setback
                program.setpoint.dehumidifying_setpoint = self.dehumidifying_setpoint
                program.setpoint.dehumidifying_setback = self.dehumidifying_setback
                program.lock()

        return program

    def profile_lighting(self, building_type: BuildingType) -> pd.Series:
        """Generate a time-indexed profile for lighting use in the building."""
        program = self.program_type(building_type=building_type)

        if program.lighting is None:
            return pd.Series(data=0, index=INDEX, name="Lighting (W/m2)")

        schedule = program.lighting.schedule.data_collection().values

        return (
            pd.Series(
                data=program.lighting.watts_per_area,
                index=INDEX,
                name="Lighting (W/m2)",
            )
            * schedule
        )

    def profile_equipment(self, building_type: BuildingType) -> pd.Series:
        """Generate a time-indexed profile for equipment use in the building."""
        program = self.program_type(building_type=building_type)

        if program.electric_equipment is None:
            return pd.Series(data=0, index=INDEX, name="Equipment (W/m2)")

        schedule = program.electric_equipment.schedule.data_collection().values

        return (
            pd.Series(
                data=program.electric_equipment.watts_per_area,
                index=INDEX,
                name="Equipment (W/m2)",
            )
            * schedule
        )

    def profile_infiltration(self, building_type: BuildingType) -> pd.Series:
        """Generate a time-indexed profile for infiltration in the building."""
        program = self.program_type(building_type=building_type)

        if program.infiltration is None:
            return pd.Series(data=0, index=INDEX, name="Infiltration (m3/s/m2)")

        schedule = program.infiltration.schedule.data_collection().values

        return (
            pd.Series(
                data=program.infiltration.flow_per_exterior_area,
                index=INDEX,
                name="Infiltration (m3/s/m2)",
            )
            * schedule
        )

    def profile_ventilation(self, building_type: BuildingType) -> pd.Series:
        """Generate a time-indexed profile for ventilation in the building."""
        program = self.program_type(building_type=building_type)

        if program.ventilation is None:
            return pd.Series(data=0, index=INDEX, name="Ventilation (m3/s)")

        if program.ventilation.schedule is None:
            schedule = 1
        else:
            schedule = program.ventilation.schedule.data_collection().values

        return (
            pd.Series(
                data=program.ventilation.flow_per_person,
                index=INDEX,
                name="Ventilation (m3/s)",
            )
            * schedule
        )

    def profile_people(self, building_type: BuildingType) -> pd.Series:
        """Generate a time-indexed profile for people in the building."""
        program = self.program_type(building_type=building_type)

        if program.people is None:
            return pd.Series(data=0, index=INDEX, name="People (W/m2)")

        schedule = program.people.occupancy_schedule.data_collection().values
        activity_schedule = program.people.activity_schedule.data_collection().values

        return (
            pd.Series(
                data=program.people.people_per_area, index=INDEX, name="People (W/m2)"
            )
            * schedule
            * activity_schedule
        )

    def profile_heating(self, building_type: BuildingType) -> pd.Series:
        """Generate a time-indexed profile for heating setpoint in the building."""
        program = self.program_type(building_type=building_type)

        if program.setpoint is None:
            return pd.Series(data=0, index=INDEX, name="Heating Setpoint (C)")

        return pd.Series(
            data=program.setpoint.heating_schedule.data_collection.values,
            index=INDEX,
            name="Heating Setpoint (C)",
        )

    def profile_cooling(self, building_type: BuildingType) -> pd.Series:
        """Generate a time-indexed profile for cooling setpoint in the building."""
        program = self.program_type(building_type=building_type)

        if program.setpoint is None:
            return pd.Series(data=0, index=INDEX, name="Cooling Setpoint (C)")

        return pd.Series(
            data=program.setpoint.cooling_schedule.data_collection.values,
            index=INDEX,
            name="Cooling Setpoint (C)",
        )

    def profile_humidifying(self, building_type: BuildingType) -> pd.Series:
        """Generate a time-indexed profile for humidifying setpoint in the building."""
        program = self.program_type(building_type=building_type)

        if program.setpoint is None:
            return pd.Series(data=0, index=INDEX, name="Humidifying Setpoint (%)")

        return pd.Series(
            data=program.setpoint.humidifying_schedule.data_collection.values,
            index=INDEX,
            name="Humidifying Setpoint (%)",
        )

    def profile_dehumidifying(self, building_type: BuildingType) -> pd.Series:
        """Generate a time-indexed profile for dehumidifying setpoint in the building."""
        program = self.program_type(building_type=building_type)

        if program.setpoint is None:
            return pd.Series(data=0, index=INDEX, name="Dehumidifying Setpoint (%)")

        return pd.Series(
            data=program.setpoint.dehumidifying_schedule.data_collection.values,
            index=INDEX,
            name="Dehumidifying Setpoint (%)",
        )

    def profile_shw(self, building_type: BuildingType) -> pd.Series:
        """Generate a time-indexed profile for service hot water in the building."""

        program = self.program_type(building_type=building_type)

        if program.service_hot_water is None:
            return pd.Series(data=0, index=INDEX, name="Hot Water (L/h/m2)")

        if program.service_hot_water.schedule is None:
            schedule = 1
        else:
            schedule = program.service_hot_water.schedule.data_collection().values

        return (
            pd.Series(
                data=program.service_hot_water.flow_per_area,
                index=INDEX,
                name="Hot Water (L/h/m2)",
            )
            * schedule
        )

