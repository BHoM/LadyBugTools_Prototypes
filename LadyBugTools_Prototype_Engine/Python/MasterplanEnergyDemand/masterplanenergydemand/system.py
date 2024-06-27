# region: IMPORTS
# pylint: disable=E0401


from copy import deepcopy

import numpy as np
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.shw import SHWSystem
from ladybug.epw import EPW
from pydantic import BaseModel, Field

from .config import logger
from .enums import (BuildingType, EconomizerType, Vintage, default_cooling_eer,
                    default_daylight_dimming,
                    default_demand_controlled_ventilation,
                    default_economizer_type, default_fan_power,
                    default_heating_cop, default_hr_effectiveness,
                    default_pump_power)

# pylint: enable=E0401
# endregion: IMPORTS


class System(BaseModel):
    """The program and conditioning of a building."""

    economizer_type: EconomizerType = Field(description="The type of economizer.")
    sensible_heat_recovery_effectiveness: float = Field(
        description="The effectiveness of sensible heat recovery.",
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    latent_heat_recovery_effectiveness: float = Field(
        description="The effectiveness of latent heat recovery.",
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    demand_controlled_ventilation: bool = Field(
        description="Whether demand-controlled ventilation is used."
    )
    daylight_dimming: bool = Field(description="Whether daylight dimming is used.")
    heating_cop: float = Field(
        description="The heating coefficient of performance (COP).",
        ge=0,
        le=5,
        allow_inf_nan=False,
    )
    cooling_eer: float = Field(
        description="The cooling energy efficiency ratio (EER).",
        ge=0,
        le=13,
        allow_inf_nan=False,
    )
    fan_power: float = Field(
        description="The fan power (W/l/s).", ge=0, allow_inf_nan=False
    )
    pump_power: float = Field(
        description="The pump power (W/l/s).", ge=0, allow_inf_nan=False
    )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({hex(id(self))})"

    @classmethod
    def random(cls, seed: int = None) -> "System":
        """Create a random system."""

        logger.info(f"Creating random {cls.__name__}")

        np.random.seed(seed)

        return cls(
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
        cls, building_type: str, epw: EPW, vintage: Vintage
    ) -> "System":
        """Generate a system from a building type."""

        shr, lhr = default_hr_effectiveness(
            building_type=building_type, vintage=vintage, epw=epw
        )

        return cls(
            economizer_type=default_economizer_type(
                building_type=building_type, vintage=vintage, epw=epw
            ),
            sensible_heat_recovery_effectiveness=shr,
            latent_heat_recovery_effectiveness=lhr,
            demand_controlled_ventilation=default_demand_controlled_ventilation(
                building_type=building_type, vintage=vintage
            ),
            daylight_dimming=default_daylight_dimming(
                building_type=building_type, vintage=vintage
            ),
            heating_cop=default_heating_cop(
                building_type=building_type, vintage=vintage, epw=epw
            ),
            cooling_eer=default_cooling_eer(
                building_type=building_type, vintage=vintage, epw=epw
            ),
            fan_power=default_fan_power(building_type=building_type, vintage=vintage),
            pump_power=default_pump_power(building_type=building_type, vintage=vintage),
        )

    def ideal_air(self) -> IdealAirSystem:
        """Return the ideal air system associated with the system."""

        logger.info(f"{self} - Creating IdealAirSystem")

        return IdealAirSystem(
            identifier="ideal_air_system",
            economizer_type=self.economizer_type.value,
            demand_controlled_ventilation=self.demand_controlled_ventilation,
            sensible_heat_recovery=self.sensible_heat_recovery_effectiveness,
            latent_heat_recovery=self.latent_heat_recovery_effectiveness,
        )

    def shw(self) -> SHWSystem:
        """Return the service hot water system associated with the system."""

        logger.info(f"{self} - Creating SHWSystem")

        return SHWSystem(identifier="shw_system", equipment_type="Electric_WaterHeater")
