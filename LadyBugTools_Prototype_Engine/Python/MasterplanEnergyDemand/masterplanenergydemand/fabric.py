# region: IMPORTS
# pylint: disable=E0401,E0611

from copy import deepcopy

import numpy as np
from honeybee.boundarycondition import Ground, Outdoors
from honeybee.room import Room
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.internalmass import InternalMass
from pydantic import BaseModel, Field

from .config import logger
from .enums import (EPW, BuildingType, ConstructionType, Vintage,
                    default_construction_type, default_constructionset)
from .util import construction_sri, estimate_sri_properties

# pylint: enable=E0401,E0611
# endregion: IMPORTS


class Fabric(BaseModel):
    """The fabric (thermal envelope) of a building."""

    wall_u_value: list[float] = Field(
        description="The U-value of the walls, in clockwise order from N, NE, ...",
        ge=0.05,
        le=6,
        min_items=8,
        max_items=8,
    )
    wall_sri: float = Field(
        description="The Solar Reflectance Index of the walls.",
        ge=0,
        le=122,
    )
    floor_u_value: float = Field(
        description="The U-value of the ground floor.",
        ge=0.05,
        le=6,
    )
    roof_u_value: float = Field(
        description="The U-value of the roof.", ge=0.05, le=6
    )
    window_u_value: list[float] = Field(
        description="The U-value of the windows, in clockwise order from N, NE, ...",
        ge=0.05,
        le=6,
        min_items=8,
        max_items=8,
    )
    window_shgc: list[float] = Field(
        description="The Solar Heat Gain Coefficient of the windows, in clockwise order from N, NE, ...",
        ge=0,
        le=1,
        min_items=8,
        max_items=8,
    )
    skylight_u_value: float = Field(
        description="The U-value of the skylight.",
        ge=0.05,
        le=6,
    )
    skylight_shgc: float = Field(
        description="The Solar Heat Gain Coefficient of the skylight.",
        ge=0,
        le=1,
    )
    roof_sri: float = Field(
        description="The Solar Reflectance Index of the roof.",
        ge=0,
        le=122,
    )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({hex(id(self))})"
    
    @classmethod
    def parse_obj_extended(cls, d: dict) -> "Fabric":
        """Create a Fabric object from an extended dictionary, where directional glazing_ratio is present."""

        # copy to prevent mutation
        d = deepcopy(d)

        # process
        lookup = {
            "wall_u_value": [
                "wall_u_value_N",
                "wall_u_value_NE",
                "wall_u_value_E",
                "wall_u_value_SE",
                "wall_u_value_S",
                "wall_u_value_SW",
                "wall_u_value_W",
                "wall_u_value_NW",
            ],
            "window_u_value": [
                "window_u_value_N",
                "window_u_value_NE",
                "window_u_value_E",
                "window_u_value_SE",
                "window_u_value_S",
                "window_u_value_SW",
                "window_u_value_W",
                "window_u_value_NW",
            ],
            "window_shgc": [
                "window_shgc_N",
                "window_shgc_NE",
                "window_shgc_E",
                "window_shgc_SE",
                "window_shgc_S",
                "window_shgc_SW",
                "window_shgc_W",
                "window_shgc_NW",
            ],
        }

        for target_var, additional_keys in lookup.items():
            # check that the target variable is not present if additional keys are present
            if target_var in d:
                if any(k in d for k in additional_keys):
                    raise ValueError(
                        f"Extended dictionary must not contain key: {target_var} if directional values are present."
                    )
                else:
                    continue

            # check if all additional keys are present
            for k in additional_keys:
                if k not in d:
                    raise ValueError(f"Extended dictionary must contain key: {k}")

            # create the list list
            target_var_values = [d[key] for key in additional_keys]

            # modify input dict to remove additional keys and add target_var
            for key in additional_keys:
                d.pop(key)
            d[target_var] = target_var_values

        return cls.parse_obj(d)
    
    @classmethod
    def random(cls, seed: int= None) -> "Fabric":
        """Return an example instance of the class populated with random data."""
        
        logger.info(f"Creating random {cls.__name__}")

        np.random.seed(seed)

        return cls(
            wall_u_value=np.random.uniform(0.05, 6, 8).tolist(),
            wall_sri=np.random.uniform(0, 122),
            floor_u_value=np.random.uniform(0.05, 6),
            roof_u_value=np.random.uniform(0.05, 6),
            window_u_value=np.random.uniform(0.05, 6, 8).tolist(),
            window_shgc=np.random.uniform(0, 1, 8).tolist(),
            skylight_u_value=np.random.uniform(0.05, 6),
            skylight_shgc=np.random.uniform(0, 1),
            roof_sri=np.random.uniform(0, 122),
        )
    
    @classmethod
    def from_building_type(cls, building_type: BuildingType, epw: EPW, vintage: Vintage) -> "Fabric":
        """_"""

        logger.info(f"Creating default {cls.__name__} for {building_type}")
        
        construction_type = default_construction_type(building_type)
        constr_set = default_constructionset(vintage=vintage, construction_type=construction_type, epw=epw)

        return cls(
            wall_u_value=[constr_set.wall_set.exterior_construction.u_factor] * 8,
            wall_sri=construction_sri(constr_set.wall_set.exterior_construction),
            floor_u_value=constr_set.floor_set.ground_construction.u_factor,
            roof_u_value=constr_set.roof_ceiling_set.exterior_construction.u_factor,
            window_u_value=[constr_set.aperture_set.window_construction.u_factor] * 8,
            window_shgc=[constr_set.aperture_set.window_construction.shgc] * 8,
            skylight_u_value=constr_set.aperture_set.skylight_construction.u_factor,
            skylight_shgc=constr_set.aperture_set.skylight_construction.shgc,
            roof_sri=construction_sri(constr_set.roof_ceiling_set.exterior_construction),
        )

    def constructions(self) -> dict:
        """Determine the constructions for a building type."""

        logger.info(f"{self} - Creating constructions")

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
                self.wall_u_value,
                self.window_u_value,
                self.window_shgc,
                ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
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

    def internal_mass(
        self,
        room: Room,
        epw: EPW,
        vintage: Vintage,
        construction_type: ConstructionType,
    ) -> dict:
        """Determine the internal mass for a building type."""

        logger.info(f"{self} - Creating internal mass for {room.identifier}")

        constr_set = default_constructionset(vintage=vintage, construction_type=construction_type, epw=epw)

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