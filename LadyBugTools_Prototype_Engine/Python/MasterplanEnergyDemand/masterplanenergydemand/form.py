# region: IMPORTS
# pylint: disable=E0401

from copy import deepcopy

import numpy as np
from honeybee.boundarycondition import Outdoors
from honeybee.model import Aperture, Face, Model, Room, Shade
from ladybug_geometry.geometry2d import Point2D, Polygon2D, Vector2D
from ladybug_geometry.geometry3d import (Face3D, LineSegment3D, Point3D,
                                         Vector3D)
from pydantic import BaseModel, Field

from .config import logger
from .enums import (BuildingType, TerrainType, default_context_shade_distance,
                    default_floor_height, default_footprint_area,
                    default_glazing_ratio, default_number_of_floors,
                    default_skylight_ratio)
from .util import face_orientation

# pylint: enable=E0401
# endregion: IMPORTS


class Form(BaseModel):
    """The massing/shape of a built form."""

    average_footprint_area: float = Field(
        description="The average footprint area of the building (m2).", gt=0
    )
    average_num_floors: int = Field(
        description="The average number of floors in the building.", gt=0
    )
    average_floor_height: float = Field(
        description="The average floor height of the building (m).", gt=0
    )
    rotation: float = Field(
        description="The rotation of the building (degrees).", ge=0, lt=360
    )
    terrain: TerrainType = Field(
        description="The terrain type for the building."
    )
    glazing_ratio: list[float] = Field(
        description="The ratio of glazing area to floor area for each orientation.",
        ge=0,
        le=0.95,
        max_items=8,
        min_items=8,
    )
    skylight_ratio: float = Field(
        description="The ratio of skylight area to floor area.",
        ge=0,
        le=0.95,
    )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({hex(id(self))})"

    @classmethod
    def random(cls, seed: int = None) -> "Form":
        """Return an example instance of the class populated with random data."""
        
        logger.info(f"Generating random {cls.__name__} object")

        np.random.seed(seed)
        
        return cls(
            average_footprint_area=np.random.uniform(0.1, 1000),
            average_num_floors=np.random.uniform(1, 10),
            average_floor_height=np.random.uniform(0.5, 5),
            rotation=np.random.uniform(0, 360),
            terrain=np.random.choice(list(TerrainType)),
            glazing_ratio=np.random.uniform(0, 0.95, 8).tolist(),
            skylight_ratio=np.random.uniform(0, 0.95),
        )

    @classmethod
    def from_building_type(
        cls,
        building_type: BuildingType,
        rotation: float,
        terrain: TerrainType,
    ) -> "Form":
        """Create a Form object from default values for a given BuildingType.

        Args:
            building_type: The type of building to create defaults for.
            rotation: The rotation of the building in degrees.
            terrain: The terrain type for the building.

        Returns:
            Form: A Form object with default values for the given BuildingType.
        """

        logger.info(f"Creating default {cls.__name__} for {building_type}")
        
        return cls(
            average_footprint_area=default_footprint_area(building_type),
            average_num_floors=default_number_of_floors(building_type),
            average_floor_height=default_floor_height(building_type),
            rotation=rotation,
            terrain=terrain,
            glazing_ratio=[default_glazing_ratio(building_type)] * 8,
            skylight_ratio=default_skylight_ratio(building_type),
        )

    @classmethod
    def parse_obj_extended(cls, d: dict) -> "Form":
        """Create a Form object from an extended dictionary, where directional glazing_ratio is present."""

        # copy to prevent mutation
        d = deepcopy(d)

        # process
        lookup = {
            "glazing_ratio": [
                "glazing_ratio_N",
                "glazing_ratio_NE",
                "glazing_ratio_E",
                "glazing_ratio_SE",
                "glazing_ratio_S",
                "glazing_ratio_SW",
                "glazing_ratio_W",
                "glazing_ratio_NW",
            ]
        }

        for target_var, additional_keys in lookup.items():
            # check that the target variable is not present
            if target_var in d:
                raise ValueError(
                    f"Extended dictionary must not contain key: {target_var}"
                )

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

    # region: CALCULATED PROPERTIES
    def building_height(self) -> float:
        """Get the typical height for an individual building."""
        logger.info(f"{self} - Calculating building height")

        return self.average_num_floors * self.average_floor_height

    def footprint(self) -> Polygon2D:
        """Create the footprint for the building."""

        logger.info(f"{self} - Creating footprint")

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
        footprint = footprint.scale(n)

        return footprint

    def base_model(
        self,
    ) -> Model:
        """Create the base model containing all geometry, without construction assignment."""

        logger.info(f"{self} - Creating base model")

        # create the lookup dict for apertures in each orientation
        gr_lookup = dict(
            zip(*[["N", "NE", "E", "SE", "S", "SW", "W", "NW"], self.glazing_ratio])
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
            context_height = self.building_height() * 0.75
            for segment in self.footprint().offset(-context_distance).segments:
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
            aperture._identifier = aperture.parent.identifier  # pylint: disable=W0212

        return model

    # endregion: CALCULATED PROPERTIES
