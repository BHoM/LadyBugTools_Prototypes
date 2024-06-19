# region: IMPORTS
# pylint: disable=E0401

import inspect
from typing import Any

import numpy as np
from honeybee.boundarycondition import Outdoors
from honeybee.model import Aperture, Face, Model, Room, Shade
from ladybug_geometry.geometry2d import Point2D, Polygon2D, Vector2D
from ladybug_geometry.geometry3d import (Face3D, LineSegment3D, Point3D,
                                         Vector3D)

from .config import logger
from .enums import (BuildingType, TerrainType, default_context_shade_distance,
                    default_floor_height, default_footprint_area,
                    default_glazing_ratio, default_number_of_floors)
from .util import (_log_message, face_orientation, list_of_nums_validator,
                   number_validator)

# pylint: enable=E0401
# endregion: IMPORTS




class Form:
    """The massing/shape of a built form."""

    def __init__(
        self,
        average_footprint_area: float = None,
        average_num_floors: int = None,
        average_floor_height: float = None,
        rotation: float = None,
        terrain: TerrainType = None,
        glazing_ratio: list[float] = None,
        skylight_ratio: float = None,
    ):
        self.average_footprint_area = average_footprint_area
        self.average_num_floors = average_num_floors
        self.average_floor_height = average_floor_height
        self.rotation = rotation
        self.terrain = terrain
        self.glazing_ratio = glazing_ratio
        self.skylight_ratio = skylight_ratio

    def __eq__(self, other: "Form") -> bool:
        return self.to_dict() == other.to_dict()

    def to_dict(self) -> dict:
        """Convert this object to a dictionary."""
        return {k[1:]: v for k, v in self.__dict__.items()}

    @classmethod
    def from_defaults(
        cls,
        building_type: BuildingType,
        rotation: float = None,
        terrain: TerrainType = None,
        skylight_ratio: float = None,
    ) -> "Form":
        """Create a Form object from default values for a given BuildingType.

        Args:
            building_type: The type of building to create defaults for.
            rotation: The rotation of the building in degrees.
            terrain: The terrain type for the building.
            skylight_ratio: The ratio of skylight area to floor area.

        Returns:
            Form: A Form object with default values for the given BuildingType.
        """
        average_footprint_area = default_footprint_area(building_type)
        logger.info(
            '> using "average_footprint_area" of %sm2 for %s',
            average_footprint_area,
            building_type,
        )

        average_num_floors = default_number_of_floors(building_type)
        logger.info(
            '> using "average_num_floors" of %s for %s',
            average_num_floors,
            building_type,
        )

        average_floor_height = default_floor_height(building_type)
        logger.info(
            '> using "average_floor_height" of %sm for %s',
            average_floor_height,
            building_type,
        )

        glazing_ratio = default_glazing_ratio(building_type)
        logger.info(
            '> using "glazing_ratio" of %s for %s', glazing_ratio, building_type
        )

        return cls(
            average_footprint_area=average_footprint_area,
            average_num_floors=average_num_floors,
            average_floor_height=average_floor_height,
            rotation=rotation,
            terrain=terrain,
            glazing_ratio=[glazing_ratio] * 8,
            skylight_ratio=skylight_ratio,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Form":
        """Create a Form object from a dictionary."""

        # convert strings back into Enums if necessary
        if "terrain" in data:
            data["terrain"] = TerrainType(data["terrain"])

        return cls(**data)

    # region: VALIDATION
    @property
    def average_footprint_area(self):
        """Getter for the average_footprint_area property."""
        return self._average_footprint_area

    @average_footprint_area.setter
    def average_footprint_area(self, value):
        """Setter for the average_footprint_area property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = 100
            _log_message(prop_name, value, "m2")
        number_validator(value, prop_name, gt=0)
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
            value = 1
            _log_message(prop_name, value)
        if not isinstance(value, int):
            raise ValueError(f"{self} - {prop_name} must be an integer")
        number_validator(value, prop_name, gt=0)
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
            value = 3.5
            _log_message(prop_name, value, "m")
        number_validator(value, prop_name, gt=0)
        self._average_floor_height = value

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
            _log_message(prop_name, value, "°")
        number_validator(value, prop_name, ge=0, lt=360)
        self._rotation = value

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
            _log_message(prop_name, value)
        else:
            try:
                value = TerrainType(value)
            except ValueError:
                value = TerrainType[value]
        self._terrain = value

    @property
    def glazing_ratio(self):
        """Getter for the glazing_ratio property."""
        return self._glazing_ratio

    @glazing_ratio.setter
    def glazing_ratio(self, value):
        """Setter for the glazing_ratio property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = [0.25] * 8
            _log_message(prop_name, value)
        list_of_nums_validator(value, prop_name, 8, ge=0, le=0.95)
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
            _log_message(prop_name, value)
        number_validator(value, prop_name, ge=0, le=0.95)
        self._skylight_ratio = value

    # endregion: VALIDATION

    # region: CALCULATED PROPERTIES
    @property
    def building_height(self) -> float:
        """Get the typical height for an individual building."""
        return self.average_num_floors * self.average_floor_height

    def footprint(self) -> Polygon2D:
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
        footprint = footprint.scale(n)

        return footprint

    def base_model(
        self,
    ) -> Model:
        """Create the base model containing all geometry, without construction assignment."""

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
            context_height = self.building_height * 0.75
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
