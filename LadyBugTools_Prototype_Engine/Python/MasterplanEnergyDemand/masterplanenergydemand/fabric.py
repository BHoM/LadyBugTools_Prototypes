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
from .enums import (BuildingType, ConstructionType, TerrainType, Vintage,
                    default_construction_type, default_constructionset,
                    default_context_shade_distance, default_floor_height,
                    default_footprint_area, default_glazing_ratio,
                    default_number_of_floors)
from .util import (_log_message, face_orientation, list_of_nums_validator,
                   number_validator)

# pylint: enable=E0401
# endregion: IMPORTS

class Fabric:
    """The fabric of a built form."""

    def __init__(
        self,
        vintage: Vintage = None,
        construction_type: ConstructionType = None,
        wall_u_value = None,
        wall_sri = None,
        floor_u_value = None,
        roof_u_value = None,
        window_u_value = None,
        window_shgc = None,
        skylight_u_value = None,
        skylight_shgc = None,
        roof_sri = None,
    ):
        self.vintage = vintage
        self.construction_type = construction_type
        self.wall_u_value = wall_u_value
        self.wall_sri = wall_sri
        self.floor_u_value = floor_u_value
        self.roof_u_value = roof_u_value
        self.window_u_value = window_u_value
        self.window_shgc = window_shgc
        self.skylight_u_value = skylight_u_value
        self.skylight_shgc = skylight_shgc
        self.roof_sri = roof_sri
        

    def __eq__(self, other: "Fabric") -> bool:
        return self.to_dict() == other.to_dict()

    def to_dict(self) -> dict:
        """Convert this object to a dictionary."""
        return {k[1:]: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "Fabric":
        """Create a Fabric object from a dictionary."""

        # convert strings back into Enums if necessary
        if "vintage" in data:
            data["vintage"] = Vintage(data["vintage"])

        return cls(**data)

    @classmethod
    def from_defaults(
        cls,
        building_type: BuildingType,
        wall_u_value = None,
        wall_sri = None,
        floor_u_value = None,
        roof_u_value = None,
        window_u_value = None,
        window_shgc = None,
        skylight_u_value = None,
        skylight_shgc = None,
        roof_sri = None,
    ) -> "Fabric":
        """"""
        construction_type = default_construction_type(building_type)
        logger.info(
            '> using "construction_type" of %s for %s',
            construction_type,
            building_type,
        )

        return cls(
            vintage=Vintage.ASHRAE_901_2019,
            construction_type=construction_type,
            wall_u_value=wall_u_value,
            wall_sri=wall_sri,
            floor_u_value=floor_u_value,
            roof_u_value=roof_u_value,
            window_u_value=window_u_value,
            window_shgc=window_shgc,
            skylight_u_value=skylight_u_value,
            skylight_shgc=skylight_shgc,
            roof_sri=roof_sri,
        )


    # region: VALIDATION
    
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
            _log_message(prop_name, value)
        else:
            try:
                try:
                    value = Vintage(value)
                except ValueError:
                    value = Vintage[value]
            except KeyError as exc:
                raise ValueError(f"Invalid vintage: {value}") from exc
        self._vintage = value
    
    @property
    def construction_type(self):
        """Getter for the construction_type property."""
        return self._construction_type

    @construction_type.setter
    def construction_type(self, value):
        """Setter for the construction_type property."""
        prop_name = inspect.currentframe().f_code.co_name
        if value is None:
            value = ConstructionType.MASS
            _log_message(prop_name, value)
        else:
            try:
                try:
                    value = ConstructionType(value)
                except ValueError:
                    value = ConstructionType[value]
            except KeyError as exc:
                raise ValueError(f"Invalid construction_type: {value}") from exc
        self._construction_type = value

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
            _log_message(prop_name, value)
        list_of_nums_validator(value, prop_name, 8, ge=0.05, le=6)
        self._wall_u_value = value

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
            _log_message(prop_name, value)
        list_of_nums_validator(value, prop_name, 8, ge=0, le=122)
        self._wall_sri = value
    
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
            _log_message(prop_name, value)
        number_validator(value, prop_name, ge=0.05, le=6)
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
            value = 0.5
            _log_message(prop_name, value)
        number_validator(value, prop_name, ge=0.05, le=6)
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
            value = [0.5] * 8
            _log_message(prop_name, value)
        list_of_nums_validator(value, prop_name, 8, ge=0.05, le=6)
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
            value = [0.5] * 8
            _log_message(prop_name, value)
        list_of_nums_validator(value, prop_name, 8, ge=0, le=1)
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
            value = 0.5
            _log_message(prop_name, value)
        number_validator(value, prop_name, ge=0.05, le=6)
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
            value = 0.5
            _log_message(prop_name, value)
        number_validator(value, prop_name, ge=0, le=1)
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
            _log_message(prop_name, value)
        number_validator(value, prop_name, ge=0, le=122)
        self._roof_sri = value
    
    # endregion: VALIDATION
