"""Predefined buildint types to be used for masterplan-level city energy simulation.

TODO - Create method to generate Excel file containing one of each of these buildings 
ready to populate and edit or actual project simulation.
"""
import inspect
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from honeybee.boundarycondition import Outdoors
from honeybee.facetype import Floor, RoofCeiling, Wall
from honeybee.model import Face, Model, Shade
from honeybee.room import Room
from honeybee.typing import clean_string
from honeybee_energy.lib.constructionsets import (
    ConstructionSet, construction_set_by_identifier)
from honeybee_energy.lib.programtypes import (
    ProgramType, building_program_type_by_identifier,
    program_type_by_identifier)
from ladybug.epw import EPW
from ladybug_geometry.geometry2d import Point2D, Polygon2D, Vector2D
from ladybug_geometry.geometry3d import (Face3D, LineSegment3D, Point3D,
                                         Vector3D)
from matplotlib.axes import Axes
from scipy.spatial import ConvexHull

from ..enum import (BuildingForm, ConstructionType, TerrainType, Vintage,
                    typical_context_distance)
from ..utilities import angle_from_north, cardinality

logger = logging.getLogger(__name__.split(".", maxsplit=1)[0])


class BuildingType:
    """The root class for buildings in the masterplan energy demand model, containing default values."""

    def __init__(
        self,
        name: str,
        gross_floor_area: float,
        floors: int,
        floor_to_floor_height: float,
        glazing_ratio: list[float] | float,
        skylight_ratio: float = 0,
        building_form: BuildingForm = BuildingForm.CUBOID,
        aspect_ratio: float = 1,
        rotation: float = 0,
        terrain_type: TerrainType = TerrainType.URBAN,
        hb_name: str = None,
        metadata: dict[str, str] = None
    ):
        self.name = name
        self.gross_floor_area = gross_floor_area
        self.floors = floors
        self.floor_to_floor_height = floor_to_floor_height
        self.glazing_ratio = glazing_ratio
        self.skylight_ratio = skylight_ratio
        self.building_form = building_form
        self.aspect_ratio = aspect_ratio
        self.rotation = rotation
        self.terrain_type = terrain_type
        self.hb_name = hb_name
        self.metadata = metadata

    # VALIDATION #
    @property
    def glazing_ratio(self):
        """Getter for the glazing_ratio property."""
        return self._glazing_ratio

    @glazing_ratio.setter
    def glazing_ratio(self, value):
        """Setter for the glazing_ratio property."""
        prop_name = inspect.currentframe().f_code.co_name
        if not isinstance(value, list) or len(value) != 8:
            raise ValueError(f"{prop_name} must be a list of 8 numbers")
        for i in value:
            if not isinstance(i, (int, float)):
                raise ValueError(f"{prop_name} values must be numbers")
            if i < 0 or i > 0.95:
                raise ValueError(
                    f"{prop_name} values must be between 0 and 0.95"
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
        if not isinstance(value, (int, float)):
            raise ValueError(f"{prop_name} must be a number")
        if value < 0 or value > 0.95:
            raise ValueError(f"{prop_name} must be between 0 and 0.95")
        self._skylight_ratio = value

    # DUNDER #

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.name}"

    def __str__(self) -> str:
        return repr(self)

    # INTEROPS #

    def to_dict(self) -> dict[str, str | list | float | int | dict[str, str]]:
        """Convert this object into a dictionary."""
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                d[k[1:]] = v
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "BuildingType":
        """Create this object from a dictionary."""
        # TODO - hide this method from child classes ... maybe
        return cls(**data)
    
    def to_series(self) -> pd.Series:
        """Convert this object into a pandas Series object."""
        raise NotImplementedError()
    
    def from_series(self) -> "BuildingType":
        """Convert a pandas Series object into this target object."""
        raise NotImplementedError()
    
    def to_json(self, json_file: Path) -> pd.Series:
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
    
    def from_json(self, json_file: Path) -> "BuildingType":
        """Convert a json file into this target object."""

        with open(json_file, "r") as f:
            data = json.load(f)

        keys_enums = {
            "building_type": BuildingType,
            "construction_type": ConstructionType,
            "vintage": Vintage,
            "terrain": TerrainType,
            "building_form": BuildingForm,
        }
        for k, v in keys_enums.items():
            if k in data:
                data[k] = v[data[k]]

        return cls.from_dict(data)
    
    # Hidden methods for internal calculation of properties
    def _typical_footprint_area(self) -> float:
        """Get the typical footprint area of the building type."""
        return np.ceil(self.gross_floor_area / self.floors)

    def _number_of_buildings(self, total_gfa: float) -> float:
        """Return the number of buildings of this type required to meet the total GFA."""
        return total_gfa / self.gross_floor_area

    def _typical_height(self):
        """Return the typical height of the building type."""
        return self.floors * self.floor_to_floor_height

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
                    height=3 / float(self.building_form),
                    base_point=base_poly.center,
                    height_vector=Vector2D(0, 1),
                )
                footprint = base_poly.boolean_difference(
                    polygon=cutting_shape, tolerance=0.1
                )[0]
            case BuildingForm.U_SHAPED:
                cutting_shape: Polygon2D = Polygon2D.from_rectangle(
                    base=1,
                    height=4 / float(self.building_form),
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
        if footprint.area < self._typical_footprint_area():
            new_area = footprint.area
            n = 1.01
            while new_area < self._typical_footprint_area():
                new_area = footprint.scale(n).area
                n += 0.001
        else:
            new_area = footprint.area
            n = 0.99
            while new_area > self._typical_footprint_area():
                new_area = footprint.scale(n).area
                n -= 0.001
        footprint = footprint.scale(n)

        return footprint

    def _base_room(
        self,
    ) -> Room:
        """Create single ground-floor room."""

        floor_face = Face3D(
            boundary=[Point3D(*i.to_array()) for i in self._footprint().vertices]
        )
        ceiling_face = floor_face.move(
            moving_vec=Vector3D(0, 0, self.floor_to_floor_height)
        ).flip()

        wall_faces = [
            Face3D.from_extrusion(
                line_segment=i,
                extrusion_vector=Vector3D(0, 0, self.floor_to_floor_height),
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

        directional_glazing_ratio = {
            k: self.glazing_ratio[i]
            for i, k in enumerate(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        }

        for face in room.walls:
            face: Face
            if isinstance(face.boundary_condition, Outdoors):
                # get face cardinal direction
                _dir = cardinality(angle_from_north(face.normal), directions=8)
                face.apertures_by_ratio(directional_glazing_ratio[_dir], 0.1)

        return room

    def _base_model(self) -> Model:
        """Create the base model (prior to assignment of program and fabric 
        properties) for the simulation."""

        base_room = self._base_room()

        rooms = [base_room]
        for i in range(self.average_num_floors):
            if i == 0:
                continue
            new_room = base_room.duplicate()
            new_room.move(moving_vec=Vector3D(0, 0, self.floor_to_floor_height * i))
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
        context_height = self._typical_height() * 0.75
        shades = []
        if context_distance < 1000 and context_height > 0:
            floor_face = base_room.floors[0].geometry
            points = np.array([i.to_array()[:2] for i in floor_face.vertices])
            hull = ConvexHull(points, qhull_options="QJ")
            hull_pts = np.stack([points[hull.vertices, 0], points[hull.vertices, 1]]).T
            context_base = Polygon2D.from_array(
                point_array=[Point2D(*i) for i in hull_pts]
            ).offset(-context_distance)
            shades = []
            # NOTE - context shade transmissivity could be added here, though proximity is probably enough
            for segment in context_base.segments:
                _shd = Shade(
                    identifier="context_shade",
                    geometry=Face3D.from_extrusion(
                        LineSegment3D.from_line_segment2d(segment),
                        extrusion_vector=Vector3D(0, 0, context_height),
                    ),
                )
                shades.append(_shd)

        # create model from rooms, and solve adjacencies
        base_model = Model.from_objects(
            identifier=self.name, objects=rooms + shades
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
    
    # Methods to obtain construction and program data
    # def construction_set(
    #     self, epw: EPW, construction_type: ConstructionType, vintage: Vintage
    # ):
    #     """Return the construction set for the building type."""

    #     return construction_set_by_identifier(
    #         f"{vintage.value}::ClimateZone{int(epw.ashrae_climate_zone[0])}::{construction_type.value}"
    #     )

    def program(self):
        """Return the program for the building type."""
        try:
            return building_program_type_by_identifier(self.hb_name)
        except (KeyError, ValueError) as exc:
            raise NotImplementedError(
                "This method must be overridden in a subclass."
            ) from exc

    def annual_hourly_load(self) -> pd.DataFrame:
        """Get time-series profiles for the program assigned to this building."""

        program = self.program()

        collections = []

        try:
            gas_equipment_collection = (
                program.gas_equipment.schedule.data_collection()
                * program.gas_equipment.watts_per_area
            )
            collections.append(
                pd.Series(
                    data=gas_equipment_collection.values,
                    index=pd.DatetimeIndex(
                        gas_equipment_collection.header.analysis_period.datetimes
                    ),
                    name="Gas Equipment (W/m2)",
                )
            )
        except AttributeError:
            pass

        try:
            people_collection = (
                program.people.occupancy_schedule.data_collection()
                * program.people.area_per_person
            )
            collections.append(
                pd.Series(
                    data=people_collection.values,
                    index=pd.DatetimeIndex(
                        people_collection.header.analysis_period.datetimes
                    ),
                    name="People (m2/person)",
                )
            )
        except AttributeError:
            pass

        try:
            lighting_collection = (
                program.lighting.schedule.data_collection()
                * program.lighting.watts_per_area
            )
            collections.append(
                pd.Series(
                    data=lighting_collection.values,
                    index=pd.DatetimeIndex(
                        lighting_collection.header.analysis_period.datetimes
                    ),
                    name="Lighting (W/m2)",
                )
            )
        except AttributeError:
            pass

        try:
            electric_equipment_collection = (
                program.electric_equipment.schedule.data_collection()
                * program.electric_equipment.watts_per_area
            )
            collections.append(
                pd.Series(
                    data=electric_equipment_collection.values,
                    index=pd.DatetimeIndex(
                        electric_equipment_collection.header.analysis_period.datetimes
                    ),
                    name="Electric Equipment (W/m2)",
                )
            )
        except AttributeError:
            pass

        try:
            htg_collection = program.setpoint.heating_schedule.data_collection()
            collections.append(
                pd.Series(
                    data=htg_collection.values,
                    index=pd.DatetimeIndex(
                        htg_collection.header.analysis_period.datetimes
                    ),
                    name="Heating Setpoint (C)",
                )
            )
        except AttributeError:
            pass

        try:
            clg_collection = program.setpoint.cooling_schedule.data_collection()
            collections.append(
                pd.Series(
                    data=clg_collection.values,
                    index=pd.DatetimeIndex(
                        clg_collection.header.analysis_period.datetimes
                    ),
                    name="Cooling Setpoint (C)",
                )
            )
        except AttributeError:
            pass

        try:
            hum_collection = program.setpoint.humidifying_schedule.data_collection()
            collections.append(
                pd.Series(
                    data=hum_collection.values,
                    index=pd.DatetimeIndex(
                        hum_collection.header.analysis_period.datetimes
                    ),
                    name="Humidifying Setpoint (%)",
                )
            )
        except AttributeError:
            pass

        try:
            dehum_collection = program.setpoint.dehumidifying_schedule.data_collection()
            collections.append(
                pd.Series(
                    data=dehum_collection.values,
                    index=pd.DatetimeIndex(
                        dehum_collection.header.analysis_period.datetimes
                    ),
                    name="Dehumidifying Setpoint (%)",
                )
            )
        except AttributeError:
            pass

        return pd.concat(collections, axis=1)


class ResidentialLowrise(BuildingType):
    """
    A low-rise residential building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Low-rise Residential",
            gross_floor_area=120,
            floors=2,
            floor_to_floor_height=3.3,
            glazing_ratio=[0.4] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name=None,
            terrain_type=TerrainType.SUBURBS,
            metadata=None
        )

    def program(self) -> ProgramType:
        # TODO - add custom program for residential lowrise
        dummy_program = "MidriseApartment"
        logger.warn(
            '%s uses the Honeybee building type "%s", and needs to be changed to reflect this building typology better.',
            self.name,
            dummy_program,
        )
        return building_program_type_by_identifier(dummy_program)


class ResidentialApartmentHighrise(BuildingType):
    """
    A high-rise apartment building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="High-rise Apartment",
            hb_name="HighriseApartment",
            gross_floor_area=12000,
            floors=10,
            floor_to_floor_height=3.5,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class ResidentialApartmentMidrise(BuildingType):
    """
    A mid-rise apartment building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Mid-rise Apartment",
            hb_name="MidriseApartment",
            gross_floor_area=3100,
            floors=4,
            floor_to_floor_height=3.5,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class College(BuildingType):
    """
    A college building. Also representative of universities and other higher education buildings.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="College",
            gross_floor_area=5000,
            floors=2,
            floor_to_floor_height=3.8,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name=None,
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class Courthouse(BuildingType):
    """
    A courthouse building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Courthouse",
            gross_floor_area=5000,
            floors=1,
            floor_to_floor_height=4,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="Courthouse",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class DataCenterLargeHighITE(BuildingType):
    """
    A large data center with a high information technology equipment (ITE) density.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Large Data Center (High ITE)",
            gross_floor_area=8000,
            floors=1,
            floor_to_floor_height=4,
            glazing_ratio=[0.05] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="LargeDataCenterHighITE",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class DataCenterLargeLowITE(BuildingType):
    """
    A large data center with a low information technology equipment (ITE) density.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Large Data Center (Low ITE)",
            gross_floor_area=8000,
            floors=1,
            floor_to_floor_height=4,
            glazing_ratio=[0.05] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="LargeDataCenterLowITE",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class DataCenterSmallHighITE(BuildingType):
    """
    A small data center with a high information technology equipment (ITE) density.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Small Data Center (High ITE)",
            gross_floor_area=2000,
            floors=1,
            floor_to_floor_height=4,
            glazing_ratio=[0.05] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="SmallDataCenterHighITE",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class DataCenterSmallLowITE(BuildingType):
    """
    A small data center with a low information technology equipment (ITE) density.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Small Data Center (Low ITE)",
            gross_floor_area=2000,
            floors=1,
            floor_to_floor_height=4,
            glazing_ratio=[0.05] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="SmallDataCenterLowITE",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class Hospital(BuildingType):
    """
    A hospital building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Hospital",
            gross_floor_area=22000,
            floors=3,
            floor_to_floor_height=3.8,
            glazing_ratio=[0.2] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="Hospital",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class HotelLarge(BuildingType):
    """
    A large hotel building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Large Hotel",
            gross_floor_area=11000,
            floors=5,
            floor_to_floor_height=3.8,
            glazing_ratio=[0.2] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="LargeHotel",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class HotelSmall(BuildingType):
    """
    A small hotel building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Small Hotel",
            gross_floor_area=4000,
            floors=3,
            floor_to_floor_height=3.8,
            glazing_ratio=[0.2] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="SmallHotel",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class Laboratory(BuildingType):
    """
    A laboratory building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Laboratory",
            gross_floor_area=1000,
            floors=1,
            floor_to_floor_height=3.8,
            glazing_ratio=[0.2] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="Laboratory",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class OfficeLarge(BuildingType):
    """
    A large office building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Large Office",
            gross_floor_area=45000,
            floors=10,
            floor_to_floor_height=3.8,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="LargeOffice",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class OfficeMedium(BuildingType):
    """
    A medium office building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Medium Office",
            gross_floor_area=5000,
            floors=3,
            floor_to_floor_height=3.8,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="MediumOffice",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class OfficeSmall(BuildingType):
    """
    A small office building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Small Office",
            gross_floor_area=500,
            floors=1,
            floor_to_floor_height=3.8,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="SmallOffice",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class Outpatient(BuildingType):
    """
    An outpatient building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Outpatient",
            gross_floor_area=3800,
            floors=1,
            floor_to_floor_height=3.8,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="Outpatient",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class RestaurantFullService(BuildingType):
    """
    A full-service restaurant building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Full-service Restaurant",
            gross_floor_area=500,
            floors=1,
            floor_to_floor_height=4,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="FullServiceRestaurant",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class RestaurantQuickService(BuildingType):
    """
    A quick-service restaurant building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Quick-service Restaurant",
            gross_floor_area=200,
            floors=1,
            floor_to_floor_height=4,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="QuickServiceRestaurant",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class Retail(BuildingType):
    """
    A retail building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Retail",
            gross_floor_area=2300,
            floors=1,
            floor_to_floor_height=4,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="Retail",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class SchoolPrimary(BuildingType):
    """
    A primary school building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Primary School",
            gross_floor_area=6900,
            floors=1,
            floor_to_floor_height=4,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="PrimarySchool",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class SchoolSecondary(BuildingType):
    """
    A secondary school building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Secondary School",
            gross_floor_area=19600,
            floors=3,
            floor_to_floor_height=4,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="SecondarySchool",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class StripMall(BuildingType):
    """
    A strip mall building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Strip Mall",
            gross_floor_area=10000,
            floors=2,
            floor_to_floor_height=4.25,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="StripMall",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class Supermarket(BuildingType):
    """
    A supermarket building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Supermarket",
            gross_floor_area=4200,
            floors=1,
            floor_to_floor_height=4.5,
            glazing_ratio=[0.05] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="Supermarket",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )

    def program(self):
        """A custom program for the supermarket building type.

        Sources:
            Floor area distribution for end-use breakdowns is based on the following sources:
            - ...

        TODO:
            - Update with a more considered program definition.
              This method is here mainly as a way to show that other program
              definitions can be used loading from JSON files.
        """

        # load the custom program source data
        custom_program = Path(__file__).parent / "custom_programs" / "NCM_A1A2.json"
        with open(custom_program, "r") as fp:
            data = json.load(fp)

        # set the proportions for each program type
        mix_dict = {
            "NCM_A1A2_Circulation": 0.0512161974754,
            "NCM_A1A2_FoodPrep": 0.0454693405569,
            "NCM_A1A2_Plant": 0.0548159093375,
            "NCM_A1A2_Sales": 0.519770059612,
            "NCM_A1A2_SalesChill": 0.0899628770905,
            "NCM_A1A2_Toilet": 0.0384789041835,
            "NCM_A1A2_Office": 0.0384789041835,
            "NCM_A1A2_RetWareSales": 0.129281081507,
            "NCM_A1A2_RetWareSalesChill": 0.0325267260542,
        }
        programs, ratios = [], []
        for sub_program, proportion in mix_dict.items():
            programs.append(ProgramType.from_dict(data[sub_program]))
            ratios.append(proportion)
        bld_program = ProgramType.average(clean_string(str(self)), programs, ratios)
        bld_program.lock()
        return bld_program


class Warehouse(BuildingType):
    """
    A warehouse building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Warehouse",
            gross_floor_area=4800,
            floors=1,
            floor_to_floor_height=4.5,
            glazing_ratio=[0.05] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name="Warehouse",
            terrain_type=TerrainType.URBAN,
            metadata=None
        )


class ConcertHall(BuildingType):
    """
    A concert hall building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Concert Hall",
            gross_floor_area=18000,
            floors=2,
            floor_to_floor_height=5,
            glazing_ratio=[0.05] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name=None,
            terrain_type=TerrainType.URBAN,
            metadata=None
        )

    def program(self) -> ProgramType:
        """The custom program for the concert hall building type.

        Source:
            AECOM. Cost Model: New-Build Concert Halls. LM00093-0817-v2.0, July 2017.

        TODO:
            - Update with a more considered program definition.
        """
        bld_mix_dict = {
            "2019::SecondarySchool::Auditorium": 0.20556,
            "2019::Courthouse::Entrance Lobby": 0.11111,
            "2019::QuickServiceRestaurant::Kitchen": 0.02778,
            "2019::Retail::Point_of_Sale": 0.01944,
            "2019::Courthouse::Restrooms": 0.01111,
            "2019::SecondarySchool::Cafeteria": 0.02222,
            "2019::Courthouse::Office": 0.06944,
            "2019::College::Media Center": 0.04444,
            "2019::Courthouse::Storage": 0.10278,
            "2019::Hospital::PhysTherapy": 0.05278,
            "2019::Courthouse::Corridor": 0.16667,
            "2019::Courthouse::Service Shaft": 0.02,
            "2019::SecondarySchool::Mechanical": 0.1,
            "2019::Courthouse::Plenum": 0.04667,
        }
        progs, ratios = [], []
        for key, val in bld_mix_dict.items():
            progs.append(program_type_by_identifier(key))
            ratios.append(val)
        bld_program = ProgramType.average(clean_string(self.name), progs, ratios)
        bld_program.lock()
        return bld_program


class PhysicalFitnessExercise(BuildingType):
    """
    A physical fitness center (exercise) building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Physical Fitness (Exercise)",
            gross_floor_area=400,
            floors=1,
            floor_to_floor_height=3.5,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name=None,
            terrain_type=TerrainType.URBAN,
            metadata=None
        )

    def program(self) -> ProgramType:
        """The custom program for the physical fitness exercise building type.

        Source:
            https://www.wbdg.org/space-types/physical-fitness-exercise-room

        TODO:
            - Update with a more considered program definition.
        """
        # from https://www.wbdg.org/space-types/physical-fitness-exercise-room
        bld_mix_dict = {
            "2019::College::Entrance Lobby": 0.01307,
            "2019::Hospital::PhysTherapy": 0.13725,
            "2019::LargeOffice::Restroom": 0.0915,
            "2019::SecondarySchool::Gym": 0.70589,
            "2019::College::Storage": 0.05229,
        }
        progs, ratios = [], []
        for key, val in bld_mix_dict.items():
            progs.append(program_type_by_identifier(key))
            ratios.append(val)
        bld_program = ProgramType.average(clean_string(self.name), progs, ratios)
        bld_program.lock()
        return bld_program


class PhysicalFitnessEvents(BuildingType):
    """
    A physical fitness center (events) building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Physical Fitness (Events)",
            gross_floor_area=29000,
            floors=1,
            floor_to_floor_height=5,
            glazing_ratio=[0.05] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name=None,
            terrain_type=TerrainType.URBAN,
            metadata=None
        )

    def program(self) -> ProgramType:
        """The custom program for the physical fitness events building type.

        Source:
            https://www.wbdg.org/space-types/auditorium

        TODO:
            - Update with a more considered program definition.
        """
        # from https://www.wbdg.org/space-types/auditorium
        bld_mix_dict = {
            "2019::Courthouse::Entrance Lobby": 0.19473,
            "2019::Courthouse::Storage": 0.0549,
            "2019::SecondarySchool::Cafeteria": 0.0244,
            "2019::SecondarySchool::Library": 0.0183,
            "2019::SecondarySchool::Auditorium": 0.43924,
            "2019::SecondarySchool::Gym": 0.14642,
            "2019::College::Media Center": 0.08541,
            "2019::Courthouse::Restrooms": 0.0366,
        }
        progs, ratios = [], []
        for key, val in bld_mix_dict.items():
            progs.append(program_type_by_identifier(key))
            ratios.append(val)
        bld_program = ProgramType.average(clean_string(self.name), progs, ratios)
        bld_program.lock()
        return bld_program


class PlaceOfWorship(BuildingType):
    """
    A place of worship building. Typical of a congregational space.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Place of Worship",
            gross_floor_area=4000,
            floors=1,
            floor_to_floor_height=5,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name=None,
            terrain_type=TerrainType.URBAN,
            metadata=None
        )

    def program(self) -> ProgramType:
        """The custom program for the place of worship building type.

        Source:
            https://www.wbdg.org/space-types/place-worship

        TODO:
            - Update with a more considered program definition.
        """
        bld_mix_dict = {
            "2019::Courthouse::Courtroom": 0.49690,
            "2019::Courthouse::Storage": 0.05797,
            "2019::Courthouse::Office": 0.10352,
            "2019::Courthouse::Utility": 0.16770,
            "2019::Courthouse::Corridor": 0.17391,
        }
        progs, ratios = [], []
        for key, val in bld_mix_dict.items():
            progs.append(program_type_by_identifier(key))
            ratios.append(val)
        bld_program = ProgramType.average(clean_string(self.name), progs, ratios)
        bld_program.lock()
        return bld_program


class LightIndustry(BuildingType):
    """
    A light industry building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Light Industry",
            gross_floor_area=1500,
            floors=1,
            floor_to_floor_height=4.5,
            glazing_ratio=[0.05] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name=None,
            terrain_type=TerrainType.URBAN,
            metadata=None
        )

    def program(self) -> ProgramType:
        """The custom program for the light industry building type.

        Source:
            https://www.wbdg.org/space-types/light-industrial

        TODO:
            - Update with a more considered program definition.
        """
        bld_mix_dict = {
            "2019::Warehouse::Office": 0.01125,
            "2019::Warehouse::Bulk": 0.53622,
            "2019::SmallDataCenterLowITE::ComputerRoom": 0.34005,
            "2019::Warehouse::Fine": 0.11248,
        }
        progs, ratios = [], []
        for key, val in bld_mix_dict.items():
            progs.append(program_type_by_identifier(key))
            ratios.append(val)
        bld_program = ProgramType.average(clean_string(self.name), progs, ratios)
        bld_program.lock()
        return bld_program


class ParkingBasement(BuildingType):
    """
    A parking garage.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Parking Basement",
            gross_floor_area=3500,
            floors=2,
            floor_to_floor_height=3.2,
            glazing_ratio=[0.05] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name=None,
            terrain_type=TerrainType.URBAN,
            metadata=None
        )

    def program(self) -> ProgramType:
        """The custom program for the parking basement building type.

        Source:
            ... estimates?

        TODO:
            - Update with a more considered program definition.
        """
        # FIXME - building space breakdown here is an estimate and needs to be updated with a more considered program definition
        bld_mix_dict = {
            "2019::Courthouse::Parking": 0.95000,
            "2019::SuperMarket::Elec/MechRoom": 0.05000,
        }
        progs, ratios = [], []
        for key, val in bld_mix_dict.items():
            progs.append(program_type_by_identifier(key))
            ratios.append(val)
        bld_program = ProgramType.average(clean_string(self.name), progs, ratios)
        bld_program.lock()
        return bld_program


class Library(BuildingType):
    """
    A library building.

    Sources:
        https://www.energy.gov/eere/buildings/commercial-reference-buildings

    """

    def __init__(self):
        super().__init__(
            name="Library",
            gross_floor_area=1200,
            floors=1,
            floor_to_floor_height=3.5,
            glazing_ratio=[0.3] * 8,
            skylight_ratio=0,
            building_form=BuildingForm.CUBOID,
            aspect_ratio=1,
            rotation=0,
            hb_name=None,
            terrain_type=TerrainType.URBAN,
            metadata=None
        )

    def program(self) -> ProgramType:
        """The custom program for the library building type.

        Source:
            https://www.wbdg.org/space-types/library

        TODO:
            - Update with a more considered program definition.
        """
        bld_mix_dict = {
            "2019::College::Entrance Lobby": 0.0594,
            "2019::Courthouse::Jury Deliberation": 0.02121,
            "2019::LargeOffice::PrintRoom": 0.09418,
            "2019::Courthouse::Office": 0.18244,
            "2019::College::Media Center": 0.1171,
            "2019::Courthouse::Library": 0.39839,
            "2019::College::Lounge": 0.11031,
            "2019::Courthouse::Utility": 0.01697,
        }
        progs, ratios = [], []
        for key, val in bld_mix_dict.items():
            progs.append(program_type_by_identifier(key))
            ratios.append(val)
        bld_program = ProgramType.average(clean_string(self.name), progs, ratios)
        bld_program.lock()
        return bld_program


class BuildingTypes(Enum):
    """The building types available in the masterplan energy demand model."""

    COLLEGE = College()
    COURTHOUSE = Courthouse()
    DATA_CENTER_LARGE_HIGH_ITE = DataCenterLargeHighITE()
    DATA_CENTER_LARGE_LOW_ITE = DataCenterLargeLowITE()
    DATA_CENTER_SMALL_HIGH_ITE = DataCenterSmallHighITE()
    DATA_CENTER_SMALL_LOW_ITE = DataCenterSmallLowITE()
    HOSPITAL = Hospital()
    HOTEL_LARGE = HotelLarge()
    HOTEL_SMALL = HotelSmall()
    LABORATORY = Laboratory()
    OFFICE_LARGE = OfficeLarge()
    OFFICE_MEDIUM = OfficeMedium()
    OFFICE_SMALL = OfficeSmall()
    OUTPATIENT = Outpatient()
    RESTAURANT_FULL_SERVICE = RestaurantFullService()
    RESTAURANT_QUICK_SERVICE = RestaurantQuickService()
    RETAIL = Retail()
    SCHOOL_PRIMARY = SchoolPrimary()
    SCHOOL_SECONDARY = SchoolSecondary()
    STRIP_MALL = StripMall()
    SUPERMARKET = Supermarket()
    WAREHOUSE = Warehouse()
    RESIDENTIAL_LOWRISE = ResidentialLowrise()
    RESIDENTIAL_APARTMENT_HIGHRISE = ResidentialApartmentHighrise()
    RESIDENTIAL_APARTMENT_MIDRISE = ResidentialApartmentMidrise()
    CONCERT_HALL = ConcertHall()
    PHYSICAL_FITNESS_EXERCISE = PhysicalFitnessExercise()
    PHYSICAL_FITNESS_EVENTS = PhysicalFitnessEvents()
    PLACE_OF_WORSHIP = PlaceOfWorship()
    LIGHT_INDUSTRY = LightIndustry()
    PARKING_BASEMENT = ParkingBasement()
    LIBRARY = Library()


def generate_excel_template(out_path: Path | str) -> Path:
    """Create a template Excel file containing all predefined buildings and their properties. 

    Args:
        out_path (Path | str): The location where the template should be saved

    Returns:
        Path: The Excel template file
    """
    
    out_path = Path(out_path)


    raise NotImplementedError("Not yet coded!")
    
    return None