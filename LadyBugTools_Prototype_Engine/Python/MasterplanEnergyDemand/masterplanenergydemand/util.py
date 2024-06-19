# region: IMPORTS
# pylint: disable=E0401


from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ladybug.datacollection import BaseCollection
from ladybug.wea import Wea
from ladybug_geometry.geometry2d import Vector2D
from ladybug_geometry.geometry3d import Face3D, Vector3D
from matplotlib.colors import colorConverter
from sklearn.linear_model import LinearRegression

from .config import logger

# pylint: enable=E0401
# endregion: IMPORTS


def cardinality(direction_angle: float, directions: int = 16):
    """Returns the cardinal orientation of a given angle, where that angle is
    related to north at 0 degrees.

    Args:
        direction_angle (float):
            The angle to north in degrees (+Ve is interpreted as clockwise
            from north at 0.0 degrees).
        directions (int):
            The number of cardinal directions into which angles shall be
            binned (This value should be one of 4, 8, 16 or 32, and is centred
            about "north").

    Returns:
        int:
            The cardinal direction the angle represents.
    """

    if direction_angle > 360 or direction_angle < 0:
        raise ValueError(
            f"The angle entered ({direction_angle}) is beyond the normally expected range for an orientation in degrees."
        )

    cardinal_directions = {
        4: ["N", "E", "S", "W"],
        8: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        16: [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ],
        32: [
            "N",
            "NbE",
            "NNE",
            "NEbN",
            "NE",
            "NEbE",
            "ENE",
            "EbN",
            "E",
            "EbS",
            "ESE",
            "SEbE",
            "SE",
            "SEbS",
            "SSE",
            "SbE",
            "S",
            "SbW",
            "SSW",
            "SWbS",
            "SW",
            "SWbW",
            "WSW",
            "WbS",
            "W",
            "WbN",
            "WNW",
            "NWbW",
            "NW",
            "NWbN",
            "NNW",
            "NbW",
        ],
    }

    if directions not in cardinal_directions:
        raise ValueError(
            f'The input "directions" must be one of {list(cardinal_directions.keys())}.'
        )

    val = int((direction_angle / (360 / directions)) + 0.5)

    arr = cardinal_directions[directions]

    return arr[(val % directions)]


def angle_from_north(vector: Vector3D) -> float:
    """For a 3D vector, determine the clockwise angle to north at [0, 1, 0].

    Args:
        vector (Vector3D):
            A ladybug_geometry Vector3D object.

    Returns:
        float:
            The angle between vector and north in degrees clockwise from [0, 1].
    """

    north = Vector2D(0, 1)
    vec2d = Vector2D(vector.x, vector.y)

    return np.rad2deg(north.angle_clockwise(vec2d))


def face_orientation(face: Face3D) -> str:
    """Determine the cardinal orientation of a face.

    Args:
        face (Face3D):
            A ladybug_geometry Face3D object.

    Returns:
        str:
            The cardinal orientation of the face.
    """

    normal = face.normal
    angle = angle_from_north(normal)

    return cardinality(angle, 8)

def _log_message(property_name: str, value: Any, unit: str = "") -> None:
    logger.info(
        '> no "%s" provided, using default value of %s%s', property_name, value, unit
    )

def number_validator(value: float | int, prop_name: str, gt: float = None, ge: float = None, lt: float = None, le: float = None) -> None:
    if not isinstance(value, (float, int)):
        raise ValueError(f"{prop_name} must be a number")
    if gt is not None and ge is not None:
        raise ValueError("Both 'gt' and 'ge' cannot be provided.")
    if lt is not None and le is not None:
        raise ValueError("Both 'lt' and 'le' cannot be provided.")
    if gt is not None and value <= gt:
        raise ValueError(f"{prop_name} must be greater than {gt}")
    if ge is not None and value < ge:
        raise ValueError(f"{prop_name} must be greater than or equal to {ge}")
    if lt is not None and value >= lt:
        raise ValueError(f"{prop_name} must be less than {lt}")
    if le is not None and value > le:
        raise ValueError(f"{prop_name} must be less than or equal to {le}")

def list_of_nums_validator(value: Any, prop_name: str, length: int, ge: float = None, le: float = None, gt: float = None, lt: float = None) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{prop_name} must be a list or tuple")
    if len(value) != length:
        raise ValueError(f"{prop_name} must have {length} items")
    for i in value:
        number_validator(i, prop_name, gt, ge, lt, le)
        