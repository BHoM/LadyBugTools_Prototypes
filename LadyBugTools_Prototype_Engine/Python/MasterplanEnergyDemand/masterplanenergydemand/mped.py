# region: IMPORTS
# pylint: disable=E0401
import concurrent
import concurrent.futures
import inspect
import json
import warnings
from enum import Enum
from pathlib import Path
from uuid import uuid4

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from honeybee.boundarycondition import Outdoors
from honeybee.config import folders as hb_folders
from honeybee.facetype import Floor, RoofCeiling, Wall
from honeybee.model import Face, Model, Shade
from honeybee.room import Room
from honeybee.typing import valid_string
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.constructionset import ConstructionSet
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.internalmass import InternalMass
from honeybee_energy.lib.constructionsets import (
    ConstructionSet, construction_set_by_identifier)
from honeybee_energy.lib.scheduletypelimits import (
    humidity, schedule_type_limit_by_identifier, temperature)
from honeybee_energy.programtype import ProgramType
from honeybee_energy.result.loadbalance import LoadBalance, SQLiteResult
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from honeybee_energy.simulation.parameter import (RunPeriod, ShadowCalculation,
                                                  SimulationControl,
                                                  SimulationOutput,
                                                  SimulationParameter,
                                                  SizingParameter)
from ladybug.wea import EPW, AnalysisPeriod, HourlyContinuousCollection
from ladybug_geometry.geometry2d import Point2D, Polygon2D, Vector2D
from ladybug_geometry.geometry3d import (Face3D, LineSegment3D, Point3D,
                                         Vector3D)
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull

from .config import logger
from .enums import TerrainType
from .form import Form

# pylint: enable=E0401
# endregion: IMPORTS



class BuildingFabric:
    """The fabric describing the thermal performance of ."""
    def __init__(self):
        pass

# method to create the program from the building type


class BuildingTypology:
    def __init__(self, form: Form, program: ProgramType):
        pass
    

class Masterplan:
    """A masterplan containing a set of building typologies, in the location represented by and EPW file."""
    identifier: str
    building_typologies: list[BuildingTypology]
    epw: EPW
    metadata: dict


