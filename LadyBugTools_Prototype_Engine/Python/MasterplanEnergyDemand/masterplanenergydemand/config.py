"""Top-level package for MasterplanEnergyDemand."""

__author__ = """Tristan Gerrish"""
__email__ = "tristan.gerrish@burohappold.com"
__version__ = "0.1.0"

# pylint: disable=E0401
import getpass
import logging
import os
from logging.config import fileConfig
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod

# pylint: enable=E0401

# PATHS
DATA_PATH = (Path(__file__).parent / "data").absolute()
os.environ["HOME"] = (Path("C:/Users/") / getpass.getuser()).as_posix()

# LOGGING
fileConfig(DATA_PATH / "logging_config.ini")
logger = logging.getLogger(__name__.split(".", maxsplit=1)[0])
logger.setLevel(logging.INFO)  # set to DEBUG or INFO for debugging, WARNING for production, and FATAL to disable

# CONSTANTS
INDEX = pd.to_datetime(AnalysisPeriod().datetimes)

# DATASETS
SRI_DATA = pd.read_csv(DATA_PATH / "sri_data.csv", header=0)
DEFAULT_SYSTEMS = pd.read_csv(DATA_PATH / "default_systems.csv", header=0, index_col=None)

# PLOT STYLING
plt.style.use(DATA_PATH / "bhom.mplstyle")
# plt.rc("font", family="Swis721 Lt BT")
DPI = 200
FIGSIZE_RECTANGLE = (12, 5)
FIGSIZE_SQUARE = (8, 8)

# COLOR DEFAULTS
colour_defaults = {
    "Heating": "#D2424C",
    "Cooling": "#5C6DD8",
    "Lighting": "#B2B042",
    "Solar": "#DF9641",
    "Hot Water": "#C75DB0",
    "Electric Equipment": "#64B546",
    "Lifts": "#8B74B8",
    "Pumps": "#5A7936",
    "Fans": "#5D9AD5",
    "Storage": "#A3793F",
    "People": "#C26C80",
    "Mechanical Ventilation": "#3DBBB8",
    "Window Conduction": "#9C9C9C",
    "Opaque Conduction": "#3F3F3F",
    "Infiltration": "#5BB57C",
    "Gas Equipment": "#CD3E78",
    "Zone Mean Air Temperature": "#FF9696",
    "Zone Mean Radiant Temperature": "#FF9100",
    "Zone Air Relative Humidity": "#ABC9FF",
    "Outdoor Air Dry Bulb Temperature": "#FF0000",
    "Outdoor Air Relative Humidity": "#002FFF",
    "Heating Setpoint": "#FF00F2",
    "Cooling Setpoint": "#86EDFF",
    "Humidifying Setpoint": "#00B52A",
    "Dehumidifying Setpoint": "#FF8C3A"
}

LOAD_BALANCE_TERMS = [
    "Heating (Wh)",
    "Solar (Wh)",
    "Service Hot Water (Wh)",
    "Electric Equipment (Wh)",
    "Lighting (Wh)",
    "People (Wh)",
    "Infiltration (Wh)",
    "Mechanical Ventilation (Wh)",
    "Opaque Conduction (Wh)",
    "Window Conduction (Wh)",
    "Cooling (Wh)",
    "Storage (Wh)",
    "Gas Equipment (Wh)",
]
