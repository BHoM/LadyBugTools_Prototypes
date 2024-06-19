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

# pylint: enable=E0401

# PATHS
DATA_PATH = (Path(__file__).parent / "data").absolute()
os.environ["HOME"] = (Path("C:/Users/") / getpass.getuser()).as_posix()

# LOGGING
fileConfig(DATA_PATH / "logging_config.ini")
logger = logging.getLogger(__name__.split(".", maxsplit=1)[0])
logger.setLevel(logging.DEBUG)  # set to DEBUG or INFO for debugging, and WARNING for production

# PLOT STYLING
plt.style.use(DATA_PATH / "bhom.mplstyle")
# plt.rc("font", family="Swis721 Lt BT")
DPI = 200
FIGSIZE_RECTANGLE = (12, 5)
FIGSIZE_SQUARE = (8, 8)

# COLOR DEFAULTS
colour_defaults = {
    "Heating": "#d2424c",
    "Cooling": "#5c6dd8",
    "Lighting": "#b2b042",
    "Solar": "#df9641",
    "Service Hot Water": "#c75db0",
    "Electric Equipment": "#64b546",
    "Lifts": "#8b74b8",
    "Pumps": "#5a7936",
    "Fans": "#5d9ad5",
    "Storage": "#a3793f",
    "People": "#c26c80",
    "Mechanical Ventilation": "#3dbbb8",
    "Window Conduction": "#9C9C9C",
    "Opaque Conduction": "#3F3F3F",
    "Infiltration": "#5bb57c",
    "Gas Equipment": "#cd3e78",
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
FORMATTING = pd.DataFrame.from_dict(colour_defaults, orient="index")
FORMATTING.columns = ["color"]

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
