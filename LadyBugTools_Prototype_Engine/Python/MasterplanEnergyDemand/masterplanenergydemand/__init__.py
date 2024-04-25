"""Top-level package for MasterplanEnergyDemand."""

__author__ = """Tristan Gerrish"""
__email__ = "tristan.gerrish@burohappold.com"
__version__ = "0.1.0"

# pylint: disable=E0401
import getpass
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,  # change to CRITICAL to silence logging (mostly)
    format="%(levelname)s - %(message)s",
)

# pylint: enable=E0401

os.environ["HOME"] = (Path("C:/Users/") / getpass.getuser()).as_posix()

# plt.style.use(Path(__file__).parent / "data" / "bhom.mplstyle")

# plt.style.use(
#     r"C:\Users\tgerrish\Buro Happold\P060941 Jumeirah Central Masterplan - Sustainability and Microclimate\JumeirahCentralMasterplan_SOM.matplotlibrc"
# )
# plt.rc("font", family="Swis721 Lt BT")

FORMATTING = pd.read_json(
    Path(__file__).parent / "data" / "color_config.json", orient="index"
)
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

DPI = 200
FIGSIZE_RECTANGLE = (12, 5)
FIGSIZE_SQUARE = (8, 8)
