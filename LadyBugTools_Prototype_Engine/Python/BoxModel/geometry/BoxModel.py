import os
import ladybug.epw as epw
import ladybug.wea as wea
from dataclasses import dataclass, field
from pathlib import Path
from honeybee.model import Model
from ladybug.wea import Wea
from .geom import BoxModelSensorGrid
from ..simulation.daylight_simulation import DaylightSimulation
from pathlib import Path
from ..file_utils import make_folder_if_not_exist
from ..results.daylight_results import DaylightSimResults
from ..results.daylight_plotting import DaylightPlot, generate_zip

def EpwToWea(epw_file):
    # epw_file = EPW(epw_filepath)

    # print(epw_file)
    epw_data = epw.EPW(epw_file)
    wea_file = wea.Wea.from_epw_file(epw_file)
    
    return wea_file