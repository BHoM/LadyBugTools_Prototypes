import os
import ladybug.epw as epw
import ladybug.wea as wea
from dataclasses import dataclass, field
from pathlib import Path
from honeybee.model import Model
from ladybug.wea import Wea

def EpwToWea(epw_file):
    epw_data = epw.EPW(epw_file)
    wea_file = wea.Wea.from_epw_file(epw_file)
    
    return wea_file