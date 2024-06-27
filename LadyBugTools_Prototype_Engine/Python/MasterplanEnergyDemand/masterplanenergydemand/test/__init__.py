# pylint: disable=E0401
from pathlib import Path

from honeybee.config import folders as hb_folders
from ladybug.epw import EPW
from ladybug.futil import nukedir

# pylint: enable=E0401

EPW_PATH = Path(__file__).absolute().parent / "test.epw"
EPW_OBJ = EPW(EPW_PATH)

MPED_ID = "TestMPED"
TYPOLOGY_ID = "TestTypology"
SIMULATION_DIRECTORY = Path(hb_folders.default_simulation_folder) / MPED_ID

EXCEL_PATH = Path(__file__).absolute().parent / "test.xlsx"

# # remove old tests from simulation directory
# TODO - uncomment for prod
# if SIMULATION_DIRECTORY.exists():
#     nukedir(SIMULATION_DIRECTORY, rmdir=True)
