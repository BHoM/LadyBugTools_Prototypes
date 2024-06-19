# pylint: disable=E0401
from pathlib import Path

from ladybug.epw import EPW

# pylint: enable=E0401

EPW_OBJ = EPW(Path(__file__).absolute().parent / "test.epw")
