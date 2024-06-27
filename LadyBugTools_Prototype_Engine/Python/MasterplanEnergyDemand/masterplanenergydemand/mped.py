# region: IMPORTS
# pylint: disable=E0401

import concurrent
import concurrent.futures
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.typing import valid_string
from ladybug.epw import EPW
from pydantic import BaseModel, Field, root_validator

from .config import logger
from .enums import BuildingType
from .typology import Typology
from .util import random_id

# pylint: enable=E0401
# endregion: IMPORTS


class Masterplan(BaseModel):
    """A masterplan containing a set of building typologies, in the location represented by and EPW file."""

    identifier: str = Field(description="A unique identifier for the masterplan.")
    epw_file: Path = Field(
        description="The EPW file representing the location of the masterplan."
    )
    typologies: list[Typology] = Field(
        description="A list of building typologies in the masterplan.",
        unique_items=True,
    )

    @root_validator(pre=False)
    def validate_atts(cls, values):
        """Validate the attributes."""

        # Check that the identifier is valid
        identifier = values.get("identifier")
        valid_string(identifier)

        # Check that the EPW file exists.
        epw_file = values.get("epw_file")
        if not Path(epw_file).exists():
            raise ValueError(f"The EPW file {epw_file} does not exist.")

        return values

    @classmethod
    def from_excel(cls, excel_file: Path, sheet_name: str) -> "Masterplan":
        """Create a masterplan from an Excel file."""

        # check that excel file exists
        if not Path(excel_file).exists():
            raise ValueError(f"The Excel file {excel_file} does not exist.")

        # Read the Excel file
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            df = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
                engine="openpyxl",
                header=None,
                index_col=0,
            )

        # if number of columns is less than 2, raise an error
        if len(df.columns) < 2:
            raise ValueError(f"Excel file {excel_file} does not have enough columns.")

        # if each column doesnt have the same epw_file value, raise an error
        epw_file = Path(df[2]["epw_file"])
        for n, (_, s) in enumerate(df.items()):
            if n == 0:
                continue
            if Path(s["epw_file"]) != epw_file:
                raise ValueError(
                    f"EPW file for {s['identifier']} does not match the file used for the rest of the masterplan ({epw_file})."
                )

        # Get the typologies
        typologies: list[Typology] = []
        for n, (_, s) in enumerate(df.items()):
            if n == 0:
                continue

            typ = Typology.parse_obj_extended(s.to_dict(), replace_null_with_defaults=True, building_type=BuildingType(s["building_type"]))

            typologies.append(typ)

        return cls(identifier=sheet_name, epw_file=epw_file, typologies=typologies)

    @classmethod
    def random(cls, n_typologies: int = 5) -> "Masterplan":
        """Create a random masterplan."""

        # reference the test EPW here ... this is bad practice, but meh
        epw_file = Path(__file__).absolute().parent / "test" / "test.epw"

        return cls(
            identifier=random_id(),
            epw_file=epw_file,
            typologies=[Typology.random() for _ in range(n_typologies)],
        )

    @property
    def epw(self) -> EPW:
        """Return the EPW object for the masterplan."""
        return EPW(self.epw_file)

    @property
    def simulation_directory(self) -> Path:
        """Return the directory for the simulation."""
        return Path(hb_folders.default_simulation_folder) / self.identifier

    @property
    def gfa_table(self) -> dict[str, float]:
        """Return a table of the GFA for each building typology in the masterplan."""
        return {
            typology.identifier: typology.total_area for typology in self.typologies
        }

    def population(self) -> pd.DataFrame:
        """Estimate the population of the masterplan.

        Note:
            This is a simple estimate based on the total number of occupants in each building typology at any time.
        """
        return pd.concat([typ.occupants for typ in self.typologies], keys=[typ.identifier for typ in self.typologies], axis=1)

    def results(self) -> None:
        """Simulate the energy demands of the masterplan."""

        # Create a directory for the simulation
        directory = self.simulation_directory
        directory.mkdir(exist_ok=True, parents=True)

        # concurrently simulate each typology
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    obj.load_results,
                    self.epw,
                    directory,
                )
                for obj in self.typologies
            ]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        results = dict(zip([typ.identifier for typ in self.typologies], results))

        # make absolute if requested
        denormalised_results = {}
        for result_k, result_v in results.items():
            new_df = []
            for col, vals in result_v.items():
                if col.endswith("/m2)"):
                    new_vals = vals * self.gfa_table[result_k]
                    new_name = col.replace("/m2)", ")")
                    new_df.append(new_vals.rename(new_name))
                else:
                    new_df.append(vals)
            denormalised_results[result_k] = pd.concat(new_df, axis=1)

        # convert to dataframe
        df = pd.concat(
            denormalised_results,
            axis=1,
            keys=[typ.identifier for typ in self.typologies],
        )

        # save to file
        df.to_csv(directory / "results.csv")

        return df

    def energy_consumption(self) -> pd.DataFrame:
        """Get the energy consumption of the typology."""

        df = self.results()

        res = []
        for typology in self.typologies:
            res.append(
                typology.energy_consumption(
                    self.epw, self.simulation_directory, df[typology.identifier]
                )
            )

        return pd.concat(res, axis=1, keys=[typ.identifier for typ in self.typologies])
