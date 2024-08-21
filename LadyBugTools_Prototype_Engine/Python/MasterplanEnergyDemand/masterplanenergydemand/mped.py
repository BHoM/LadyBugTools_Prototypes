# region: IMPORTS
# pylint: disable=E0401

import concurrent
import concurrent.futures
import copy
import inspect
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.typing import valid_string
from ladybug.epw import EPW, AnalysisPeriod
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, root_validator
from tqdm import tqdm

from .config import FIGSIZE_RECTANGLE, FIGSIZE_SQUARE, INDEX, logger
from .enums import (BuildingType, ConstructionType, EconomizerType,
                    TerrainType, Vintage)
from .plot import diurnal, duration_curve, pie, stacked_bar
from .typology import Typology
from .util import describe_analysis_period, get_color

# pylint: enable=E0401
# endregion: IMPORTS

RELOAD = True

class Masterplan(BaseModel):
    """A masterplan containing a set of building typologies, in the location represented by and EPW file."""

    identifier: str = Field(description="A unique identifier for the masterplan.")
    typologies: list[Typology] = Field(
        description="A list of building typologies in the masterplan.",
        unique_items=True,
    )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.identifier})"

    def __repr__(self) -> str:
        return self.__str__()

    # pylint: disable=no-self-argument
    @root_validator(pre=False)
    def validate_atts(cls, values):
        """Validate the attributes."""

        # Check that the identifier is valid
        identifier = values.get("identifier")
        valid_string(identifier)

        return values
    # pylint: enable=no-self-argument

    @classmethod
    def from_excel(
        cls, excel_file: Path, sheet_name: str, use_defaults: bool = True, parallel: bool = True
    ) -> "Masterplan":
        """Create a masterplan from an Excel file."""

        excel_file = Path(excel_file)

        logger.info(f"Creating Masterplan from {excel_file.stem}.")

        # check that excel file exists
        if not excel_file.exists():
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
        
        # remove the units columns
        df = df.drop(columns=[1])
        df.columns = range(len(df.columns))

        # check all typologies share the same EPW file
        epw_file = Path(df[0]["epw_file"])
        for n, (_, s) in enumerate(df.items()):
            if Path(s["epw_file"]) != epw_file:
                raise ValueError(
                    f"EPW file for {s['identifier']} does not match the file used for the rest of the masterplan ({epw_file})."
                )
        
        # overwrite the masterplan_identifier value with the name of this case
        df.loc[df.index == "masterplan_identifier"] = [[sheet_name] * len(df.columns)]

        # Get the typologies
        if (len(df.columns) > 3) and parallel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for n, (_, s) in enumerate(df.items()):
                    if n == 0:
                        continue
                    futures.append(
                        executor.submit(Typology.from_series, s, use_defaults)
                    )
                typologies = [future.result() for future in futures]
        else:
            typologies: list[Typology] = []
            for n, (_, s) in enumerate(df.items()):
                if n == 0:
                    continue
                typologies.append(
                    Typology.from_series(s, use_defaults=use_defaults)
                )

        obj = cls(identifier=sheet_name, typologies=typologies)

        return obj

    def to_excel(self, excel_file: Path, sheet_name: str = None) -> None:
        """Write the masterplan typologies to an Excel file.

        Args:
            excel_file: The path to the Excel file.
            sheet_name: The name of the sheet to write to. If None, the identifier of the masterplan will be used.

        Returns:
            None
        """

        if sheet_name is None:
            sheet_name = self.identifier

        # get metadata to add as additional sheet
        meta_df = []
        enums = [BuildingType, TerrainType, Vintage, EconomizerType, ConstructionType]
        for enm in enums:
            meta_df.append(pd.Series([i.value for i in enm], name=enm))
        meta_df = pd.concat(meta_df, axis=1)

        # get data for the masterplan
        data_df = self.df().reset_index()
        data_df.insert(
            1,
            "units",
            [
                Typology.__fields__[i].field_info.extra["unit"]
                for i in Typology.__fields__
            ],
        )
        data_df.set_index(keys=[data_df.columns[0], data_df.columns[1]], inplace=True)

        with pd.ExcelWriter(excel_file, engine="openpyxl", mode="w") as writer:
            data_df.to_excel(writer, sheet_name=sheet_name, header=False, index=True)
            meta_df.to_excel(writer, sheet_name="metadata", header=True, index=False)

        return None

    def df(self, extra: bool = False) -> pd.DataFrame:
        """Return the data for the masterplan."""

        df = pd.concat(
            [pd.Series(json.loads(typ.json())) for typ in self.typologies], axis=1
        )

        if extra:
            df = df.T
            df["number_of_buildings"] = [
                typ._number_of_buildings() for typ in self.typologies
            ]
            df["max_occupants_per_building"] = [
                typ.occupant_density * typ.typical_gfa
                for typ in self.typologies
            ]
            df = df.T

        return df

    @property
    def epw(self) -> EPW:
        """Return the EPW object for the masterplan."""
        return self.typologies[0].epw

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

    def eui(self) -> pd.DataFrame:
        """Get the EUI for each of the typologies.

        Args:
            annual: If True, return the annual EUI. If False, return the monthly EUI.

        Returns:
            A DataFrame with the EUI for each typology.
        """

        df = pd.concat(
            [
                typ.annual_eui(directory=self.simulation_directory)
                for typ in self.typologies
            ],
            axis=1,
            keys=[typ.identifier for typ in self.typologies],
        )

        return df

    def population(self, per_typology: bool = False) -> pd.DataFrame | pd.Series:
        """Estimate the population of the masterplan.

        Note:
            This is a simple estimate based on the total number of occupants in all buildings in the masterplan at any time.
        """

        df = pd.concat(
            [typ._occupants(per_building=False) for typ in self.typologies],
            keys=[typ.identifier for typ in self.typologies],
            axis=1,
        ).astype(int)

        if not per_typology:
            df = df.sum(axis=1)

        return df

    def simulate(self) -> None:
        """Simulate the energy demands of the masterplan."""

        # Create a directory for the simulation
        directory = self.simulation_directory
        directory.mkdir(exist_ok=True, parents=True)

        # get typologies as list
        typologies = copy.copy(self.typologies)

        # remove typologies that already have results
        for typology in self.typologies:
            if typology._sql_file_exists():
                typologies.remove(typology)

        if len(typologies) == 0:
            return None

        # concurrently simulate each typology
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    typology.run_all,
                )
                for typology in self.typologies
            ]
            _ = [future.result() for future in futures]

        return None

    def energy_consumption(self, combine_buildings: bool = False) -> pd.DataFrame:
        """Get the energy consumption of the typology.

        Args:
            normalise: If True, normalise the energy consumption to the total GFA of the masterplan.

        Returns:
            A DataFrame with the energy consumption of each typology.
        """

        # run the simulation first to get results
        self.simulate()

        # load each typology's results
        df = pd.concat(
            [
                typ.energy_consumption(
                    directory=self.simulation_directory,
                    normalised=False,
                    single_building=False,
                )
                for typ in self.typologies
            ],
            axis=1,
            keys=[typ.identifier for typ in self.typologies],
        )

        if combine_buildings:
            df = df.T.groupby(df.columns.get_level_values(1)).sum().T

        return df

    def external_conditions(self) -> pd.DataFrame:
        """Get the external conditions from the EPW file being used."""
        return self.typologies[0].external_conditions()

    def _all_data(self, normalised: bool = False) -> pd.DataFrame:
        """Get all room conditions for all typologies, and the their conditions.

        This method, when called also acts as a "do everything" method, as it
        will run the simulation and prepare datasets for onward methods.
        """

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    obj._all_data,
                    self.simulation_directory,
                    normalised,
                    False,
                    True,
                )
                for obj in self.typologies
            ]
            _ = [future.result() for future in futures]

        # combine data
        all_objects = [
            i._all_data(
                directory=self.simulation_directory,
                normalised=normalised,
                include_external=True,
            )
            for i in self.typologies
        ]
        keys = [typ.identifier for typ in self.typologies]
        df = pd.concat(all_objects, axis=1, keys=keys)

        return df
    
    def plot_annual_monthly(
        self,
        ax: plt.Axes = None,
        rule: str = "MS",
        label: bool = True,
        legend: bool = True,
    ) -> plt.Axes:
        """Plot the monthly energy consumption of the Masterplan."""

        logger.info(f"{self} - Plotting annual monthly energy consumption")

        df = self.energy_consumption(combine_buildings=True)

        if ax is None:
            ax = plt.gca()

        ax = stacked_bar(df=df, ax=ax, rule=rule, label=label, legend=legend)

        _ = ax.set_title(f"{self.identifier} - Energy Consumption")

        return ax

    def plot_pie(
        self,
        ax: plt.Axes = None,
        label: bool = True,
        legend: bool = True,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        **kwargs,
    ) -> plt.Axes:
        """Plot a pie chart of the annual energy consumption of the Masterplan."""

        logger.info(f"{self} - Plotting annual energy consumption pie chart")

        df = self.energy_consumption(combine_buildings=True)
        series = df.loc[pd.to_datetime(analysis_period.datetimes)].sum(axis=0)
        series.sort_values(ascending=False, inplace=True)

        # change units within index
        series.index = [i.split(" (")[0] for i in series.index]

        if ax is None:
            ax = plt.gca()

        # create the autopct labeller for the wedges
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                value = int(round(pct * total / 100.0))
                # don't show wedges smaller than 7.5%
                if pct / 100 > 0.075:
                    return f"{pct / 100:0.1%}\n({value / 1000:,.0f}MWh)"
                return ""

            return my_autopct

        ax = pie(
            series=series,
            ax=ax,
            legend=legend,
            label=label,
            autopct=make_autopct(series.values),
            **kwargs,
        )

        _ = ax.set_title(
            f"{self.identifier} - Energy Consumption\n{describe_analysis_period(analysis_period)}\n{series.sum() / 1000:,.0f}MWh"
        )

        return ax

    def plot_diurnal(
        self,
        ax: plt.Axes = None,
        legend: bool = True,
        logy: bool = False,
    ) -> plt.Axes:
        """Plot a monthly diurnal profile for energy consumption of the Masterplan."""

        logger.info(f"{self} - Plotting diurnal energy consumption")

        df = self.energy_consumption(combine_buildings=True)

        if ax is None:
            ax = plt.gca()

        ax = diurnal(df, ax=ax, legend=legend, logy=logy)

        _ = ax.set_title(f"{self.identifier} - Energy Consumption")

        return ax

    def plot_duration_curve(
        self,
        ax: plt.Axes = None,
        remove_zero: bool = True,
        legend: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """Plot a duration curve for energy consumption of the Masterplan."""

        logger.info(f"{self} - Plotting duration curve")

        df = self.energy_consumption(combine_buildings=True)

        if ax is None:
            ax = plt.gca()

        ax = duration_curve(df, ax=ax, legend=legend, remove_zero=remove_zero, **kwargs)

        _ = ax.set_title(f"{self.identifier} - Energy Consumption - Duration Curve")

        return ax

    def plot_pie_distribution(
        self,
        ax: plt.Axes = None,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        legend: bool = True,
    ) -> plt.Axes:
        """Plot a pie chart, with all typologies represented as a proportion of the total GFA."""

        df = self.energy_consumption(combine_buildings=False)
        temp = df.loc[pd.to_datetime(analysis_period.datetimes)].sum(axis=0).unstack()
        temp = temp.loc[temp.sum(axis=1).sort_values(ascending=False).index]

        # reorder to get legend in descending consumption order
        temp = temp[temp.sum().sort_values(ascending=False).index]

        size = 0.1
        vals = temp.values
        inner_colors = [get_color(i) for i in temp.columns]

        # rename columns to remove units
        temp.columns = [i.split(" (")[0] for i in temp.columns]

        if ax is None:
            ax = plt.gca()

        inner_wedge, _ = ax.pie(
            vals.flatten(),
            radius=1 - size,
            wedgeprops={"width": 0.66, "edgecolor": "w", "linewidth": 0},
            startangle=90,
            counterclock=False,
            colors=inner_colors,
        )
        _, outer_txt = ax.pie(
            vals.sum(axis=1),
            radius=1,
            wedgeprops={"width": size, "edgecolor": "w", "linewidth": 0.5},
            startangle=90,
            counterclock=False,
            colors=["grey"],
        )
        for n, (name, vals) in enumerate(temp.iterrows()):
            total_prop = vals.sum() / temp.sum().sum()
            if total_prop > 0.02:
                outer_txt[n].set_text(f"{name}\n{total_prop:0.1%}")
                outer_txt[n].set_fontsize("xx-small")

        if legend:
            ax.legend(
                inner_wedge[: len(temp.columns)],
                temp.columns,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                ncols=1,
            )

        ax.set_title(
            f"{self.identifier} - Energy Consumption\n{describe_analysis_period(analysis_period)}\n{temp.sum().sum() / 1000:,.0f}MWh"
        )

        return ax

    # def run_everything(self) -> None:
    #     """Run all methods and return a DataFrame with all the data."""
        
    #     _ = self._all_data()

    #     # create individual plots per typology
    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         futures = [
    #             executor.submit(
    #                 obj.run_everything,
    #                 self.simulation_directory,
    #             )
    #             for obj in self.typologies
    #         ]
    #         _ = [future.result() for future in futures]

    #     # create plots for the masterplan
    #     for legend in [True, False]:
    #         # PIE DISTRIBUTION #
    #         fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SQUARE)
    #         self.plot_pie_distribution(ax=ax, legend=legend)
    #         plt.savefig(self.simulation_directory / f"fig_diurnal{'' if legend else '_nolegend'}.png", bbox_inches="tight", transparent=True, dpi=300)
    #         plt.close(fig)

    #         # ANNUAL MONTHLY #
    #         fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RECTANGLE)
    #         self.plot_annual_monthly(ax=ax, label=True, legend=legend)
    #         plt.savefig(self.simulation_directory / f"fig_annual_monthly{'' if legend else '_nolegend'}.png", bbox_inches="tight", transparent=True, dpi=300)
    #         plt.close(fig)

    #         # PIE #
    #         fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SQUARE)
    #         self.plot_pie(ax=ax, label=True, legend=legend)
    #         plt.savefig(self.simulation_directory / f"fig_pie{'' if legend else '_nolegend'}.png", bbox_inches="tight", transparent=True, dpi=300)
    #         plt.close(fig)


def create_all_typologies(epw_file: Path | str, masterplan_identifier: str = "AllTypologies") -> Masterplan:

    def run(bt: BuildingType, epw_file: Path) -> Typology:
        return Typology.from_building_type(
            building_type=bt,
            total_area=1000,
            epw_file=epw_file,
            masterplan_identifier=masterplan_identifier,
        )

    iterations = product(*[BuildingType, [epw_file]])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                run,
                typ,
                e,
            )
            for typ, e in iterations
        ]
    typologies = [future.result() for future in futures]

    # # create all combinations of BuildingType and Vintage
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     results = executor.map(run, *iterations)

    return Masterplan(identifier=masterplan_identifier, typologies=typologies)
