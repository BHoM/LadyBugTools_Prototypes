"""Methods for post-processing honeybee-radiance daylight results."""

import json
from enum import Enum, auto
from pathlib import Path
from typing import Tuple
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from honeybee.model import Model
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_radiance_command.options.rfluxmtx import RfluxmtxOptions
from honeybee_radiance_command.options.rpict import RpictOptions
from honeybee_radiance_command.options.rtrace import RtraceOptions
from honeybee_radiance_postprocess.en17037 import en17037_to_folder
from honeybee_radiance_postprocess.results import Results, _filter_grids_by_pattern
from ladybug.wea import Wea
from ladybugtools_toolkit.categorical.categories import Categorical
from ladybugtools_toolkit.honeybee_extension.results import load_npy, load_res
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
from ladybugtools_toolkit.plot.utilities import colormap_sequential
from matplotlib.colors import Colormap, Normalize

EN17037_ILLUMINANCE_CATEGORIES = Categorical(
    bins=(-np.inf, 0, 1, 2, 3),
    bin_names=("Non-compliant", "Minimum", "Medium", "High"),
    colors=("#1f78b4", "#a6cee3", "#b2df8a", "#33a02c"),
    name="EN 17037 Target Illuminance",
)


class DaylightMetric(Enum):
    """A set of pre-defined daylight visualisation metric methods."""

    DAYLIGHT_FACTOR = auto()
    DAYLIGHT_AUTONOMY = auto()
    CONTINUOUS_DAYLIGHT_AUTONOMY = auto()
    USEFUL_DAYLIGHT_ILLUMINANCE = auto()
    USEFUL_DAYLIGHT_ILLUMINANCE_LT = auto()
    USEFUL_DAYLIGHT_ILLUMINANCE_GT = auto()
    DAYLIGHT_SATURATION_PERCENTAGE = auto()
    EN17037_TARGET_ILLUMINANCE = auto()
    EN17037_MINIMUM_ILLUMINANCE = auto()
    EN17037_DA_MINIMUM_ILLUMINANCE_100 = auto()
    EN17037_DA_MINIMUM_ILLUMINANCE_300 = auto()
    EN17037_DA_MINIMUM_ILLUMINANCE_500 = auto()
    EN17037_DA_TARGET_ILLUMINANCE_300 = auto()
    EN17037_DA_TARGET_ILLUMINANCE_500 = auto()
    EN17037_DA_TARGET_ILLUMINANCE_750 = auto()

    @property
    def acronym(self) -> str:
        """Get the acronym of the metric."""

        d = {
            DaylightMetric.DAYLIGHT_FACTOR.value: "DF",
            DaylightMetric.DAYLIGHT_AUTONOMY.value: "DA",
            DaylightMetric.CONTINUOUS_DAYLIGHT_AUTONOMY.value: "CDA",
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE.value: "UDI",
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_LT.value: "UDI_LT",
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_GT.value: "UDI_GT",
            DaylightMetric.DAYLIGHT_SATURATION_PERCENTAGE.value: "DSP",
            DaylightMetric.EN17037_TARGET_ILLUMINANCE.value: "EN17037_TARGET",
            DaylightMetric.EN17037_MINIMUM_ILLUMINANCE.value: "EN17037_MINIMUM_ILLUMINANCE",
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_100.value: "EN17037_DA_MINIMUM_ILLUMINANCE_100",
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_300.value: "EN17037_DA_MINIMUM_ILLUMINANCE_300",
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_500.value: "EN17037_DA_MINIMUM_ILLUMINANCE_500",
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_300.value: "EN17037_DA_TARGET_ILLUMINANCE_300",
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_500.value: "EN17037_DA_TARGET_ILLUMINANCE_500",
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_750.value: "EN17037_DA_TARGET_ILLUMINANCE_750",
        }
        return d[self.value]

    @property
    def name(self) -> str:
        """Get the name of the metric."""

        d = {
            DaylightMetric.DAYLIGHT_FACTOR.value: "Daylight Factor",
            DaylightMetric.DAYLIGHT_AUTONOMY.value: "Daylight Autonomy",
            DaylightMetric.CONTINUOUS_DAYLIGHT_AUTONOMY.value: "Continuous Daylight Autonomy",
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE.value: "Useful Daylight Illuminance",
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_LT.value: "Useful Daylight Illuminance (lower)",
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_GT.value: "Useful Daylight Illuminance (greater)",
            DaylightMetric.DAYLIGHT_SATURATION_PERCENTAGE.value: "Daylight Saturation Percentage",
            DaylightMetric.EN17037_TARGET_ILLUMINANCE.value: "EN 17037 Target Illuminance",
            DaylightMetric.EN17037_MINIMUM_ILLUMINANCE.value: "EN 17037 Minimum Illuminance",
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_100.value: "EN 17037 DA Minimum Illuminance (100lx)",
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_300.value: "EN 17037 DA Minimum Illuminance (300lx)",
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_500.value: "EN 17037 DA Minimum Illuminance (500lx)",
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_300.value: "EN 17037 DA Target Illuminance (300lx)",
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_500.value: "EN 17037 DA Target Illuminance (500lx)",
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_750.value: "EN 17037 DA Target Illuminance (750lx)",
        }
        return d[self.value]

    @property
    def cmap(self) -> Colormap:
        """Return the colormap assigned to the metric."""

        d = {
            DaylightMetric.DAYLIGHT_FACTOR.value: plt.get_cmap("magma"),
            DaylightMetric.DAYLIGHT_AUTONOMY.value: plt.get_cmap("viridis"),
            DaylightMetric.CONTINUOUS_DAYLIGHT_AUTONOMY.value: plt.get_cmap("viridis"),
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE.value: plt.get_cmap("Greens"),
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_LT.value: plt.get_cmap("Blues"),
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_GT.value: plt.get_cmap("Reds"),
            DaylightMetric.DAYLIGHT_SATURATION_PERCENTAGE.value: plt.get_cmap(
                "inferno"
            ),
            DaylightMetric.EN17037_TARGET_ILLUMINANCE.value: colormap_sequential(
                "#1f78b4", "#a6cee3", "#b2df8a", "#33a02c"
            ),
            DaylightMetric.EN17037_MINIMUM_ILLUMINANCE.value: colormap_sequential(
                "#1f78b4", "#a6cee3", "#b2df8a", "#33a02c"
            ),
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_100.value: plt.get_cmap(
                "cividis"
            ),
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_300.value: plt.get_cmap(
                "cividis"
            ),
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_500.value: plt.get_cmap(
                "cividis"
            ),
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_300.value: plt.get_cmap(
                "cividis"
            ),
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_500.value: plt.get_cmap(
                "cividis"
            ),
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_750.value: plt.get_cmap(
                "cividis"
            ),
        }
        return d[self.value]

    @property
    def norm(self) -> Normalize:
        """Return the normalisation method assigned to the metric."""

        d = {
            DaylightMetric.DAYLIGHT_FACTOR.value: Normalize(vmin=0, vmax=10),
            DaylightMetric.DAYLIGHT_AUTONOMY.value: Normalize(vmin=0, vmax=100),
            DaylightMetric.CONTINUOUS_DAYLIGHT_AUTONOMY.value: Normalize(
                vmin=0, vmax=100
            ),
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE.value: Normalize(
                vmin=0, vmax=100
            ),
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_LT.value: Normalize(
                vmin=0, vmax=100
            ),
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_GT.value: Normalize(
                vmin=0, vmax=100
            ),
            DaylightMetric.DAYLIGHT_SATURATION_PERCENTAGE.value: Normalize(
                vmin=0, vmax=100
            ),
            DaylightMetric.EN17037_TARGET_ILLUMINANCE.value: None,
            # BoundaryNorm(
            #     boundaries=[0, 1, 2, 3], ncolors=3
            # ),
            DaylightMetric.EN17037_MINIMUM_ILLUMINANCE.value: None,
            # BoundaryNorm(
            #     boundaries=[0, 1, 2, 3], ncolors=3
            # ),
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_100.value: Normalize(
                vmin=0, vmax=100
            ),
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_300.value: Normalize(
                vmin=0, vmax=100
            ),
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_500.value: Normalize(
                vmin=0, vmax=100
            ),
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_300.value: Normalize(
                vmin=0, vmax=100
            ),
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_500.value: Normalize(
                vmin=0, vmax=100
            ),
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_750.value: Normalize(
                vmin=0, vmax=100
            ),
        }

        return d[self.value]

    def summarize(self, values: Tuple[float]) -> Tuple[str]:
        """Generate summary for association with the metric and passed values.

        Note:
            This methods assumes that all values are weighted equally, and therefore represent the same areas in as given sensor grid.

        Args:
            values (Tuple[float]):
                A list of values for the metric.

        Returns:
            Tuple[str]:
                A list of strings summarizing the metric and the passed values.
        """

        if len(values) <= 1:
            raise ValueError(
                "The passed values must be a list of at least two values to be summarized."
            )

        values: np.ndarray = np.array(values)

        # NOTE - The padding (>7.2%) assumes that 'Uniformity' is the longest string in the keys below. Adjust for longer strings.

        if self == DaylightMetric.DAYLIGHT_FACTOR:
            return [
                f"Minimum:    {values.min() / 100:>7.2%}",
                f"Average:    {values.mean() / 100:>7.2%}",
                f"Median:     {np.median(values) / 100:>7.2%}",
                f"Maximum:    {values.max() / 100:>7.2%}",
                f"Uniformity: {values.min() / values.mean():>7.2%}",
                f"Area <2%:   {(values < 2).sum() / len(values):>7.2%}",
                f"Area >5%:   {(values > 5).sum() / len(values):>7.2%}",
            ]
        elif self in [
            DaylightMetric.DAYLIGHT_AUTONOMY,
            DaylightMetric.CONTINUOUS_DAYLIGHT_AUTONOMY,
            DaylightMetric.DAYLIGHT_SATURATION_PERCENTAGE,
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_100,
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_300,
            DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_500,
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_300,
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_500,
            DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_750,
        ]:
            return [
                f"Minimum:    {values.min() / 100:>7.2%}",
                f"Average:    {values.mean() / 100:>7.2%}",
                f"Maximum:    {values.max() / 100:>7.2%}",
                f"Area ≥50%:  {(values >= 50).sum() / len(values):>7.2%}",
            ]
        elif self in [
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE,
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_LT,
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_GT,
        ]:
            return [
                f"Minimum:    {values.min() / 100:>7.2%}",
                f"Average:    {values.mean() / 100:>7.2%}",
                f"Maximum:    {values.max() / 100:>7.2%}",
            ]
        elif self in [
            DaylightMetric.EN17037_TARGET_ILLUMINANCE,
            DaylightMetric.EN17037_MINIMUM_ILLUMINANCE,
        ]:
            return [
                f"Area 'non-compliant': {sum(values == 0) / len(values):>7.2%}",
                f"Area 'minimum':       {sum(values == 1) / len(values):>7.2%}",
                f"Area 'medium':        {sum(values == 2) / len(values):>7.2%}",
                f"Area 'high':          {sum(values == 3) / len(values):>7.2%}",
            ]
        else:
            raise ValueError(f"Unsupported metric: {self}")

    def get_values(
        self, model_simulation_folder: Path, sensorgrid: SensorGrid
    ) -> Tuple[float]:
        """Load the metric values for the associated sensorgrid.

        Args:
            sensorgrid (SensorGrid): _description_

        Returns:
            Tuple[float]: _description_
        """
        if self.value == DaylightMetric.DAYLIGHT_FACTOR.value:
            return daylight_factor(
                model_simulation_folder=model_simulation_folder,
                grids_filter=[sensorgrid.identifier],
            )

        if self.value == DaylightMetric.DAYLIGHT_AUTONOMY.value:
            return (
                annual_metrics(
                    model_simulation_folder=model_simulation_folder,
                    grids_filter=[sensorgrid.identifier],
                )
                .droplevel([0], axis=1)["da_300lx"]
                .values
            )

        # if self.value == DaylightMetric.CONTINUOUS_DAYLIGHT_AUTONOMY.value:
        #     return postprocess_metrics_sensorgrid(
        #         model_simulation_folder=model_simulation_folder,
        #         sensorgrid=sensorgrid,
        #         metric="cda",
        #     )

        # if self.value == DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE.value:
        #     return postprocess_metrics_sensorgrid(
        #         model_simulation_folder=model_simulation_folder,
        #         sensorgrid=sensorgrid,
        #         metric="udi",
        #     )

        # if self.value == DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_LT.value:
        #     return postprocess_metrics_sensorgrid(
        #         model_simulation_folder=model_simulation_folder,
        #         sensorgrid=sensorgrid,
        #         metric="udi_lower",
        #     )

        # if self.value == DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_GT.value:
        #     return postprocess_metrics_sensorgrid(
        #         model_simulation_folder=model_simulation_folder,
        #         sensorgrid=sensorgrid,
        #         metric="udi_upper",
        #     )

        # if self.value == DaylightMetric.DAYLIGHT_SATURATION_PERCENTAGE.value:
        #     return postprocess_daylightsaturationpercentage_sensorgrid(
        #         model_simulation_folder=model_simulation_folder,
        #         sensorgrid=sensorgrid,
        #     )

        # if self.value == DaylightMetric.EN17037_TARGET_ILLUMINANCE.value:
        #     return (
        #         postprocess_en17037_sensorgrid(
        #             model_simulation_folder=model_simulation_folder,
        #             sensorgrid=sensorgrid,
        #         )["en17037_target_illuminance"]
        #         .dropna()
        #         .values
        #     )

        # if self.value == DaylightMetric.EN17037_MINIMUM_ILLUMINANCE.value:
        #     return (
        #         postprocess_en17037_sensorgrid(
        #             model_simulation_folder=model_simulation_folder,
        #             sensorgrid=sensorgrid,
        #         )["en17037_minimum_illuminance"]
        #         .dropna()
        #         .values
        #     )

        # if self.value == DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_100.value:
        #     return (
        #         postprocess_en17037_sensorgrid(
        #             model_simulation_folder=model_simulation_folder,
        #             sensorgrid=sensorgrid,
        #         )["en17037_da_minimum_illuminance_100"]
        #         .dropna()
        #         .values
        #     )

        # if self.value == DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_300.value:
        #     return (
        #         postprocess_en17037_sensorgrid(
        #             model_simulation_folder=model_simulation_folder,
        #             sensorgrid=sensorgrid,
        #         )["en17037_da_minimum_illuminance_300"]
        #         .dropna()
        #         .values
        #     )

        # if self.value == DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_500.value:
        #     return (
        #         postprocess_en17037_sensorgrid(
        #             model_simulation_folder=model_simulation_folder,
        #             sensorgrid=sensorgrid,
        #         )["en17037_da_minimum_illuminance_500"]
        #         .dropna()
        #         .values
        #     )

        # if self.value == DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_300.value:
        #     return (
        #         postprocess_en17037_sensorgrid(
        #             model_simulation_folder=model_simulation_folder,
        #             sensorgrid=sensorgrid,
        #         )["en17037_da_target_illuminance_300"]
        #         .dropna()
        #         .values
        #     )

        # if self.value == DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_500.value:
        #     return (
        #         postprocess_en17037_sensorgrid(
        #             model_simulation_folder=model_simulation_folder,
        #             sensorgrid=sensorgrid,
        #         )["en17037_da_target_illuminance_500"]
        #         .dropna()
        #         .values
        #     )

        # if self.value == DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_750.value:
        #     return (
        #         postprocess_en17037_sensorgrid(
        #             model_simulation_folder=model_simulation_folder,
        #             sensorgrid=sensorgrid,
        #         )["en17037_da_target_illuminance_750"]
        #         .dropna()
        #         .values
        #     )

        raise ValueError(f"Unsupported metric: {self}")


def radiance_parameters(
    model: Model,
    detail_dim: float,
    recipe_type: str,
    detail_level: int = 0,
    additional_parameters: str = None,
) -> str:
    """Generate the default "recommended" Radiance parameters for a Honeybee Radiance simulation.

    This method also includes the estimation of ambient resolution based on the model dimensions.

    Args:
        model: Model
            A Honeybee Model.
        detail_dim: float
            The detail dimension in meters.
        recipe_type: str
            One of the following: 'point-in-time-grid', 'daylight-factor', 'point-in-time-image', 'annual'.
        detail_level: int
            One of 0 (low), 1 (medium) or 2 (high).
        additional_parameters: str
            Additional parameters to add to the Radiance command. Should be in the format of a Radiance command string e.g. '-ab 2 -aa 0.25'.

    Returns:
        str: The Radiance parameters as a string.
    """

    # recommendations for radiance parameters
    RTRACE = {
        "ab": [2, 3, 6],
        "ad": [512, 2048, 4096],
        "as_": [128, 2048, 4096],
        "ar": [16, 64, 128],
        "aa": [0.25, 0.2, 0.1],
        "dj": [0, 0.5, 1],
        "ds": [0.5, 0.25, 0.05],
        "dt": [0.5, 0.25, 0.15],
        "dc": [0.25, 0.5, 0.75],
        "dr": [0, 1, 3],
        "dp": [64, 256, 512],
        "st": [0.85, 0.5, 0.15],
        "lr": [4, 6, 8],
        "lw": [0.05, 0.01, 0.005],
        "ss": [0, 0.7, 1],
    }

    RPICT = {
        "ab": [2, 3, 6],
        "ad": [512, 2048, 4096],
        "as_": [128, 2048, 4096],
        "ar": [16, 64, 128],
        "aa": [0.25, 0.2, 0.1],
        "ps": [8, 4, 2],
        "pt": [0.15, 0.10, 0.05],
        "pj": [0.6, 0.9, 0.9],
        "dj": [0, 0.5, 1],
        "ds": [0.5, 0.25, 0.05],
        "dt": [0.5, 0.25, 0.15],
        "dc": [0.25, 0.5, 0.75],
        "dr": [0, 1, 3],
        "dp": [64, 256, 512],
        "st": [0.85, 0.5, 0.15],
        "lr": [4, 6, 8],
        "lw": [0.05, 0.01, 0.005],
        "ss": [0, 0.7, 1],
    }

    RFLUXMTX = {
        "ab": [3, 5, 6],
        "ad": [5000, 15000, 25000],
        "as_": [128, 2048, 4096],
        "ds": [0.5, 0.25, 0.05],
        "dt": [0.5, 0.25, 0.15],
        "dc": [0.25, 0.5, 0.75],
        "dr": [0, 1, 3],
        "dp": [64, 256, 512],
        "st": [0.85, 0.5, 0.15],
        "lr": [4, 6, 8],
        "lw": [0.000002, 6.67e-07, 4e-07],
        "ss": [0, 0.7, 1],
        "c": [1, 1, 1],
    }

    # VALIDATION
    RECIPE_TYPES = {
        "point-in-time-grid": [RTRACE, RtraceOptions()],
        "daylight-factor": [RTRACE, RtraceOptions()],
        "point-in-time-image": [RPICT, RpictOptions()],
        "annual": [RFLUXMTX, RfluxmtxOptions()],
    }
    if recipe_type not in RECIPE_TYPES:
        raise ValueError(f"detail_level must be one of {RECIPE_TYPES.keys()}")

    if detail_level not in [0, 1, 2]:
        raise ValueError("Detail level must be one of 0 (low), 1 (medium) or 2 (high).")

    options, obj = RECIPE_TYPES[recipe_type]
    for opt, vals in options.items():
        setattr(obj, opt, vals[detail_level])

    min_pt, max_pt = model.min, model.max
    x_dim = max_pt.x - min_pt.x
    y_dim = max_pt.y - min_pt.y
    z_dim = max_pt.z - min_pt.z
    longest_dim = max((x_dim, y_dim, z_dim))
    try:
        obj.ar = int((longest_dim * obj.aa) / detail_dim)
    except TypeError as _:
        obj.ar = int((longest_dim * 0.1) / detail_dim)

    if additional_parameters:
        obj.update_from_string(additional_parameters)

    return obj.to_radiance()


def annual_metrics(
    model_simulation_folder: Path,
    threshold: float = 300,
    min_t: float = 100,
    max_t: float = 3000,
    grids_filter: Tuple[str] = "*",
    occ_schedule: Tuple[int] = None,
) -> pd.DataFrame:
    """Calculate daylight autonomy for the given model simulation folder.

    Args:
        model_simulation_folder (Path):
            Path to the model simulation folder.
        occ_schedule (Tuple[int], optional):
            A list of integers representing the occupancy schedule. Defaults to None which uses LB default occupancy profile.
        dathreshold (float, optional):
            The threshold in lux. Defaults to 300.
        udi_min_t (float, optional):
            The minimum threshold in lux. Defaults to 100.
        udi_max_t (float, optional):
            The maximum threshold in lux. Defaults to 3000.
        grids_filter (Tuple[str], optional):
            A list of strings to filter the grids. Defaults to ("*") which includes all grids.

    Returns:
        pd.DataFrame:
            A pandas DataFrame with daylight autonomy results per grid requested.
    """

    annual_daylight_folder = model_simulation_folder / "annual_daylight/results"
    if not annual_daylight_folder.exists():
        raise ValueError("The given folder does not contain annual daylight results.")

    if not isinstance(grids_filter, (list, tuple, str)):
        print(type(grids_filter))
        raise ValueError("The grids_filter must be a string, list or tuple of strings.")

    # run metrics calculation
    results = Results(annual_daylight_folder, schedule=occ_schedule, load_arrays=False)
    metrics_folder = annual_daylight_folder / "metrics"
    results.annual_metrics_to_folder(
        target_folder=metrics_folder,
        threshold=threshold,
        min_t=min_t,
        max_t=max_t,
        grids_filter=grids_filter,
    )

    # load metrics
    da = load_res(list((metrics_folder / "da").glob("*.da")))
    cda = load_res(list((metrics_folder / "cda").glob("*.cda")))
    udi = load_res(list((metrics_folder / "udi").glob("*.udi")))
    udi_lower = load_res(list((metrics_folder / "udi_lower").glob("*.udi")))
    udi_upper = load_res(list((metrics_folder / "udi_upper").glob("*.udi")))

    return (
        pd.concat(
            [da, cda, udi, udi_lower, udi_upper],
            axis=1,
            keys=[
                f"da_{threshold}lx",
                f"cda_{threshold}lx",
                f"udi_{min_t}lx_{max_t}lx",
                f"udi_lower_{min_t}lx",
                f"udi_upper_{max_t}lx",
            ],
        )
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1)
    )


def daylight_saturation_percentage(
    model_simulation_folder: Path,
    min_t: float = 430,
    max_t: float = 4300,
    grids_filter: Tuple[str] = "*",
    occ_schedule: Tuple[int] = None,
) -> pd.DataFrame:
    annual_daylight_folder = Path(model_simulation_folder) / "annual_daylight/results"
    if not annual_daylight_folder.exists():
        raise ValueError("The given folder does not contain annual daylight results.")

    if not isinstance(grids_filter, (list, tuple, str)):
        raise ValueError("The grids_filter must be a string, list or tuple of strings.")

    results = Results(annual_daylight_folder, schedule=occ_schedule, load_arrays=False)
    dsp_min = 430
    dsp_max = 4300
    dsp = []
    names = []
    for grid_info in results._filter_grids(grids_filter):
        names.append(grid_info["identifier"])
        npy_file = results._get_file(
            grid_info["identifier"],
            light_path="__static_apertures__",
            state_identifier="default",
            res_type="total",
        )
        ill = load_npy(npy_file)
        gt_min = ill >= dsp_min
        lt_max = ill <= dsp_max
        gt_max = ill > dsp_max
        dsp.append(
            (
                (
                    (((gt_min & lt_max).sum(axis=0) - gt_max.sum(axis=0)) / len(gt_min))
                    .unstack()
                    .T
                )
                * 100
            )
            .squeeze()
            .values
        )

    dsp_results = (
        pd.concat(
            [
                pd.DataFrame(dsp, index=names).T,
            ],
            axis=1,
            keys=[f"dsp_{min_t}lx_{max_t}lx"],
        )
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1)
    )
    metrics_folder = annual_daylight_folder / "metrics" / "dsp"
    metrics_folder.mkdir(parents=True, exist_ok=True)
    for grid_name, values in dsp_results.droplevel([1], axis=1).iteritems():
        values.dropna().round(2).to_csv(
            metrics_folder / f"{grid_name}.dsp", header=False, index=False
        )

    return dsp_results


def daylight_factor(
    model_simulation_folder: Path,
    grids_filter: Tuple[str] = "*",
) -> pd.DataFrame:
    """Obtain daylight factor results for the given model simulation folder.

    Args:
        model_simulation_folder (Path): The model simulation folder.
        grids_filter (Tuple[str], optional): A list of strings to filter the grids. Defaults to ("*") which includes all grids.

    Returns:
        pd.DataFrame: A pandas DataFrame with daylight factor results per grid requested.
    """
    daylight_factor_folder = model_simulation_folder / "daylight_factor/results"
    if not daylight_factor_folder.exists():
        raise ValueError("The given folder does not contain daylight factor results.")

    if not isinstance(grids_filter, (list, tuple, str)):
        raise ValueError("The grids_filter must be a string, list or tuple of strings.")

    with open(daylight_factor_folder / "grids_info.json") as f:
        grids_info = json.load(f)

    # get files
    # FIXME - the replacement of '::' is a hack to avoid the ':' in the grid name
    daylight_factor_files = [
        daylight_factor_folder / f"{i['name'].replace('::', '')}.res"
        for i in _filter_grids_by_pattern(grids_info, grids_filter)
    ]

    return (
        pd.concat([load_res(daylight_factor_files)], axis=1, keys=["df"])
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1)
    )


def en17037(
    model_simulation_folder: Path,
    grids_filter: Tuple[str] = "*",
    occ_schedule: Tuple[int] = None,
) -> pd.DataFrame:
    annual_daylight_folder = Path(model_simulation_folder) / "annual_daylight/results"
    if not annual_daylight_folder.exists():
        raise ValueError("The given folder does not contain annual daylight results.")

    if not isinstance(grids_filter, (list, tuple, str)):
        raise ValueError("The grids_filter must be a string, list or tuple of strings.")

    results = Results(annual_daylight_folder, schedule=occ_schedule, load_arrays=False)

    wea = Wea.from_file(list(Path(model_simulation_folder).glob("*.wea"))[0])
    global_rad = collection_to_series(wea.global_horizontal_irradiance)
    peak_sun_schedule = (
        global_rad.index.isin(
            global_rad.sort_values()[-int(len(global_rad) / 2) :].index
        )
        .astype(int)
        .tolist()
    )

    en17071_dir = en17037_to_folder(
        results,
        schedule=peak_sun_schedule,
        grids_filter=grids_filter,
        sub_folder=(annual_daylight_folder / "metrics/en17037"),
    )

    # load results
    tgt_ill = load_res(
        list((en17071_dir / "compliance_level/target_illuminance").glob("*.pf"))
    )
    min_ill = load_res(
        list((en17071_dir / "compliance_level/minimum_illuminance").glob("*.pf"))
    )
    da_min_ill_100 = load_res(
        list((en17071_dir / "da/minimum_illuminance_100").glob("*.da"))
    )
    da_min_ill_300 = load_res(
        list((en17071_dir / "da/minimum_illuminance_300").glob("*.da"))
    )
    da_min_ill_500 = load_res(
        list((en17071_dir / "da/minimum_illuminance_500").glob("*.da"))
    )
    da_tgt_ill_300 = load_res(
        list((en17071_dir / "da/target_illuminance_300").glob("*.da"))
    )
    da_tgt_ill_500 = load_res(
        list((en17071_dir / "da/target_illuminance_500").glob("*.da"))
    )
    da_tgt_ill_750 = load_res(
        list((en17071_dir / "da/target_illuminance_750").glob("*.da"))
    )

    # combine and return results
    return (
        pd.concat(
            [
                tgt_ill,
                min_ill,
                da_min_ill_100,
                da_min_ill_300,
                da_min_ill_500,
                da_tgt_ill_300,
                da_tgt_ill_500,
                da_tgt_ill_750,
            ],
            keys=[
                "target_illuminance",
                "minimum_illuminance",
                "da_minimum_illuminance_100",
                "da_minimum_illuminance_300",
                "da_minimum_illuminance_500",
                "da_target_illuminance_300",
                "da_target_illuminance_500",
                "da_target_illuminance_750",
            ],
            axis=1,
        )
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1)
    )


def annual_sunlight_exposure(
    model_simulation_folder: Path,
    direct_threshold: float = 1000,
    occ_hours: int = 250,
    grids_filter: Tuple[str] = "*",
    occ_schedule: Tuple[int] = None,
) -> pd.DataFrame:
    annual_daylight_folder = Path(model_simulation_folder) / "annual_daylight/results"
    if not annual_daylight_folder.exists():
        raise ValueError("The given folder does not contain annual daylight results.")

    if not isinstance(grids_filter, (list, tuple, str)):
        raise ValueError("The grids_filter must be a string, list or tuple of strings.")

    # print(annual_daylight_folder)
    results = Results(annual_daylight_folder, schedule=occ_schedule, load_arrays=False)
    metrics_folder = annual_daylight_folder / "metrics" / "ase"
    results.annual_sunlight_exposure_to_folder(
        target_folder=metrics_folder,
        direct_threshold=direct_threshold,
        occ_hours=occ_hours,
        grids_filter=grids_filter,
    )

    # get files
    ase_files = [
        metrics_folder / "hours_above" / f"{i['identifier']}.res"
        for i in _filter_grids_by_pattern(results.grids_info, grids_filter)
    ]

    return (
        pd.concat(
            [load_res(ase_files)],
            axis=1,
            keys=[f"ase_{direct_threshold}lx_{occ_hours}hrs"],
        )
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1)
    )


def breeam_hea01(
    model_simulation_folder: Path,
    avg_threshold: float = 300,
    min_threshold: float = 90,
    number_of_hours: int = 2000,
    grids_filter: Tuple[str] = "*",
    occ_schedule: Tuple[int] = None,
    avg_illuminance_method: int = 0,
    min_illuminance_method: int = 0,
) -> pd.DataFrame:
    """Calculate the BREEAM HEA01 metric for the given model simulation folder.

    Args:
        model_simulation_folder (Path): A path to the model simulation folder.
        avg_illuminance (float, optional): The average illuminance in lux to be achieved for number_of_hours hours per year. Defaults to 300.
        min_illuminance (float, optional): The illuminance in lux to be exceeded for number_of_hours hours per year. Defaults to 90.
        number_of_hours (int, optional): The number of hours to use for the metric calculation. Defaults to 2000. This value changes per space type according to BREEAM requirements.
        occ_schedule (Tuple[int], optional): An occupancy schedule. Defaults to None which uses the default LB occupancy schedule.
        grids_filter (Tuple[str], optional): A list of strings to filter the grids. Defaults to ("*") which includes all grids.
        avg_illuminance_method (int, optional): The method to use for the Average daylight illuminance calculation. Defaults to 0.
        min_illuminance_method (int, optional): The method to use for the Minimum daylight illuminance calculation. Defaults to 0.

    Notes:
        The method parameter is currently used to switch how the guidance is interpreted. ONce an answer is received from BRE then this will be removed.

    Returns:
        pd.DataFrame: A pandas DataFrame with the BREEAM HEA01 results per grid requested.
    """

    annual_daylight_folder = Path(model_simulation_folder) / "annual_daylight/results"
    if not annual_daylight_folder.exists():
        raise ValueError("The given folder does not contain annual daylight results.")

    if not isinstance(grids_filter, (list, tuple, str)):
        raise ValueError("The grids_filter must be a string, list or tuple of strings.")

    results = Results(annual_daylight_folder, schedule=occ_schedule, load_arrays=False)

    min_achieved_res = []
    avg_achieved_res = []
    names = []
    for grid_info in results._filter_grids(grids_filter):
        names.append(grid_info["identifier"])
        npy_file = results._get_file(
            grid_info["identifier"],
            light_path="__static_apertures__",
            state_identifier="default",
            res_type="total",
        )
        ill = load_npy(npy_file).droplevel(0, axis=1)

        ###################
        # WORST LIT POINT #
        ###################

        if min_illuminance_method == 0:
            warn(
                "The worst lit point is being interpreted as the point with the lowest cumulative amount of daylight."
            )
            worst_lit_point = ill.sum(axis=0).idxmin()
        elif min_illuminance_method == 1:
            warn(
                "The worst lit point is being interpreted as the point with the lowest average daylight."
            )
            worst_lit_point = ill.mean(axis=0).idxmin()
        elif min_illuminance_method == 2:
            warn(
                "The worst lit point is being interpreted as the point with the lowest median daylight."
            )
            worst_lit_point = ill.median(axis=0).idxmin()
        else:
            raise ValueError("The min_illuminance_method chosen is not known.")
        # create a spatial grid indicating whether the min_illuminance is met for the target number of hours
        min_achieved = (
            ((ill >= min_threshold).sum() > number_of_hours)
            .astype(int)
            .rename(grid_info["identifier"])
        )
        metrics_folder = (
            annual_daylight_folder / "metrics" / "breeam_hea01" / "minimum_illuminance"
        )
        metrics_folder.mkdir(parents=True, exist_ok=True)
        min_achieved.to_csv(
            metrics_folder / f"{grid_info['identifier']}.hea01_min",
            header=False,
            index=False,
        )

        # get a single value, describing whether the worst list pt achieves the min_illuminance for the target number_of_hours
        is_min_achieved_for_worst_lit_pt = bool(min_achieved.iloc[worst_lit_point])
        metrics_folder = (
            annual_daylight_folder / "metrics" / "breeam_hea01" / "minimum_achieved"
        )
        metrics_folder.mkdir(parents=True, exist_ok=True)
        with open(metrics_folder / f"{grid_info['identifier']}.hea01", "w") as f:
            f.write(str(int(is_min_achieved_for_worst_lit_pt)))

        ###################
        # AVG ILLUMINANCE #
        ###################

        if avg_illuminance_method == 0:
            warn(
                "The number of hours exceeding the avg_illuminance are counted, and compared against the number_of_hours, to give a True/False for each point achieving that target."
            )
            avg_achieved = ((ill >= avg_threshold).sum() > number_of_hours).astype(int)
            space_achieving_avg = (avg_achieved.sum() / len(avg_achieved)) * 100
        elif avg_illuminance_method == 1:
            warn(
                "The top number_of_hours of illuminance are obtained for each point, then the average is taken for each of those timesteps. Whether the avg_illuminance is exceeded or not is returned."
            )
            avg_achieved = pd.Series(
                data=[
                    int(
                        i.sort_values(ascending=False)[:number_of_hours].mean()
                        >= avg_threshold
                    )
                    for idx, i in ill.iteritems()
                ],
                index=ill.columns,
                name=grid_info["identifier"],
            )
            space_achieving_avg = (avg_achieved.sum() / len(avg_achieved)) * 100
        else:
            raise ValueError("The avg_illuminance_method chosen is not known.")

        metrics_folder = (
            annual_daylight_folder / "metrics" / "breeam_hea01" / "average_illuminance"
        )
        metrics_folder.mkdir(parents=True, exist_ok=True)
        avg_achieved.to_csv(
            metrics_folder / f"{grid_info['identifier']}.hea01_avg",
            header=False,
            index=False,
        )

        metrics_folder = (
            annual_daylight_folder / "metrics" / "breeam_hea01" / "average_achieved"
        )
        metrics_folder.mkdir(parents=True, exist_ok=True)
        with open(metrics_folder / f"{grid_info['identifier']}.hea01", "w") as f:
            f.write(str(space_achieving_avg))

        min_achieved_res.append(min_achieved)
        avg_achieved_res.append(avg_achieved)

    return pd.concat(
        [
            pd.concat(avg_achieved_res, axis=1),
            pd.concat(min_achieved_res, axis=1),
        ],
        axis=1,
        keys=[
            f"Average daylight illuminance ≥ {avg_threshold}lx over {number_of_hours}hrs",
            f"Minimum daylight illuminance ≥ {min_threshold}lx over {number_of_hours}hrs",
        ],
    )


def postprocess_all(
    model_simulation_folder: Path,
    _en17037: bool = True,
    _annual_metrics: bool = True,
    _daylight_saturation_percentage: bool = True,
    _annual_sunlight_exposure: bool = True,
    _daylight_factor: bool = True,
    _breeam_hea01: bool = False,
    grids_filter: Tuple[str] = "*",
) -> pd.DataFrame:
    """Run all post-processing methods with default settings.

    Args:
        model_simulation_folder (Path): The path to the model simulation folder.

    Returns:
        pd.DataFrame: A pandas DataFrame with all post-processing results.
    """

    if (
        sum(
            [
                _en17037,
                _annual_metrics,
                _daylight_saturation_percentage,
                _annual_sunlight_exposure,
                _daylight_factor,
                _breeam_hea01,
            ]
        )
        == 0
    ):
        raise ValueError(
            "No post-processing method was selected. Please select at least one method."
        )

    dfs = []

    print(f"Processing {model_simulation_folder.name}")

    if _en17037:
        try:
            print("- Running EN17037")
            dfs.append(en17037(model_simulation_folder, grids_filter=grids_filter))
        except Exception as e:
            warn(f"Failed to run EN 17037: {e}")

    if _annual_metrics:
        try:
            print("- Running annual metrics (DA, CDA, UDI)")
            dfs.append(
                annual_metrics(model_simulation_folder, grids_filter=grids_filter)
            )
        except Exception as e:
            warn(f"Failed to run annual metrics: {e}")

    if _daylight_saturation_percentage:
        try:
            print("- Running daylight saturation percentage")
            dfs.append(
                daylight_saturation_percentage(
                    model_simulation_folder, grids_filter=grids_filter
                )
            )
        except Exception as e:
            warn(f"Failed to run daylight saturation percentage: {e}")

    if _annual_sunlight_exposure:
        try:
            print("- Running annual sunlight exposure")
            dfs.append(
                annual_sunlight_exposure(
                    model_simulation_folder, grids_filter=grids_filter
                )
            )
        except Exception as e:
            warn(f"Failed to run annual sunlight exposure: {e}")

    if _daylight_factor:
        try:
            print("- Running daylight factor")
            dfs.append(
                daylight_factor(model_simulation_folder, grids_filter=grids_filter)
            )
        except Exception as e:
            warn(f"Failed to run daylight factor: {e}")

    if _breeam_hea01:
        try:
            print("- Running BEEAM HEA01")
            dfs.append(breeam_hea01(model_simulation_folder, grids_filter=grids_filter))
        except Exception as e:
            warn(f"Failed to run BREEAM HEA01: {e}")

    return pd.concat(dfs, axis=1).reorder_levels([1, 0], axis=1).sort_index(axis=1)


# def plot_level_by_level(model_simulation_folder: Path) -> None:
#     model_simulation_folder = Path(model_simulation_folder)

#     # Load model
#     model = Model.from_hbjson(
#         model_simulation_folder / f"{model_simulation_folder.name}.hbjson"
#     )

#     # get sensor grids and sort by z-level
#     grids = get_sensorgrids(model_simulation_folder=model_simulation_folder)
#     level_grids = sensorgrid_groupby_level(grids)

#     pbar1 = tqdm(DaylightMetric)
#     for metric in pbar1:
#         # for each level
#         for nn, (level, grids) in enumerate(level_grids.items()):
#             pbar1.set_description(
#                 f"Plotting {metric.name}, level {level:0.2f}m [{nn+1:02d}/{len(level_grids):02d}]"
#             )
#             fig, ax = plt.subplots(
#                 1,
#                 1,
#             )

#             # plot mesh
#             level_vals = []  # < this is used to generate level-wise metrics
#             level_pts = []  # < this is used to get the axis limits
#             for grid in grids:
#                 # load/process the metric values to plot
#                 vals = metric.get_values(
#                     sensorgrid=grid, model_simulation_folder=model_simulation_folder
#                 )
#                 # create patch collection to render
#                 pc = sensorgrid_to_patchcollection(
#                     sensorgrid=grid, cmap=metric.cmap, norm=metric.norm, zorder=1
#                 )
#                 # assign valeus to patch collection
#                 pc.set_array(vals)
#                 # place patch collection in plot
#                 ax.add_collection(pc)
#                 level_vals.extend(vals)
#                 level_pts.extend([grid.mesh.min, grid.mesh.max])

#             # Plot wireframe, for each of the HB goemetry types (e.g. wall, floor, window, ...)
#             for b in HbModelGeometry:
#                 pc = b.slice_polycollection(
#                     model=model, plane=Plane(o=Point3D(0, 0, level))
#                 )
#                 j = ax.add_collection(pc)

#             # Tidy up the plot a little
#             ax.set_aspect("equal")
#             ax.autoscale()
#             ax.axis("off")

#             # narrow to only show the region around the level
#             ax.set_xlim(
#                 min([i.x for i in level_pts]) - 1, max([i.x for i in level_pts]) + 1
#             )
#             ax.set_ylim(
#                 min([i.y for i in level_pts]) - 1, max([i.y for i in level_pts]) + 1
#             )

#             # add the summary stats for the level
#             xa, xb = ax.get_xlim()
#             ya, yb = ax.get_ylim()
#             ax.text(
#                 xa,
#                 yb,
#                 "\n".join(metric.summarize(level_vals)),
#                 ha="left",
#                 va="top",
#                 fontdict=({"family": "monospace"}),
#                 bbox=dict(boxstyle="square, pad=0", fc="w", alpha=0.75, ec="none"),
#             )

#             # PLace the legend
#             divider = make_axes_locatable(ax)
#             cax = divider.append_axes("right", size="5%", pad=0.15)
#             cb = fig.colorbar(
#                 mappable=ax.get_children()[0], cax=cax, label=f"{metric.name}"
#             )
#             cb.outline.set_visible(False)

#             # Add a title
#             ax.set_title(f"z-level: {level:0.2f}m")

#             plt.tight_layout()

#             # save the figure
#             img_dir = model_simulation_folder / "images"
#             img_dir.mkdir(exist_ok=True, parents=True)
#             plt.savefig(img_dir / f"{metric.acronym}_level_{nn:02d}.png", dpi=300)

#             plt.close(fig)
