"""Methods for post-processing honeybee-radiance daylight results."""

from honeybee_radiance.sensorgrid import SensorGrid, Sensor
from pathlib import Path
from tqdm import tqdm
from ladybugtools_toolkit.honeybee_extension.results import (
    load_ill,
    load_npy,
    load_res,
    load_pts,
    make_annual,
)
from honeybee.model import Model
from .model import HbModelGeometry
from .sensorgrid import (
    get_sensorgrids,
    sensorgrid_groupby_level,
    sensorgrid_to_patchcollection,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ladybug_geometry.geometry3d import Point3D, Plane
import numpy as np
from honeybee_radiance_postprocess.results import Results
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
from ladybug.wea import Wea
from typing import Dict, List
from honeybee.config import folders
import pandas as pd
from warnings import warn
from honeybee_radiance_postprocess.en17037 import en17037_to_folder
from honeybee_radiance_postprocess.annualdaylight import metrics_to_folder
from ladybugtools_toolkit.plot.utilities import colormap_sequential
from .sensorgrid import get_sensorgrids
from ladybugtools_toolkit.categorical.categories import Categorical
import subprocess
from enum import Enum, auto
from matplotlib.colors import Colormap, Normalize, BoundaryNorm
import matplotlib.pyplot as plt

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
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_LT.value: "Useful Daylight Illuminance (LT)",
            DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_GT.value: "Useful Daylight Illuminance (GT)",
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

    def summarize(self, values: List[float]) -> List[str]:
        """Generate summary for association with the metric and passed values.

        Note:
            This methods assumes that all values are weighted equally, and therefore represent the same areas in as given sensor grid.

        Args:
            values (List[float]):
                A list of values for the metric.

        Returns:
            List[str]:
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
    ) -> List[float]:
        """Load the metric values for the associated sensorgrid.

        Args:
            sensorgrid (SensorGrid): _description_

        Returns:
            List[float]: _description_
        """
        if self.value == DaylightMetric.DAYLIGHT_FACTOR.value:
            return postprocess_daylightfactor_sensorgrid(
                model_simulation_folder=model_simulation_folder,
                sensorgrid=sensorgrid,
            )

        if self.value == DaylightMetric.DAYLIGHT_AUTONOMY.value:
            return postprocess_metrics_sensorgrid(
                model_simulation_folder=model_simulation_folder,
                sensorgrid=sensorgrid,
                metric="da",
            )

        if self.value == DaylightMetric.CONTINUOUS_DAYLIGHT_AUTONOMY.value:
            return postprocess_metrics_sensorgrid(
                model_simulation_folder=model_simulation_folder,
                sensorgrid=sensorgrid,
                metric="cda",
            )

        if self.value == DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE.value:
            return postprocess_metrics_sensorgrid(
                model_simulation_folder=model_simulation_folder,
                sensorgrid=sensorgrid,
                metric="udi",
            )

        if self.value == DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_LT.value:
            return postprocess_metrics_sensorgrid(
                model_simulation_folder=model_simulation_folder,
                sensorgrid=sensorgrid,
                metric="udi_lower",
            )

        if self.value == DaylightMetric.USEFUL_DAYLIGHT_ILLUMINANCE_GT.value:
            return postprocess_metrics_sensorgrid(
                model_simulation_folder=model_simulation_folder,
                sensorgrid=sensorgrid,
                metric="udi_upper",
            )

        if self.value == DaylightMetric.DAYLIGHT_SATURATION_PERCENTAGE.value:
            return postprocess_daylightsaturationpercentage_sensorgrid(
                model_simulation_folder=model_simulation_folder,
                sensorgrid=sensorgrid,
            )

        if self.value == DaylightMetric.EN17037_TARGET_ILLUMINANCE.value:
            return (
                postprocess_en17037_sensorgrid(
                    model_simulation_folder=model_simulation_folder,
                    sensorgrid=sensorgrid,
                )["en17037_target_illuminance"]
                .dropna()
                .values
            )

        if self.value == DaylightMetric.EN17037_MINIMUM_ILLUMINANCE.value:
            return (
                postprocess_en17037_sensorgrid(
                    model_simulation_folder=model_simulation_folder,
                    sensorgrid=sensorgrid,
                )["en17037_minimum_illuminance"]
                .dropna()
                .values
            )

        if self.value == DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_100.value:
            return (
                postprocess_en17037_sensorgrid(
                    model_simulation_folder=model_simulation_folder,
                    sensorgrid=sensorgrid,
                )["en17037_da_minimum_illuminance_100"]
                .dropna()
                .values
            )

        if self.value == DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_300.value:
            return (
                postprocess_en17037_sensorgrid(
                    model_simulation_folder=model_simulation_folder,
                    sensorgrid=sensorgrid,
                )["en17037_da_minimum_illuminance_300"]
                .dropna()
                .values
            )

        if self.value == DaylightMetric.EN17037_DA_MINIMUM_ILLUMINANCE_500.value:
            return (
                postprocess_en17037_sensorgrid(
                    model_simulation_folder=model_simulation_folder,
                    sensorgrid=sensorgrid,
                )["en17037_da_minimum_illuminance_500"]
                .dropna()
                .values
            )

        if self.value == DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_300.value:
            return (
                postprocess_en17037_sensorgrid(
                    model_simulation_folder=model_simulation_folder,
                    sensorgrid=sensorgrid,
                )["en17037_da_target_illuminance_300"]
                .dropna()
                .values
            )

        if self.value == DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_500.value:
            return (
                postprocess_en17037_sensorgrid(
                    model_simulation_folder=model_simulation_folder,
                    sensorgrid=sensorgrid,
                )["en17037_da_target_illuminance_500"]
                .dropna()
                .values
            )

        if self.value == DaylightMetric.EN17037_DA_TARGET_ILLUMINANCE_750.value:
            return (
                postprocess_en17037_sensorgrid(
                    model_simulation_folder=model_simulation_folder,
                    sensorgrid=sensorgrid,
                )["en17037_da_target_illuminance_750"]
                .dropna()
                .values
            )

        raise ValueError(f"Unsupported metric: {self}")


def postprocess_en17037_sensorgrid(
    model_simulation_folder: Path, sensorgrid: SensorGrid
) -> pd.DataFrame:
    model_simulation_folder = Path(model_simulation_folder)
    annual_daylight_folder = model_simulation_folder / "annual_daylight"
    metrics_folder = annual_daylight_folder / "results" / "metrics" / "en17037"

    if not metrics_folder.exists():
        postprocess_en17037(model_simulation_folder)

    target_illuminance_res = (
        metrics_folder
        / "compliance_level"
        / "target_illuminance"
        / f"{sensorgrid.identifier}.pf"
    )
    minimum_illuminance_res = (
        metrics_folder
        / "compliance_level"
        / "minimum_illuminance"
        / f"{sensorgrid.identifier}.pf"
    )
    da_minimum_illuminance_100_res = (
        metrics_folder
        / "da"
        / "minimum_illuminance_100"
        / f"{sensorgrid.identifier}.da"
    )
    da_minimum_illuminance_300_res = (
        metrics_folder
        / "da"
        / "minimum_illuminance_300"
        / f"{sensorgrid.identifier}.da"
    )
    da_minimum_illuminance_500_res = (
        metrics_folder
        / "da"
        / "minimum_illuminance_500"
        / f"{sensorgrid.identifier}.da"
    )
    da_target_illuminance_300_res = (
        metrics_folder / "da" / "target_illuminance_300" / f"{sensorgrid.identifier}.da"
    )
    da_target_illuminance_500_res = (
        metrics_folder / "da" / "target_illuminance_500" / f"{sensorgrid.identifier}.da"
    )
    da_target_illuminance_750_res = (
        metrics_folder / "da" / "target_illuminance_750" / f"{sensorgrid.identifier}.da"
    )

    return pd.concat(
        [
            load_res(target_illuminance_res).squeeze(),
            load_res(minimum_illuminance_res).squeeze(),
            load_res(da_minimum_illuminance_100_res).squeeze(),
            load_res(da_minimum_illuminance_300_res).squeeze(),
            load_res(da_minimum_illuminance_500_res).squeeze(),
            load_res(da_target_illuminance_300_res).squeeze(),
            load_res(da_target_illuminance_500_res).squeeze(),
            load_res(da_target_illuminance_750_res).squeeze(),
        ],
        axis=1,
        keys=[
            "en17037_target_illuminance",
            "en17037_minimum_illuminance",
            "en17037_da_minimum_illuminance_100",
            "en17037_da_minimum_illuminance_300",
            "en17037_da_minimum_illuminance_500",
            "en17037_da_target_illuminance_300",
            "en17037_da_target_illuminance_500",
            "en17037_da_target_illuminance_750",
        ],
    ).sort_index(axis=1)


def postprocess_en17037(model_simulation_folder: Path) -> pd.DataFrame:
    """Post-process annual daylight results to get EN 17037 metrics.

    Args:
        model_simulation_folder (Path):
            Path to the model simulation folder.

    Returns:
        pd.DataFrame:
            A pandas DataFrame with EN 17037 metrics.
    """
    model_simulation_folder = Path(model_simulation_folder)
    annual_daylight_folder = model_simulation_folder / "annual_daylight"
    if not annual_daylight_folder.exists():
        raise ValueError("The given folder does not contain annual daylight results.")

    sensorgrids = get_sensorgrids(model_simulation_folder)
    sub_folder = annual_daylight_folder / "results/metrics/en17037"

    # check for existence of results and load if already run
    target_illuminance_res = list(
        (sub_folder / "compliance_level/target_illuminance").glob("*.pf")
    )
    minimum_illuminance_res = list(
        (sub_folder / "compliance_level/minimum_illuminance").glob("*.pf")
    )
    da_minimum_illuminance_100_res = list(
        (sub_folder / "da/minimum_illuminance_100").glob("*.da")
    )
    da_minimum_illuminance_300_res = list(
        (sub_folder / "da/minimum_illuminance_300").glob("*.da")
    )
    da_minimum_illuminance_500_res = list(
        (sub_folder / "da/minimum_illuminance_500").glob("*.da")
    )
    da_target_illuminance_300_res = list(
        (sub_folder / "da/target_illuminance_300").glob("*.da")
    )
    da_target_illuminance_500_res = list(
        (sub_folder / "da/target_illuminance_500").glob("*.da")
    )
    da_target_illuminance_750_res = list(
        (sub_folder / "da/target_illuminance_750").glob("*.da")
    )

    if (
        len(
            set(
                [
                    len(target_illuminance_res),
                    len(minimum_illuminance_res),
                    len(da_minimum_illuminance_100_res),
                    len(da_minimum_illuminance_300_res),
                    len(da_minimum_illuminance_500_res),
                    len(da_target_illuminance_300_res),
                    len(da_target_illuminance_500_res),
                    len(da_target_illuminance_750_res),
                    len(sensorgrids),
                ]
            )
        )
        == 1
    ):
        return pd.concat(
            [
                load_res(target_illuminance_res),
                load_res(minimum_illuminance_res),
                load_res(da_minimum_illuminance_100_res),
                load_res(da_minimum_illuminance_300_res),
                load_res(da_minimum_illuminance_500_res),
                load_res(da_target_illuminance_300_res),
                load_res(da_target_illuminance_500_res),
                load_res(da_target_illuminance_750_res),
            ],
            axis=1,
            keys=[
                "en17037_target_illuminance",
                "en17037_minimum_illuminance",
                "en17037_da_minimum_illuminance_100",
                "en17037_da_minimum_illuminance_300",
                "en17037_da_minimum_illuminance_500",
                "en17037_da_target_illuminance_300",
                "en17037_da_target_illuminance_500",
                "en17037_da_target_illuminance_750",
            ],
        )

    # determine peak sun hours from WEA
    wea = Wea.from_file(list(annual_daylight_folder.glob("*.wea"))[0])
    global_rad = collection_to_series(wea.global_horizontal_irradiance)
    peak_sun_schedule = (
        global_rad.index.isin(
            global_rad.sort_values()[-int(len(global_rad) / 2) :].index
        )
        .astype(int)
        .tolist()
    )

    results = Results(
        annual_daylight_folder / "results", schedule=None, load_arrays=False
    )
    en17037_to_folder(
        results,
        schedule=peak_sun_schedule,
        sub_folder=sub_folder,
    )

    target_illuminance_res = list(
        (sub_folder / "compliance_level/target_illuminance").glob("*.pf")
    )
    minimum_illuminance_res = list(
        (sub_folder / "compliance_level/minimum_illuminance").glob("*.pf")
    )
    da_minimum_illuminance_100_res = list(
        (sub_folder / "da/minimum_illuminance_100").glob("*.da")
    )
    da_minimum_illuminance_300_res = list(
        (sub_folder / "da/minimum_illuminance_300").glob("*.da")
    )
    da_minimum_illuminance_500_res = list(
        (sub_folder / "da/minimum_illuminance_500").glob("*.da")
    )
    da_target_illuminance_300_res = list(
        (sub_folder / "da/target_illuminance_300").glob("*.da")
    )
    da_target_illuminance_500_res = list(
        (sub_folder / "da/target_illuminance_500").glob("*.da")
    )
    da_target_illuminance_750_res = list(
        (sub_folder / "da/target_illuminance_750").glob("*.da")
    )

    return pd.concat(
        [
            load_res(target_illuminance_res),
            load_res(minimum_illuminance_res),
            load_res(da_minimum_illuminance_100_res),
            load_res(da_minimum_illuminance_300_res),
            load_res(da_minimum_illuminance_500_res),
            load_res(da_target_illuminance_300_res),
            load_res(da_target_illuminance_500_res),
            load_res(da_target_illuminance_750_res),
        ],
        axis=1,
        keys=[
            "en17037_target_illuminance",
            "en17037_minimum_illuminance",
            "en17037_da_minimum_illuminance_100",
            "en17037_da_minimum_illuminance_300",
            "en17037_da_minimum_illuminance_500",
            "en17037_da_target_illuminance_300",
            "en17037_da_target_illuminance_500",
            "en17037_da_target_illuminance_750",
        ],
    )


def postprocess_metrics_sensorgrid(
    model_simulation_folder: Path,
    sensorgrid: SensorGrid,
    metric: str,
    threshold: float = 300,
    udi_min: float = 100,
    udi_max: float = 3000,
) -> pd.DataFrame:
    """Generate or load daylight autonomy results for a sensorgrid.

    Args:
        model_simulation_folder (Path): The path to the model simulation folder.
        sensorgrid (SensorGrid): The sensorgrid to generate or load results for.
        metric (str): The metric to generate or load results for. One of 'da', 'cda', 'udi', 'udi_lower', 'udi_upper'.
        threshold (float, optional): The threshold in lux above which the space is considered to be daylit. Defaults to 300.
        udi_min (float, optional): The minimum value for the useful daylight illuminance (UDI) in lux. Defaults to 100.
        udi_max (float, optional): The maximum value for the useful daylight illuminance (UDI) in lux. Defaults to 3000.

    Returns:
        pd.DataFrame: A pandas DataFrame with daylight autonomy results.
    """

    dd = {
        "da": ("da", "da"),
        "cda": ("cda", "cda"),
        "udi": ("udi", "udi"),
        "udi_lower": ("udi_lower", "udi"),
        "udi_upper": ("udi_upper", "udi"),
    }
    if metric not in dd.keys():
        raise ValueError(
            "The metric must be one of 'da', 'cda', 'udi', 'udi_lower', 'udi_upper'."
        )

    model_simulation_folder = Path(model_simulation_folder)
    annual_daylight_folder = model_simulation_folder / "annual_daylight"
    metrics_folder = annual_daylight_folder / "results" / "metrics" / dd[metric][0]

    if not metrics_folder.exists():
        postprocess_metrics(
            model_simulation_folder,
            threshold=threshold,
            udi_min=udi_min,
            udi_max=udi_max,
        )

    res = metrics_folder / f"{sensorgrid.identifier}.{dd[metric][1]}"
    return load_res(res).squeeze().values


def postprocess_metrics(
    model_simulation_folder: Path,
    threshold: float = 300,
    udi_min: float = 100,
    udi_max: float = 3000,
) -> pd.DataFrame:
    """Post-process annual daylight results to get autonomy metrics.

    Args:
        model_simulation_folder (Path):
            Path to the model simulation folder.
        threshold (float, optional):
            The threshold in lux above which the space is considered to be
            daylit. Defaults to 300.
        udi_min (float, optional):
            The minimum value for the useful daylight illuminance (UDI) in lux.
            Defaults to 100.
        udi_max (float, optional):
            The maximum value for the useful daylight illuminance (UDI) in lux.
            Defaults to 3000.

    Returns:
        pd.DataFrame:
            A pandas DataFrame with autonomy metrics.
    """
    model_simulation_folder = Path(model_simulation_folder)
    annual_daylight_folder = model_simulation_folder / "annual_daylight"
    if not annual_daylight_folder.exists():
        raise ValueError("The given folder does not contain annual daylight results.")
    sensorgrids = get_sensorgrids(model_simulation_folder)

    sub_folder = annual_daylight_folder / "results/metrics"

    # check for existence of results and load if already run
    da_res = list((sub_folder / "da").glob("*.da"))
    cda_res = list((sub_folder / "cda").glob("*.cda"))
    udi_res = list((sub_folder / "udi").glob("*.udi"))
    udi_lower_res = list((sub_folder / "udi_lower").glob("*.udi"))
    udi_upper_res = list((sub_folder / "udi_upper").glob("*.udi"))

    if (
        len(
            set(
                [
                    len(da_res),
                    len(cda_res),
                    len(udi_res),
                    len(udi_lower_res),
                    len(udi_upper_res),
                    len(sensorgrids),
                ]
            )
        )
        == 1
    ):
        return pd.concat(
            [
                load_res(da_res),
                load_res(cda_res),
                load_res(udi_res),
                load_res(udi_lower_res),
                load_res(udi_upper_res),
            ],
            axis=1,
            keys=[
                f"da",
                f"cda",
                f"udi",
                f"udi_lt",
                f"udi_gt",
            ],
        )

    res_folder = annual_daylight_folder / "results"
    cmds = [
        folders.python_exe_path,
        "-m",
        "honeybee_radiance_postprocess",
        "post-process",
        "annual-daylight",
        res_folder.as_posix(),
        "-sf",
        "metrics",
        "-t",
        str(threshold),
        "-lt",
        str(udi_min),
        "-ut",
        str(udi_max),
    ]
    process = subprocess.Popen(
        cmds,
        cwd=res_folder,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.wait()

    da_res = list((sub_folder / "da").glob("*.da"))
    cda_res = list((sub_folder / "cda").glob("*.cda"))
    udi_res = list((sub_folder / "udi").glob("*.udi"))
    udi_lower_res = list((sub_folder / "udi_lower").glob("*.udi"))
    udi_upper_res = list((sub_folder / "udi_upper").glob("*.udi"))
    return pd.concat(
        [
            load_res(da_res),
            load_res(cda_res),
            load_res(udi_res),
            load_res(udi_lower_res),
            load_res(udi_upper_res),
        ],
        axis=1,
        keys=[
            f"da",
            f"cda",
            f"udi",
            f"udi_lt",
            f"udi_gt",
        ],
    )


def postprocess_daylightsaturationpercentage_sensorgrid(
    model_simulation_folder: Path,
    sensorgrid: SensorGrid,
    dsp_min: float = 430,
    dsp_max: float = 4300,
) -> pd.DataFrame:
    """Generate or load daylight saturation percentage results for a sensorgrid.

    Args:
        model_simulation_folder (Path): The path to the model simulation folder.
        sensorgrid (SensorGrid): The sensorgrid to generate or load results for.
        dsp_min (float, optional): The minimum value for the daylight saturation percentage (DSP). Defaults to 430.
        dsp_max (float, optional): The maximum value for the daylight saturation percentage (DSP). Defaults to 4300.

    Returns:
        pd.DataFrame: A pandas DataFrame with daylight saturation percentage results.
    """

    model_simulation_folder = Path(model_simulation_folder)
    annual_daylight_folder = model_simulation_folder / "annual_daylight"
    metrics_folder = annual_daylight_folder / "results" / "metrics" / "dsp"

    if not metrics_folder.exists():
        postprocess_daylightsaturationpercentage(
            model_simulation_folder, dsp_min=dsp_min, dsp_max=dsp_max
        )

    res = metrics_folder / f"{sensorgrid.identifier}.dsp"
    return load_res(res).squeeze().values


def postprocess_daylightsaturationpercentage(
    model_simulation_folder: Path, dsp_min: float = 430, dsp_max: float = 4300
) -> pd.DataFrame:
    model_simulation_folder = Path(model_simulation_folder)
    annual_daylight_folder = model_simulation_folder / "annual_daylight"

    if not annual_daylight_folder.exists():
        raise ValueError("The given folder does not contain annual daylight results.")

    sensorgrids = get_sensorgrids(model_simulation_folder)

    sub_folder = annual_daylight_folder / "results/metrics"

    # check for existence of results and load if already run
    dsp_res = list((sub_folder / "dsp").glob("*.dsp"))

    if (
        len(
            set(
                [
                    len(dsp_res),
                    len(sensorgrids),
                ]
            )
        )
        == 1
    ):
        return pd.concat(
            [
                load_res(dsp_res),
            ],
            axis=1,
            keys=[
                f"dsp",
            ],
        )

    # load files
    npy_files = list(
        (annual_daylight_folder / "results/__static_apertures__/default/total").glob(
            "*.npy"
        )
    )
    ill = load_npy(npy_files)

    # calculate metrics
    gt_min = ill >= dsp_min
    lt_max = ill <= dsp_max
    gt_max = ill > dsp_max
    dsps = (
        (((gt_min & lt_max).sum(axis=0) - gt_max.sum(axis=0)) / len(gt_min)).unstack().T
    ) * 100

    # save to file
    submetrics_folder = sub_folder / "dsp"
    submetrics_folder.mkdir(parents=True, exist_ok=True)

    for gridname, vals in dsps.iteritems():
        vals.dropna().to_csv(
            submetrics_folder / f"{gridname}.dsp", index=False, header=False
        )

    # return dataframe
    return pd.concat(
        [
            dsps,
        ],
        axis=1,
        keys=[
            f"dsp",
        ],
    )


def postprocess_daylightfactor(model_simulation_folder: Path) -> pd.DataFrame:
    """Post-process daylight factor results.

    Args:
        model_simulation_folder (Path):
            Path to the model simulation folder.

    Returns:
        pd.DataFrame:
            A pandas DataFrame with daylight factor metrics.
    """

    model_simulation_folder = Path(model_simulation_folder)
    daylightfactor_folder = model_simulation_folder / "daylight_factor"
    if not daylightfactor_folder.exists():
        raise ValueError("The given folder does not contain daylight factor.")
    sensorgrids = get_sensorgrids(model_simulation_folder)

    sub_folder = daylightfactor_folder / "results"
    # check for existence of results and load if already run
    df_res = list(sub_folder.glob("*.res"))
    if not len(df_res) == len(sensorgrids):
        raise ValueError(
            f"The number of daylight factor results ({len(df_res)}) does not match the number of sensorgrids ({len(sensorgrids)})."
        )

    return pd.concat([load_res(df_res)], axis=1, keys=["df"])


def postprocess_daylightfactor_sensorgrid(
    model_simulation_folder: Path, sensorgrid: SensorGrid
) -> List[float]:
    """Generate or load daylight factor results for a sensorgrid.

    Args:
        model_simulation_folder (Path): The path to the model simulation folder.
        sensorgrid (SensorGrid): The sensorgrid to generate or load results for.

    Returns:
        pd.DataFrame: A pandas DataFrame with daylight factor results.
    """

    model_simulation_folder = Path(model_simulation_folder)
    daylightfactor_folder = model_simulation_folder / "daylight_factor" / "results"
    if not daylightfactor_folder.exists():
        raise ValueError("The given folder does not contain daylight factor results.")

    return (
        load_res(daylightfactor_folder / f"{sensorgrid.identifier}.res")
        .squeeze()
        .values
    )


def postprocess_all(
    model_simulation_folder: Path,
    en17037: bool = True,
    cbdm: bool = True,
    dsp: bool = True,
    daylightfactor: bool = True,
) -> pd.DataFrame:
    """Run all post-processing methods with default settings.

    Args:
        model_simulation_folder (Path): The path to the model simulation folder.

    Returns:
        pd.DataFrame: A pandas DataFrame with all post-processing results.
    """

    if sum([en17037, cbdm, dsp, daylightfactor]) == 0:
        raise ValueError(
            "No post-processing method was selected. Please select at least one method."
        )

    dfs = []

    if en17037:
        try:
            dfs.append(postprocess_en17037(model_simulation_folder))
        except Exception as e:
            warn(f"Failed to run EN 17037 post-processing: {e}")

    if cbdm:
        try:
            dfs.append(postprocess_metrics(model_simulation_folder))
        except Exception as e:
            warn(f"Failed to run CBD metrics post-processing: {e}")

    if dsp:
        try:
            dfs.append(
                postprocess_daylightsaturationpercentage(model_simulation_folder)
            )
        except Exception as e:
            warn(f"Failed to run daylight saturation percentage post-processing: {e}")

    if daylightfactor:
        try:
            dfs.append(postprocess_daylightfactor(model_simulation_folder))
        except Exception as e:
            warn(f"Failed to run daylight factor post-processing: {e}")

    return pd.concat(dfs, axis=1).reorder_levels([1, 0], axis=1).sort_index(axis=1)


def plot_level_by_level(model_simulation_folder: Path) -> None:
    model_simulation_folder = Path(model_simulation_folder)

    # Load model
    model = Model.from_hbjson(
        model_simulation_folder / f"{model_simulation_folder.name}.hbjson"
    )

    # get sensor grids and sort by z-level
    grids = get_sensorgrids(model_simulation_folder=model_simulation_folder)
    level_grids = sensorgrid_groupby_level(grids)

    pbar1 = tqdm(DaylightMetric)
    for metric in pbar1:
        # for each level
        for nn, (level, grids) in enumerate(level_grids.items()):
            pbar1.set_description(
                f"Plotting {metric.name}, level {level:0.2f}m [{nn+1:02d}/{len(level_grids):02d}]"
            )
            fig, ax = plt.subplots(
                1,
                1,
            )

            # plot mesh
            level_vals = []  # < this is used to generate level-wise metrics
            level_pts = []  # < this is used to get the axis limits
            for grid in grids:
                # load/process the metric values to plot
                vals = metric.get_values(
                    sensorgrid=grid, model_simulation_folder=model_simulation_folder
                )
                # create patch collection to render
                pc = sensorgrid_to_patchcollection(
                    sensorgrid=grid, cmap=metric.cmap, norm=metric.norm, zorder=1
                )
                # assign valeus to patch collection
                pc.set_array(vals)
                # place patch collection in plot
                ax.add_collection(pc)
                level_vals.extend(vals)
                level_pts.extend([grid.mesh.min, grid.mesh.max])

            # Plot wireframe, for each of the HB goemetry types (e.g. wall, floor, window, ...)
            for b in HbModelGeometry:
                pc = b.slice_polycollection(
                    model=model, plane=Plane(o=Point3D(0, 0, level))
                )
                j = ax.add_collection(pc)

            # Tidy up the plot a little
            ax.set_aspect("equal")
            ax.autoscale()
            ax.axis("off")

            # narrow to only show the region around the level
            ax.set_xlim(
                min([i.x for i in level_pts]) - 1, max([i.x for i in level_pts]) + 1
            )
            ax.set_ylim(
                min([i.y for i in level_pts]) - 1, max([i.y for i in level_pts]) + 1
            )

            # add the summary stats for the level
            xa, xb = ax.get_xlim()
            ya, yb = ax.get_ylim()
            ax.text(
                xa,
                yb,
                "\n".join(metric.summarize(level_vals)),
                ha="left",
                va="top",
                fontdict=({"family": "monospace"}),
                bbox=dict(boxstyle="square, pad=0", fc="w", alpha=0.75, ec="none"),
            )

            # PLace the legend
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cb = fig.colorbar(
                mappable=ax.get_children()[0], cax=cax, label=f"{metric.name}"
            )
            cb.outline.set_visible(False)

            # Add a title
            ax.set_title(f"z-level: {level:0.2f}m")

            plt.tight_layout()

            # save the figure
            img_dir = model_simulation_folder / "images"
            img_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(img_dir / f"{metric.acronym}_level_{nn:02d}.png", dpi=300)

            plt.close(fig)
