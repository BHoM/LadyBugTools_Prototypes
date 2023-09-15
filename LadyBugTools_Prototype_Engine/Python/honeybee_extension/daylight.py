"""Methods for post-processing honeybee-radiance daylight results."""

from honeybee_radiance.sensorgrid import SensorGrid, Sensor
from pathlib import Path
from ladybugtools_toolkit.honeybee_extension.results import (
    load_ill,
    load_npy,
    load_res,
    load_pts,
    make_annual,
)
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
from .sensorgrid import get_sensorgrids
from ladybugtools_toolkit.categorical.categories import Categorical
import subprocess

EN17037_ILLUMINANCE_CATEGORIES = Categorical(
    bins=(-np.inf, 0, 1, 2, 3),
    bin_names=("Non-compliant", "Minimum", "Medium", "High"),
    colors=("#1f78b4", "#a6cee3", "#b2df8a", "#33a02c"),
    name="EN 17037 Target Illuminance",
)


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
                "da_minimum_illuminance_100",
                "da_minimum_illuminance_300",
                "da_minimum_illuminance_500",
                "da_target_illuminance_300",
                "da_target_illuminance_500",
                "da_target_illuminance_750",
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
            "da_minimum_illuminance_100",
            "da_minimum_illuminance_300",
            "da_minimum_illuminance_500",
            "da_target_illuminance_300",
            "da_target_illuminance_500",
            "da_target_illuminance_750",
        ],
    )


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
                f"da_{threshold}",
                f"cda_{threshold}",
                f"udi_{udi_min}_{udi_max}",
                f"udi_lt{udi_min}",
                f"udi_gt{udi_max}",
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
            f"da_{threshold}",
            f"cda_{threshold}",
            f"udi_{udi_min}_{udi_max}",
            f"udi_lt{udi_min}",
            f"udi_gt{udi_max}",
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
