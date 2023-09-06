import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ladybug.epw import EPW
from pathlib import Path
from typing import Union
from ladybugtools_toolkit.honeybee_extension.results import (
    load_npy,
    load_ill,
    make_annual,
    load_pts,
    collection_to_series,
)
from ladybugtools_toolkit.ladybug_extension.epw import epw_to_dataframe
from ladybug.wea import Wea
from ladybugtools_toolkit.wind.direction_bins import DirectionBins, cardinality
from ladybugtools_toolkit.plot.utilities import contrasting_color
import textwrap
from tqdm import tqdm


def directional_irradiance_matrix(
    epw_file: str,
    altitude: float = 0,
    n_directions: int = 32,
    ground_reflectance: float = 0.2,
    isotropic: bool = True,
) -> pd.DataFrame:
    """Create a matrix of hourly irradiance values for a given epw file.

    Args:
        epw_file (str): The path to an epw file.
        altitude (float, optional): The altitude of the facade. Defaults to 0.
        n_directions (int, optional): The number of directions to calculate. Defaults to 32.
        ground_reflectance (float, optional): The ground reflectance. Defaults to 0.2.
        isotropic (bool, optional): Whether to use isotropic sky. Defaults to True.

    Returns:
        pd.DataFrame: A matrix of hourly irradiance values.
    """

    wea = Wea.from_epw_file(epw_file)
    db = DirectionBins(directions=n_directions)

    res = []
    pbar = tqdm(db.midpoints)
    for az in pbar:
        pbar.set_description(
            f"Calculating insolation for {az:0.02f} azimuth, {altitude:0.02f} altitude"
        )
        res.append(
            wea.directional_irradiance(
                azimuth=az,
                altitude=altitude,
                ground_reflectance=ground_reflectance,
                isotropic=isotropic,
            )
        )

    res = (
        pd.concat(
            [
                pd.concat(
                    [collection_to_series(i) for i in j],
                    axis=1,
                    keys=["total", "direct", "diffuse", "reflected"],
                )
                for j in res
            ],
            axis=1,
            keys=db.midpoints,
        )
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1)
    )

    return res


def format_polar_plot(ax: plt.Axes) -> plt.Axes:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # format plot area
    ax.spines["polar"].set_visible(False)
    ax.grid(True, which="both", ls="--", zorder=0, alpha=0.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    plt.setp(ax.get_yticklabels(), fontsize="small")
    ax.set_xticks(np.radians((0, 90, 180, 270)), minor=False)
    ax.set_xticklabels(("N", "E", "S", "W"), minor=False, **{"fontsize": "medium"})
    ax.set_xticks(
        np.radians(
            (22.5, 45, 67.5, 112.5, 135, 157.5, 202.5, 225, 247.5, 292.5, 315, 337.5)
        ),
        minor=True,
    )
    ax.set_xticklabels(
        (
            "NNE",
            "NE",
            "ENE",
            "ESE",
            "SE",
            "SSE",
            "SSW",
            "SW",
            "WSW",
            "WNW",
            "NW",
            "NNW",
        ),
        minor=True,
        **{"fontsize": "x-small"},
    )
    ax.set_yticklabels([])


def annual_surface_insolation(
    directional_irradiance_matrix: pd.DataFrame,
    ax: plt.Axes = None,
    labelling: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    width
    cmap
    vmin
    vmax
    """

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    format_polar_plot(ax)

    cmap = plt.get_cmap(
        kwargs.pop(
            "cmap",
            "Spectral_r",
        )
    )
    width = kwargs.pop("width", 0.9)
    vmin = kwargs.pop("vmin", 0)

    # plot data
    data = (directional_irradiance_matrix.total.sum(axis=0)) / 1000
    vmax = kwargs.pop("vmax", data.max())
    colors = [cmap(i) for i in np.interp(data.values, [vmin, vmax], [0, 1])]
    thetas = np.deg2rad(data.index)
    radiis = data.values
    bars = ax.bar(
        thetas,
        radiis,
        zorder=2,
        width=width * np.deg2rad(360 / len(thetas)),
        color=colors,
    )

    # colorbar
    if (data.min() < vmin) and (data.max() > vmax):
        extend = "both"
    elif data.min() < vmin:
        extend = "min"
    elif data.max() > vmax:
        extend = "max"
    else:
        extend = "neither"
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(
        sm, ax=ax, orientation="vertical", label="kWh/m$^2$", extend=extend
    )
    cb.outline.set_visible(False)

    # labelling
    if labelling:
        for rect, idx, val, colr in list(zip(*[bars, data.index, data.values, colors])):
            if len(set(data.values)) == 1:
                ax.text(
                    0,
                    0,
                    textwrap.fill(
                        f"{val:,.0f}kWh/m$^2$ peak annual total insolation",
                        16,
                    ),
                    ha="center",
                    va="center",
                    bbox=dict(ec="none", fc="w", alpha=0.5, boxstyle="round,pad=0.3"),
                )
                break
            if val == data.max():
                rect.set_edgecolor("k")
            if val > data.max() / 1.5:
                ax.text(
                    np.deg2rad(idx),
                    val,
                    f" {val:,.0f}Wh/m$^2$ ",
                    rotation_mode="anchor",
                    rotation=(-idx + 90) if idx < 180 else 180 + (-idx + 90),
                    ha="right" if idx < 180 else "left",
                    va="center",
                    fontsize="xx-small",
                    c=contrasting_color(colr)
                    # bbox=dict(ec="none", fc="w", alpha=0.5, boxstyle="round,pad=0.3"),
                )

    title = [
        "Annual total insolation",
        kwargs.pop(
            "title",
            None,
        ),
    ]
    ax.set_title("\n".join([i for i in title if i is not None]))

    return ax


def peak_surface_insolation(
    directional_irradiance_matrix: pd.DataFrame,
    ax: plt.Axes = None,
    labelling: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    width
    cmap
    vmin
    vmax
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    format_polar_plot(ax)

    cmap = plt.get_cmap(
        kwargs.pop(
            "cmap",
            "Spectral_r",
        )
    )
    width = kwargs.pop("width", 0.9)
    vmin = kwargs.pop("vmin", 0)

    # plot data
    data = pd.concat(
        [
            directional_irradiance_matrix.total.idxmax(),
            directional_irradiance_matrix.total.max(),
        ],
        axis=1,
        keys=["time", "val"],
    )
    vmax = kwargs.pop("vmax", data.val.max())
    colors = [cmap(i) for i in np.interp(data.val, [vmin, vmax], [0, 1])]
    thetas = np.deg2rad(data.index)
    radiis = data.val
    bars = ax.bar(
        thetas,
        radiis,
        zorder=2,
        width=width * np.deg2rad(360 / len(thetas)),
        color=colors,
    )

    # colorbar
    if (data.val.min() < vmin) and (data.val.max() > vmax):
        extend = "both"
    elif data.val.min() < vmin:
        extend = "min"
    elif data.val.max() > vmax:
        extend = "max"
    else:
        extend = "neither"
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(
        sm, ax=ax, orientation="vertical", label="Wh/m$^2$", extend=extend
    )
    cb.outline.set_visible(False)

    # labelling
    if labelling:
        for rect, idx, time, val, colr in list(
            zip(*[bars, data.index, data.time.values, data.val.values, colors])
        ):
            if len(data.val.unique()) == 1:
                ax.text(
                    0,
                    0,
                    textwrap.fill(
                        f"{val:,.0f}Wh/m$^2$ peak insolation at {pd.Timestamp(time):%b %d %H:%M}",
                        16,
                    ),
                    ha="center",
                    va="center",
                    bbox=dict(ec="none", fc="w", alpha=0.5, boxstyle="round,pad=0.3"),
                )
                break
            if val == data.val.max():
                rect.set_edgecolor("k")
            if val > data.val.max() / 1.5:
                ax.text(
                    np.deg2rad(idx),
                    val,
                    f" {val:,.0f}Wh/m$^2$ {pd.Timestamp(time):%b %d %H:%M} ",
                    rotation_mode="anchor",
                    rotation=(-idx + 90) if idx < 180 else 180 + (-idx + 90),
                    ha="right" if idx < 180 else "left",
                    va="center",
                    fontsize="xx-small",
                    c=contrasting_color(colr)
                    # bbox=dict(ec="none", fc="w", alpha=0.5, boxstyle="round,pad=0.3"),
                )

    title = [
        "Peak insolation",
        kwargs.pop(
            "title",
            None,
        ),
    ]
    ax.set_title("\n".join([i for i in title if i is not None]))

    return ax
