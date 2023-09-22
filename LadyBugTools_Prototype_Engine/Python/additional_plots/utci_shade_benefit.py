from ladybugtools_toolkit.categorical.categories import Categorical
from ladybugtools_toolkit.plot.utilities import add_bar_labels
from ladybugtools_toolkit.plot._heatmap import heatmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from calendar import month_abbr
import textwrap
from ladybugtools_toolkit.external_comfort.utci.calculate import utci
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ladybugtools_toolkit.external_comfort.simulate import SimulationResult

from ladybugtools_toolkit.external_comfort.utci.postprocess import (
    shade_benefit_category,
)


def categorical_monthly_histogram(
    series: pd.Series, categorical: Categorical, ax: plt.Axes = None
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    t = categorical.timeseries_summary_monthly(series, density=True).T
    t.columns = [month_abbr[i] for i in t.columns]
    t.T.plot.bar(
        ax=ax,
        stacked=True,
        color=categorical.colors,
        legend=False,
        width=1,
    )
    ax.set_xlabel(None)
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(t.columns, ha="center", rotation=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    if True:
        add_bar_labels(ax, orientation="vertical", threshold=0.05)

    return ax


def shade_benefit_heatmap_histogram(simulation_result: SimulationResult) -> plt.Figure:
    utci_shaded = utci(
        air_temperature=simulation_result.epw.dry_bulb_temperature,
        relative_humidity=simulation_result.epw.relative_humidity,
        wind_speed=simulation_result.epw.wind_speed,
        mean_radiant_temperature=simulation_result.shaded_mean_radiant_temperature,
    )
    utci_unshaded = utci(
        air_temperature=simulation_result.epw.dry_bulb_temperature,
        relative_humidity=simulation_result.epw.relative_humidity,
        wind_speed=simulation_result.epw.wind_speed,
        mean_radiant_temperature=simulation_result.unshaded_mean_radiant_temperature,
    )

    shade_benefit = shade_benefit_category(
        shaded_utci=utci_shaded, unshaded_utci=utci_unshaded, comfort_limits=(9, 26)
    )
    cat = pd.Categorical(shade_benefit)

    SHADE_BENEFIT_CATEGORIES = Categorical(
        bins=(0, 1, 2, 3, np.inf),
        bin_names=cat.categories,
        colors=("#00A499", "#5D822D", "#EE7837", "#585253"),
        name="UTCI shade benefit",
    )

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    spec = fig.add_gridspec(
        ncols=1, nrows=2, width_ratios=[1], height_ratios=[5, 2], hspace=0.0
    )
    heatmap_ax = fig.add_subplot(spec[0, 0])
    histogram_ax = fig.add_subplot(spec[1, 0])
    categorical_monthly_histogram(
        pd.Series(cat.codes, index=shade_benefit.index),
        SHADE_BENEFIT_CATEGORIES,
        ax=histogram_ax,
    )
    heatmap(
        pd.Series(cat.codes, index=shade_benefit.index),
        cmap=SHADE_BENEFIT_CATEGORIES.cmap,
        norm=SHADE_BENEFIT_CATEGORIES.norm,
        extend="both",
        ax=heatmap_ax,
        show_colorbar=False,
    )

    divider = make_axes_locatable(histogram_ax)
    colorbar_ax = divider.append_axes("bottom", size="20%", pad=0.65)
    cb = fig.colorbar(
        mappable=heatmap_ax.get_children()[0],
        cax=colorbar_ax,
        orientation="horizontal",
        drawedges=False,
        extend="both",
    )
    cb.outline.set_visible(False)
    for bin_name, interval in list(
        zip(
            *[
                SHADE_BENEFIT_CATEGORIES.bin_names,
                SHADE_BENEFIT_CATEGORIES.interval_index,
            ]
        )
    ):
        if np.isinf(interval.left):
            ha = "right"
            position = interval.right
        elif np.isinf(interval.right):
            ha = "left"
            position = interval.left
        else:
            ha = "center"
            position = np.mean([interval.left, interval.right])

        colorbar_ax.text(
            position,
            1,
            textwrap.fill(bin_name, 11),
            ha=ha,
            va="bottom",
            size="small",
            # transform=colorbar_ax.transAxes,
        )

    return fig
