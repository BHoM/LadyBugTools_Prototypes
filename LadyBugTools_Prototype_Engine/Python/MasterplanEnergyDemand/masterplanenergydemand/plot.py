"""Helper methods for generating plots."""

# pylint: disable=E0401
import calendar
import logging
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from . import FORMATTING
from .utilities import contrasting_color

# pylint: enable=E0401

logger = logging.getLogger(__name__.split(".", maxsplit=1)[0])

def plot_monthly_stacked_bar(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """_"""

    # remove units from dataframe columns (if they exist)
    df = df.rename(columns=lambda x: x.split(" (")[0])

    # resample to monthly
    data = df.resample("MS").sum()

    if ax is None:
        ax = plt.gca()

    # get colors
    colors = [FORMATTING.color[i] for i in data.columns]

    # plot
    data.plot(
        ax=ax,
        kind="bar",
        stacked=True,
        color=colors,
        legend=False,
        zorder=3,
        width=0.8,
    )

    # add labels to each bar segment
    ylims = ax.get_ylim()
    for rect in ax.patches:
        y_value = rect.get_y() + rect.get_height() / 2
        x_value = rect.get_x() + rect.get_width() / 2
        fc = rect.get_facecolor()
        actual_value = rect.get_height()

        if abs(actual_value) / (max(ylims) - min(ylims)) > 0.025:
            if abs(actual_value) < 100:
                val = f"{actual_value:,.1f}"
            else:
                val = f"{actual_value:,.0f}"
            # Create annotation
            ax.text(
                x_value,
                y_value,
                val,
                ha="center",
                va="center",
                fontsize="xx-small",
                color=contrasting_color(fc),
                alpha=0.75,
                zorder=8,
            )

    # add axes formatting
    ax.set_xticklabels([f"{i:%b}" for i in data.index], rotation=0)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )

    return ax


def plot_pie(series: pd.Series, ax: plt.Axes = None) -> plt.Axes:
    """_"""

    if ax is None:
        ax = plt.gca()

    colors = [FORMATTING.color[i] for i in series.index]

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            if pct / 100 > 0.075:
                return f"{pct/100:.2%}\n({val:,})"
            return ""

        return my_autopct

    wedges, _, autotexts = ax.pie(
        series,
        startangle=90,
        counterclock=False,
        colors=colors,
        autopct=make_autopct(series.values),
        textprops={"color": "w", "fontsize": "x-small"},
    )
    ax.legend(wedges, series.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    for txt, wdg in zip(autotexts, wedges):
        txt.set_color(contrasting_color(wdg.get_facecolor()))

    return ax


def plot_diurnal(data: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """_"""

    # remove units from dataframe columns (if they exist)
    data = data.rename(columns=lambda x: x.split(" (")[0])

    colors = [FORMATTING.color[i] for i in data.columns]

    grp_mean = data.groupby([data.index.month, data.index.hour]).mean()

    target_idx = pd.MultiIndex.from_product([range(1, 13, 1), range(24)])
    major_ticks = range(len(target_idx))[::12]
    minor_ticks = range(len(target_idx))[::6]
    major_ticklabels = []
    for i in target_idx:
        if i[1] == 0:
            major_ticklabels.append(f"{calendar.month_abbr[i[0]]}")
        elif i[1] == 12:
            major_ticklabels.append("")

    if ax is None:
        ax = plt.gca()

    for n, (var, vals) in enumerate(grp_mean.items()):
        for month in range(1, 13, 1):
            x_vals = np.arange((month - 1) * 24, (month * 24))
            y_vals = vals.loc[month].tolist()
            ax.plot(
                x_vals,
                y_vals,
                c=colors[n],
                label=var if month == 1 else "_nolegend_",
            )
            ax.axvline(x=x_vals[0], c="k", lw=0.6, ls="-", alpha=0.25)

    ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
    ax.set_xticklabels(
        major_ticklabels,
        minor=False,
        ha="left",
    )
    ax.set_xlim(0, 288)
    ax.set_ylim(grp_mean.min().min(), grp_mean.max().max())
    ax.legend(loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.grid(which="minor", alpha=0.25)

    return ax


def plot_duration_curve(
    data: pd.Series | pd.DataFrame,
    ax: plt.Axes = None,
    remove_zero: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot a duration curve for a given pandas Series of data."""

    if ax is None:
        ax = plt.gca()

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # sort by most to least
    data = data[data.sum(axis=0).sort_values(ascending=False).index]

    density = kwargs.pop("density", False)
    if density:
        raise NotImplementedError("Density is not yet implemented for duration curves.")

    orientation = kwargs.pop("orientation", "horizontal")

    # get the defaults
    color = [FORMATTING.color[i] for i in data]
    label = [kwargs.pop("label", i) for i in data]
    # cumulative = kwargs.pop("cumulative", -1)

    if remove_zero:
        data[data == 0] = np.nan

    ax.hist(
        data,
        color=color,
        density=kwargs.pop("density", False),
        bins=kwargs.pop("bins", 100),
        label=label,
        histtype=kwargs.pop("histtype", "step"),
        orientation=orientation,
        cumulative=kwargs.pop("cumulative", -1),
        **kwargs,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if orientation == "horizontal":
            ax.set_xlabel("Annual hours")
            ax.set_ylim(0, np.nanmax(data.values) * 1.05)
        else:
            ax.set_ylabel("Annual hours")
            ax.set_xlim(0, np.nanmax(data.values) * 1.05)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.grid(which="minor", alpha=0.25)

    plt.tight_layout()

    return ax


def plot_heating_cooling_series(
    heating: pd.Series, cooling: pd.Series, ax: plt.Axes = None, kind: str = "line"
) -> plt.Axes:
    """Plot heating and cooling series on the same axis."""

    if ax is None:
        ax = plt.gca()

    if kind == "line":
        ax.plot(heating, label="Heating", c=FORMATTING.color["Heating"], lw=0.5)
        ax.plot(cooling, label="Cooling", c=FORMATTING.color["Cooling"], lw=0.5)
    elif kind == "stacked":
        ax.stackplot(
            heating.index,
            heating,
            cooling,
            labels=["Heating", "Cooling"],
            colors=[FORMATTING.color["Heating"], FORMATTING.color["Cooling"]],
        )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_xlim(
        min(heating.index.min(), cooling.index.min()),
        max(heating.index.max(), cooling.index.max()),
    )
    ax.set_ylim(0, ax.get_ylim()[-1])

    return ax


def create_legend(ax: plt.Axes = None, legend_type: str = "energy") -> plt.Axes:
    """Create a legend from the FORMATTING color dictionary."""

    if legend_type not in ["energy", "thermal"]:
        raise ValueError('The "type" argument must be one of "energy" or "thermal".')

    if legend_type == "energy":
        variables = [
            "Heating",
            "Cooling",
            "Service Hot Water",
            "Lighting",
            "Mechanical Ventilation",
            "Electric Equipment",
            "Gas Equipment",
            "Fans",
            "Pumps",
        ]
    elif legend_type == "thermal":
        variables = [
            "Heating",
            "Cooling",
            "Solar",
            "Service Hot Water",
            "Lighting",
            "Infiltration",
            "Mechanical Ventilation",
            "Window Conduction",
            "Opaque Conduction",
            "Electric Equipment",
            "People",
            "Gas Equipment",
            "Storage",
        ]

    if ax is None:
        ax = plt.gca()

    for k, v in FORMATTING.color.items():
        if k in variables:
            ax.bar(0, 0, color=v, label=k)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    ax.axis("off")

    return ax
