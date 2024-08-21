"""Helper methods for generating plots."""

# pylint: disable=E0401
import calendar
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .util import consistent_units, contrasting_color, get_color, get_unit

# pylint: enable=E0401


def stacked_bar(
    df: pd.DataFrame,
    ax: plt.Axes = None,
    rule: str = "MS",
    label: bool = True,
    legend: bool = True,
    units: str = None,
) -> plt.Axes:
    """Create a stacked bar chart from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index and columns of data to be plotted. The 
        columns can have units in brackets, which will be used for the y-axis 
        and removed, or will require units to be stated.
    ax : plt.Axes, optional
        Axis to plot the data on. If None, the current axis is used.
    rule : str, optional
        Resampling rule for the data. Default is 'MS' (monthly start).
    label : bool, optional
        Whether to add labels to the bars. Default is True.
    legend : bool, optional
        Whether to add a legend to the plot. Default is True.
    units : str, optional
        Units for the y-axis label. If None, the units are taken from the 
        column names. Default is None.

    Returns
    -------
    plt.Axes
        Axis with the stacked bar chart.
    """

    # resample to target periodicity
    data = df.resample(rule).sum()

    if ax is None:
        ax = plt.gca()

    # get colors based on default color scheme
    colors = [get_color(i) for i in data.columns]

    # check unit consistency
    consistent_units(data.columns)

    # remove units from column names and obtain unit for y-axis label
    unit = get_unit(data.columns[0])
    if unit is None:
        if units is None:
            raise ValueError("Units must be provided for the y-axis label.")
        else:
            unit = units
    else:
        data.columns = [i.split(" (")[0] for i in data.columns]

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

    # format, if annual monthly plot
    if len(data) == 12:
        ax.set_xticklabels([f"{i:%b}" for i in data.index], rotation=0)

        if label:
            # add labels
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

        # comma-thousands y-axis formatter
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    # get the unit for y-axis label
    ax.set_ylabel(unit)

    if legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

    return ax


def pie(
    series: pd.Series, ax: plt.Axes = None, legend: bool = True, label: bool = True, units: str = None, **kwargs
) -> plt.Axes:
    """_"""

    # check unit consistency
    consistent_units(series.index)

    # get unit
    # remove units from series index and obtain unit
    unit = get_unit(series.index[0])
    if unit is None:
        if units is None:
            raise ValueError("Units must be provided if none are found on the variable names.")
        else:
            unit = units
    else:
        series.index = [i.split(" (")[0] for i in series.index]

    if ax is None:
        ax = plt.gca()

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            if pct / 100 > 0.075:
                return f"{pct / 100:.1%}\n({val:,}{unit})"
            return ""
        return my_autopct

    textprops = kwargs.pop("textprops", {"color": "w", "fontsize": "x-small"})
    startangle = kwargs.pop("startangle", 90)
    counterclock = kwargs.pop("counterclock", False)
    colors = kwargs.pop("color", [get_color(i) for i in series.index])
    autopct = kwargs.pop("autopct", make_autopct(series.values) if label else None)
    
    wedges, _, autotexts = ax.pie(
        series,
        startangle=startangle,
        counterclock=counterclock,
        colors=colors,
        autopct=autopct,
        textprops=textprops,
        **kwargs,
    )

    for txt, wdg in zip(autotexts, wedges):
        txt.set_color(contrasting_color(wdg.get_facecolor()))

    if legend:
        ax.legend(
            wedges,
            series.index,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

    return ax


def diurnal(
    data: pd.DataFrame, ax: plt.Axes = None, legend: bool = True, logy: bool = False, units: str = None
) -> plt.Axes:
    """_"""

    if ax is None:
        ax = plt.gca()

    # sort by most to least
    data = data[data.sum(axis=0).sort_values(ascending=False).index]

    # get colors based on default color scheme
    colors = [get_color(i) for i in data.columns]

    # check unit consistency
    consistent_units(data.columns)

    # get unit for y-axis label
    unit = get_unit(data.columns[0])
    if unit is None:
        if units is None:
            raise ValueError("Units must be provided for the y-axis label.")
        else:
            unit = units
    else:
        data.columns = [i.split(" (")[0] for i in data.columns]

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

    if logy:
        ax.set_yscale("log")

    # get the unit for y-axis label
    ax.set_ylabel(unit)

    ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
    ax.set_xticklabels(
        major_ticklabels,
        minor=False,
        ha="left",
    )
    ax.set_xlim(0, 288)
    ax.set_ylim(grp_mean.min().min(), grp_mean.max().max())

    if legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
    plt.grid(which="minor", alpha=0.25)

    return ax


def duration_curve(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    remove_zero: bool = True,
    legend: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot a load duration curve."""

    if ax is None:
        ax = plt.gca()

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # sort by most to least
    data = data[data.sum(axis=0).sort_values(ascending=False).index]

    if remove_zero:
        data[data == 0] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.hist(
            data,
            color=[get_color(i) for i in data.columns],
            label=data.columns,
            **kwargs,
        )

    ax.set_ylabel("Frequency")
    if kwargs.get("density", False):
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
    else:
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_xlabel(
        f'{data.columns[0].split(" (")[1][:-1]} (per hour, {data.index.min():%b %H:%M}-{data.index.max():%b %H:%M})'
    )

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     if orientation == "horizontal":
    #         ax.set_xlabel("Annual hours")
    #         ax.set_ylim(0, np.nanmax(data.values) * 1.05)
    #     else:
    #         ax.set_ylabel("Annual hours")
    #         ax.set_xlim(0, np.nanmax(data.values) * 1.05)

    if legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

    ax.grid(which="minor", alpha=0.25)

    return ax


