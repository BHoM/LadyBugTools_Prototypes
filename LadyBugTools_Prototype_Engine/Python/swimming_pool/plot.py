import pandas as pd
import matplotlib.pyplot as plt


def plot_timeseries(heat_balance_df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """Create a timeseries plot.

    Args:
        heat_balance_df (pd.DataFrame): DataFrame containing heat balance results.

    Returns:
        plt.Axes: A matplotlib Axes object.
    """

    if ax is None:
        ax = plt.gca()

    heat_balance_df.plot(ax=ax, ylabel="W")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

    return ax


def plot_monthly_balance(
    heat_balance_df: pd.DataFrame, ax: plt.Axes = None
) -> plt.Axes:
    """Create a monthly heat balance plot.

    Args:
        heat_balance_df (pd.DataFrame): DataFrame containing heat balance results.

    Returns:
        plt.Axes: A matplotlib Axes object.
    """

    if ax is None:
        ax = plt.gca()

    (heat_balance_df.resample("MS").sum() / 1000).plot(
        ax=ax, kind="bar", stacked=True, width=0.9, ylabel="kWh"
    )
    ax.set_xticklabels(
        [t.strftime("%b") for t in heat_balance_df.resample("MS").sum().index],
        rotation=0,
    )
    ax.set_xlim(-0.5, 11.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

    return ax
