from ladybugtools_toolkit.plot._diurnal import diurnal, textwrap
import pandas as pd
import matplotlib.pyplot as plt


def radiant_cooling_potential(
    dpt: pd.Series, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """Plot radiant cooling potential.

    Args:
        dpt (pd.Series): A pandas series of dry bulb temperature.
        ax (plt.Axes, optional): The matplotlib axes to plot the figure on. Defaults to None.

    Returns:
        plt.Axes: The matplotlib axes.
    """

    if ax is None:
        ax = plt.gca()

    diurnal(series=dpt, ax=ax, period="monthly", zorder=2, **kwargs)

    # ad vertical lines
    [ax.axvline(i * 24, ls=":", c="k") for i in range(12)]

    _temp = (
        dpt.groupby([dpt.index.month, dpt.index.day, dpt.index.hour])
        .mean()
        .reorder_levels([0, 2, 1])
        .unstack()
        .reset_index(drop=True)
        .T
    )
    ax.scatter(
        [_temp.columns.values] * 31,
        _temp.values,
        s=0.5,
        c=kwargs.get("color", "#907090"),
        zorder=1,
        alpha=0.3,
    )

    ax.axhline(20, ls="--", c="k", lw=1)
    ax.annotate(
        textwrap.fill("Underfloor cooling (20°C)", 16),
        xy=(288, 20),
        xycoords="data",
        xytext=(10, 10),
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"
        ),
    )

    ax.axhline(18, ls="--", c="k", lw=1)
    ax.annotate(
        textwrap.fill("Thermally activated building structure (TABS) (18°C)", 20),
        xy=(288, 18),
        xycoords="data",
        xytext=(10, -45),
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="->", connectionstyle="angle,angleA=90,angleB=0,rad=10"
        ),
    )

    ax.axhline(17, ls="--", c="k", lw=1)
    ax.annotate(
        textwrap.fill("Chilled beams/ ceiling (17°C)", 16),
        xy=(288, 17),
        xycoords="data",
        xytext=(10, -90),
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"
        ),
    )
    ax.set_title("Radiant cooling potential")

    return ax
