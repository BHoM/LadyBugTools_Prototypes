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
import warnings
from ladybugtools_toolkit.ladybug_extension.epw import epw_to_dataframe
from ladybug.wea import Wea
from ladybugtools_toolkit.wind.direction_bins import DirectionBins, cardinality
from ladybugtools_toolkit.plot.utilities import contrasting_color
import textwrap
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Colormap
from matplotlib.figure import Figure
from matplotlib.tri.triangulation import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def spatial_heatmapnew(
    triangulations: List[Triangulation],
    values: List[List[float]],
    ax: plt.Axes = None,
    kwargs_for_tricontourf: Dict[str, Any] = {},
    kwargs_for_colorbar: Dict[str, Any] = {},
    kwargs_for_tricontour: Dict[str, Any] = {},
    kwargs_for_clabel: Dict[str, Any] = {},
) -> plt.Axes:
    # validation
    for tri, zs in list(zip(*[triangulations, values])):
        if len(tri.x) != len(zs):
            raise ValueError(
                "The shape of the triangulations and values given do not match."
            )

    # plot preparation
    if ax is None:
        ax = plt.gca()

    ax.set_aspect("equal")
    ax.axis("off")

    tcls = []
    for tri, zs in list(zip(*[triangulations, values])):
        # add contour heatmap
        tcf = ax.tricontourf(tri, zs, **kwargs_for_tricontourf)

        # add contour lines
        if kwargs_for_tricontour.get("levels", None) is not None:
            tcl = ax.tricontour(tri, zs, **kwargs_for_tricontour)
            tcls.append(tcl)

            # add clabels
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.clabel(tcl, inline=1, **kwargs_for_clabel)

    if len(kwargs_for_colorbar) > 0:
        # add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
        cbar = plt.colorbar(tcf, cax=cax, **kwargs_for_colorbar)
        cbar.outline.set_visible(False)
        for tcl in tcls:
            cbar.add_lines(tcl)

    return ax
