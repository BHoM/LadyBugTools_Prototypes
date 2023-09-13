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

from honeybee_radiance.sensorgrid import SensorGrid, Sensor
from honeybee_radiance.view import View
from ladybugtools_toolkit.plot.utilities import create_triangulation
from honeybee_radiance_folder.folder import ModelFolder
import json
from ladybug_geometry.geometry3d.plane import Plane, Point3D
from scipy.spatial.distance import cdist
from warnings import warn


def sensorgrids_from_results_folder(results_folder: Path) -> Tuple[SensorGrid]:
    """Create a list of sensorgrids from a results folder.

    Args:
        results_folder (Path): A directory contianing radiance simulation results from a lbt-recipe.

    Returns:
        Tuple[SensorGrid]: A tuple of honeybee-radiance SensorGrids.
    """
    if not isinstance(results_folder, Path):
        results_folder = Path(results_folder)

    # check that one of the following directories exists
    res_dirs = [
        "annual_daylight",
        "annual_irradiance",
        "cumulative_radiation",
        "daylight_factor",
        "direct_sun_hours",
        "sky_view",
        "point_in_time_grid",
    ]
    for res_dir in res_dirs:
        if (results_folder / res_dir).exists():
            # load pts if they exist
            sensor_grids = []
            for i in ModelFolder(project_folder=results_folder / res_dir).grid_files():
                sensor_grids.append(SensorGrid.from_file(results_folder / res_dir / i))
            return sensor_grids

    raise ValueError("No grid files found in the given folder.")


def views_from_results_folder(results_folder: Path) -> Tuple[View]:
    """Create a list of views from a results folder.

    Args:
        results_folder (Path): A directory contianing radiance simulation results from a lbt-recipe.

    Returns:
        Tuple[SensorGrid]: A tuple of honeybee-radiance Views.
    """

    if not isinstance(results_folder, Path):
        results_folder = Path(results_folder)

    # check that one of the following directories exists
    res_dirs = ["point_in_time_view"]
    for res_dir in res_dirs:
        if (results_folder / res_dir).exists():
            # load views if they exist
            return [
                View.from_file(results_folder / res_dir / i)
                for i in ModelFolder(
                    project_folder=results_folder / res_dir
                ).view_files()
            ]

    raise ValueError("No view files found in the given folder.")


def is_sensorgrid_planar(sensorgrid: SensorGrid) -> bool:
    """Check if a sensorgrid is planar.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid.

    Returns:
        bool: True if the sensorgrid is planar, False otherwise.
    """

    # create plane
    plane = Plane.from_three_points(
        *[Point3D.from_array(i.pos) for i in sensorgrid.sensors[0:3]]
    )
    for sensor in sensorgrid.sensors:
        if not np.isclose(
            a=plane.distance_to_point(point=Point3D.from_array(sensor.pos)), b=0
        ):
            return False
    return True


def plane_from_sensorgrid(sensorgrid: SensorGrid) -> Plane:
    """Create a plane from a sensorgrid.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid.

    Returns:
        Plane: A ladybug-geometry Plane.
    """

    plane = Plane.from_three_points(
        *[Point3D.from_array(i.pos) for i in sensorgrid.sensors[0:3]]
    )

    if not is_sensorgrid_planar(sensorgrid=sensorgrid):
        raise ValueError("sensorgrid must be planar to create a plane.")

    return plane


def estimate_sensorgrid_spacing(sensorgrid: SensorGrid) -> float:
    """Estimate the spacing of a sensorgrid.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid.

    Returns:
        float: The estimated spacing (between sensors) of the sensorgrid.
    """
    if not isinstance(sensorgrid, SensorGrid):
        raise ValueError("sensorgrid must be a honeybee-radiance SensorGrid.")

    if not is_sensorgrid_planar(sensorgrid):
        raise ValueError("sensorgrid must be planar.")

    pts = np.array([i.pos for i in sensorgrid.sensors])

    return np.sort(np.sort(cdist(pts, pts))[:, 1])[-1]


def triangulation_from_sensorgrid(
    sensorgrid: SensorGrid, alpha_adjust: float = 0.1
) -> Triangulation:
    """Create matploltib triangulation from a SensorGrid object.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid.
        alpha_adjust (float, optional): A value to adjust the alpha value of the triangulation. Defaults to 0.1.

    Returns:
        Triangulation: A matplotlib triangulation.
    """

    warn("This method only currently works for sensor grids in XY plane!")

    alpha = estimate_sensorgrid_spacing(sensorgrid=sensorgrid) + alpha_adjust
    plane = plane_from_sensorgrid(sensorgrid=sensorgrid)

    # create triangulation
    x, y, z = np.array([i.pos for i in sensorgrid.sensors]).T
    return create_triangulation(x, y, alpha=alpha)


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
