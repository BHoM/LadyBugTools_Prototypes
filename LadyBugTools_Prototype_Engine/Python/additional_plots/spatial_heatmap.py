import json
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from warnings import warn

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from honeybee_radiance.sensorgrid import Sensor, SensorGrid
from honeybee_radiance.view import View
from honeybee_radiance_folder.folder import ModelFolder
from ladybug.epw import EPW
from ladybug.wea import Wea
from ladybug_geometry.geometry3d.plane import Plane, Point3D
from ladybugtools_toolkit.honeybee_extension.results import (
    collection_to_series, load_ill, load_npy, load_pts, make_annual)
from ladybugtools_toolkit.ladybug_extension.epw import epw_to_dataframe
from ladybugtools_toolkit.plot.utilities import (contrasting_color,
                                                 create_triangulation)
from ladybugtools_toolkit.wind.direction_bins import DirectionBins, cardinality
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, Colormap
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.tri.triangulation import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import cdist
from tqdm import tqdm


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

def sensorgrid_to_patches(sensor_grid: SensorGrid) -> List[PathPatch]:
    """Takes a HB SensorGrid and returns a list of matplotlib PathPatch objects

    Args:
        sensor_grid (SensorGrid):
            A single HB SensorGrid object

    Returns:
        List[PathPatch]:
            A list of matplotlib PathPatch objects
    """

    mesh = sensor_grid.mesh
    vertices = mesh.face_vertices
    patches = []
    for face in vertices:
        face_vertices = []
        for vertice in face:
            x , y = vertice.x, vertice.y
            face_vertices.append([x,y])
        starting_vertices = face_vertices[0]
        face_vertices.append(starting_vertices)
        path = mpath.Path(face_vertices)
        patch = PathPatch(path, rasterized = True)
        patches.append(patch)
    return patches
  
def spatial_heatmap_patches(
        patches: List[PathPatch],
        values: List[float],
        show_legend: bool = True,
        cmap: Colormap = "viridis",
        colorbar_label: str = "",
        title: str ="",
        legend_min: float = None, # TODO needs adding below
        legend_max: float = None # TODO needs adding below
) -> Figure:
    """Generates a matplotlib Figure composed of a Patch for each value
    similar to LBTs grasshopper spatial heatmap. 

    Args:
        patches (List[PathPatch]): Patches, such as generated from the mesh faces
        values (List[float]): Associated values to colour each patch
        show_legend (bool, optional): Toggle for including legend. Defaults to True.
        cmap (Colormap, optional): Matplotlib colormap for patch / legend colours. Defaults to "viridis".
        colorbar_label (str, optional): Label for the legend. Defaults to "".
        title (str, optional): Main title for the figure. Defaults to "".
        legend_min (float, optional): Minimum legend value. Defaults to None.
        legend_max (float, optional): Maximum legend value. Defaults to None.

    Returns:
        Figure: A matplotlib Figure
    """

    p = PatchCollection(patches=patches, cmap = cmap, alpha=1)
    p.set_array(values)

    fig, ax = plt.subplots()
    ax.add_collection(p)
    ax.autoscale(True)
    ax.axis('equal')
    ax.axis('off')

    if show_legend:
        # Plot colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)

        cbar = plt.colorbar(
            p, cax=cax  # , format=mticker.StrMethodFormatter("{x:04.1f}")
        )
        cbar.outline.set_visible(False)
        cbar.set_label(colorbar_label)

    ax.set_title(title, ha="left", va="bottom", x=0)

    return fig