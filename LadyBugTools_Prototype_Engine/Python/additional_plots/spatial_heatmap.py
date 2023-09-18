import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from warnings import warn

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_radiance.view import View
from honeybee_radiance_folder.folder import ModelFolder
from ladybug_geometry.geometry3d.plane import Plane, Point3D
from ladybugtools_toolkit.plot.utilities import create_triangulation
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.tri.triangulation import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import cdist
from ladybugtools_toolkit.categorical import Categorical


def sensorgrid_to_patches(sensor_grid: SensorGrid) -> List[mpatches.PathPatch]:
    """Takes a HB SensorGrid and returns a list of matplotlib PathPatch objects

    Args:
        sensor_grid (SensorGrid):
            A single HB SensorGrid object

    Returns:
        List[PathPatch]:
            A list of matplotlib PathPatch objects
    """

    if sensor_grid.mesh is None:
        raise ValueError(
            "sensorgrid must have a mesh. This would have been assigned when the sensorgrid was created."
        )

    warn(
        "This method assumes that the sensorgrid is planar and that each mesh face is aligned XY."
    )

    mesh = sensor_grid.mesh
    vertices = mesh.face_vertices
    patches = []
    for face in vertices:
        face_vertices = []
        for vertice in face:
            x, y = vertice.x, vertice.y
            face_vertices.append([x, y])
        starting_vertices = face_vertices[0]
        face_vertices.append(starting_vertices)
        path = mpath.Path(face_vertices)
        patch = PathPatch(path, rasterized=True)
        patches.append(patch)
    return patches


def spatial_heatmap_categorical(
    triangulations: List[Triangulation],
    values: List[List[float]],
    categorical: Categorical,
    ax: plt.Axes = None,
    kwargs_for_legend: Dict[str, Any] = {"bbox_to_anchor": (1, 1), "loc": "upper left"},
    kwargs_for_tricontour: Dict[str, Any] = {},
    kwargs_for_clabel: Dict[str, Any] = {},
) -> plt.Axes:
    for tri, zs in list(zip(*[triangulations, values])):
        validate_triangulation_values(triangulation=tri, values=zs)

    if ax is None:
        ax = plt.gca()

    ax.set_aspect("equal")
    ax.axis("off")

    tcls = []
    for tri, zs in list(zip(*[triangulations, values])):
        tcf = ax.tricontourf(tri, zs, levels=len(categorical), cmap=categorical.cmap)

        if kwargs_for_tricontour.get("levels", None) is not None:
            tcl = ax.tricontour(tri, zs, **kwargs_for_tricontour)
            tcls.append(tcl)

            # add clabels
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.clabel(tcl, inline=1, **kwargs_for_clabel)

    # create a legend for the contour set
    artists = [
        mpatches.Patch(color=color, edgecolor=None) for color in categorical.colors
    ]
    labels = categorical.bin_names
    ti = kwargs_for_legend.pop("title", categorical.name)
    ax.legend(artists, labels, title=ti, **kwargs_for_legend)

    return ax


def spatial_heatmapnew(
    triangulations: List[Triangulation],
    values: List[List[float]],
    ax: plt.Axes = None,
    kwargs_for_tricontourf: Dict[str, Any] = {},
    kwargs_for_colorbar: Dict[str, Any] = {},
    kwargs_for_tricontour: Dict[str, Any] = {},
    kwargs_for_clabel: Dict[str, Any] = {},
) -> plt.Axes:
    for tri, zs in list(zip(*[triangulations, values])):
        validate_triangulation_values(triangulation=tri, values=zs)

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
        if ("colors" in kwargs_for_tricontourf) and ("norm" in kwargs_for_tricontourf):
            artists, labels = tcf.legend_elements(str_format="{:2.1f}".format)
        # add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
        cbar = plt.colorbar(tcf, cax=cax, **kwargs_for_colorbar)
        cbar.outline.set_visible(False)
        for tcl in tcls:
            cbar.add_lines(tcl)

    return ax


def spatial_heatmap_patches(
    patches: List[mpatches.PathPatch],
    values: List[float],
    show_legend: bool = True,
    cmap: Colormap = "viridis",
    colorbar_label: str = "",
    title: str = "",
    legend_min: float = None,  # TODO needs adding below
    legend_max: float = None,  # TODO needs adding below
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

    p = PatchCollection(patches=patches, cmap=cmap, alpha=1)
    p.set_array(values)

    fig, ax = plt.subplots()
    ax.add_collection(p)
    ax.autoscale(True)
    ax.axis("equal")
    ax.axis("off")

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
