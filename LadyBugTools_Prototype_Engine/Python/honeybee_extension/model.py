from honeybee.model import Model
from honeybee.facetype import Wall, Floor, RoofCeiling, AirBoundary
import matplotlib.pyplot as plt
from ladybug_geometry.geometry3d import Plane, LineSegment3D
from typing import List
from matplotlib.collections import PolyCollection


def slice_geometry(
    hb_objects: List[object], plane: Plane, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """Slice a set of Honeybee objects with a plane and plot their intersections.

    Args:
        hb_objects (List[object]): A set of Honeybee objects to slice. These must have a geometry attribute.
        plane (Plane): A Ladybug 3D Plane object to slice the objects with.
        ax (plt.Axes, optional): The matplotlib axes to plot the intersections on. Defaults to None which uses the current axes.
        **kwargs: Additional keyword arguments to pass to the matplotlib PolyCollection constructor.

    Returns:
        plt.Axes: A matplotlib axes with the intersections plotted.
    """

    if ax is None:
        ax = plt.gca()

    _vertices = []
    for obj in hb_objects:
        segments: List[LineSegment3D] = obj.geometry.intersect_plane(plane)
        if segments is None:
            continue
        _vertices.extend(
            [
                [plane.xyz_to_xy(pt).to_array() for pt in segment.vertices]
                for segment in segments
            ]
        )
    ax.add_artist(
        PolyCollection(
            verts=_vertices,
            closed=False,
            fc=None,
            **kwargs,
        ),
    )

    return ax


def slice_model(model: Model, plane: Plane, ax: plt.Axes = None) -> plt.Axes:
    """Slice a Honeybee model with a plane and plot the intersections.

    Args:
        model (Model): A Honeybee model to slice.
        plane (Plane): A Ladybug 3D Plane object to slice the model with.
        ax (plt.Axes, optional): The matplotlib axes to plot the intersections on. Defaults to None which uses the current axes.

    Returns:
        plt.Axes: A matplotlib axes with the intersections plotted.
    """

    if ax is None:
        ax = plt.gca()

    meta = {
        "wall": {
            "objects": [i for i in model.faces if isinstance(i.type, Wall)],
            "poly_kwargs": {
                "color": "grey",
                "zorder": 4,
                "alpha": 1,
                "lw": 0.5,
            },
        },
        "floor": {
            "objects": [i for i in model.faces if isinstance(i.type, Floor)],
            "poly_kwargs": {
                "color": "black",
                "zorder": 5,
                "alpha": 1,
                "lw": 0.5,
            },
        },
        "roofceiling": {
            "objects": [i for i in model.faces if isinstance(i.type, RoofCeiling)],
            "poly_kwargs": {
                "color": "brown",
                "zorder": 5,
                "alpha": 1,
                "lw": 0.5,
            },
        },
        "airboundary": {
            "objects": [i for i in model.faces if isinstance(i.type, AirBoundary)],
            "poly_kwargs": {
                "color": "pink",
                "zorder": 3,
                "alpha": 1,
                "lw": 0.5,
            },
        },
        "shade": {
            "objects": [i for i in model.shades],
            "poly_kwargs": {
                "color": "green",
                "zorder": 3,
                "alpha": 1,
                "lw": 0.5,
            },
        },
        "aperture": {
            "objects": [i for i in model.apertures],
            "poly_kwargs": {
                "color": "blue",
                "zorder": 6,
                "alpha": 1,
                "lw": 0.5,
            },
        },
    }

    for _, v in meta.items():
        slice_geometry(hb_objects=v["objects"], plane=plane, ax=ax, **v["poly_kwargs"])

    return ax
