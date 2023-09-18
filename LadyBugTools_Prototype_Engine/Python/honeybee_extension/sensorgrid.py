from typing import List, Tuple, Dict
from warnings import warn

from honeybee_radiance_folder.folder import ModelFolder
from honeybee.model import Model
from honeybee_radiance.sensorgrid import SensorGrid
from ladybug_geometry.bounding import bounding_rectangle
from ladybug_geometry.geometry3d import Plane, Point3D, Vector3D, Mesh3D
from ladybug_geometry.geometry2d import Mesh2D, Point2D
from ladybugtools_toolkit.plot.utilities import create_triangulation
from matplotlib.tri.triangulation import Triangulation
from pathlib import Path
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np


def find_pts_files(model_simulation_folder: Path) -> List[Path]:
    """Find pts files in a results folder.

    Args:
        model_simulation_folder (Path): A directory containing radiance simulation results from a lbt-recipe.

    Returns:
        List[Path]: A list of pts files.

    Raises:
        ValueError: If no pts files are found in the given folder.
    """

    if not isinstance(model_simulation_folder, Path):
        model_simulation_folder = Path(model_simulation_folder)

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
        if (model_simulation_folder / res_dir).exists():
            pts_files = []
            for i in ModelFolder(
                project_folder=model_simulation_folder / res_dir
            ).grid_files():
                pts_files.append(model_simulation_folder / res_dir / i)
            return tuple(pts_files)

    raise ValueError(f"No pts files found in {model_simulation_folder}.")


def get_sensorgrids(model_simulation_folder: Path) -> Tuple[SensorGrid]:
    """Load sensorgrids from a model residing within a results folder.

    Args:
        model_simulation_folder (Path): A directory containing radiance simulation results from a lbt-recipe.

    Returns:
        Tuple[SensorGrid]: A tuple of honeybee-radiance SensorGrids.
    """

    model_simulation_folder = Path(model_simulation_folder)
    model = Model.from_hbjson(
        model_simulation_folder / f"{model_simulation_folder.name}.hbjson"
    )
    return model.properties.radiance.sensor_grids


def get_sensorgrid_by_name(model: Model, name: str) -> SensorGrid:
    """Find and return a sensorgrid by name.

    Args:
        model_simulation_folder (Path): The folder containing the sensorgrid.
        name (str): The name of the sensorgrid.

    Returns:
        SensorGrid: A honeybee-radiance SensorGrid.
    """

    for grid in model.properties.radiance.sensor_grids:
        if grid.identifier == name:
            return grid

    raise ValueError(f"Sensorgrid with name {name} not found in {model}.")


def get_triangulations(model_simulation_folder: Path) -> Tuple[Triangulation]:
    """Create a list of triangulations from a results folder.

    Args:
        model_simulation_folder (Path): A directory contianing radiance simulation results from a lbt-recipe.

    Returns:
        Tuple[Triangulation]: A tuple of matplotlib triangulations.
    """

    grids = get_sensorgrids(model_simulation_folder=model_simulation_folder)
    return tuple([sensorgrid_to_triangulation(grid) for grid in grids])


def sensorgrid_groupby_level(
    sensorgrids: List[SensorGrid],
) -> Dict[float, List[SensorGrid]]:
    """Group sensorgrids by their level.

    Args:
        sensorgrids (List[SensorGrid]): A list of honeybee-radiance SensorGrids.

    Returns:
        Dict[float, List[SensorGrid]]: A dictionary of lists of honeybee-radiance SensorGrids.
    """

    d = {}
    for grid in sensorgrids:
        level = grid.sensors[0].pos[-1]
        if not sensorgrid_is_planar(grid):
            warn(
                f"{grid} is not planar. This may cause issues when grouping by level. It will be assumed to be at level {level}."
            )
        if level not in d:
            d[level] = [grid]
        else:
            d[level].append(grid)

    return d


def sensorgrid_to_array(sensorgrid: SensorGrid) -> np.ndarray:
    """Convert a honeybee-radiance SensorGrid to a numpy array.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid.

    Returns:
        np.ndarray: A numpy array of the sensorgrid.
    """

    return np.array([i.pos for i in sensorgrid.sensors])


def sensorgrid_is_planar(sensorgrid: SensorGrid) -> bool:
    """Check if a sensorgrid is planar.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid.

    Returns:
        bool: True if the sensorgrid is planar, False otherwise.
    """

    plane = Plane.from_three_points(
        *[Point3D.from_array(i.pos) for i in sensorgrid.sensors[0:3]]
    )
    for sensor in sensorgrid.sensors:
        if not np.isclose(
            a=plane.distance_to_point(point=Point3D.from_array(sensor.pos)), b=0
        ):
            return False
    return True


def sensorgrid_to_plane(sensorgrid: SensorGrid) -> Plane:
    """Create a plane from a sensorgrid.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid.

    Returns:
        Plane: A ladybug-geometry Plane.
    """

    plane = Plane.from_three_points(
        *[Point3D.from_array(i.pos) for i in np.random.choice(sensorgrid.sensors, 3)]
    )

    if not sensorgrid_is_planar(sensorgrid=sensorgrid):
        raise ValueError("sensorgrid must be planar to create a plane.")

    return plane


def sensorgrid_spacing(sensorgrid: SensorGrid) -> float:
    """Estimate the spacing of a sensorgrid.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid.

    Returns:
        float: The estimated spacing (between sensors) of the sensorgrid.
    """
    if not isinstance(sensorgrid, SensorGrid):
        raise ValueError("sensorgrid must be a honeybee-radiance SensorGrid.")

    if not sensorgrid_is_planar(sensorgrid):
        raise ValueError("sensorgrid must be planar.")

    pts = sensorgrid_to_array(sensorgrid=sensorgrid)

    return np.sort(np.sort(cdist(pts, pts))[:, 1])[-1]


def sensorgrid_to_triangulation(
    sensorgrid: SensorGrid,
    alpha_adjust: float = 0.1,
) -> Triangulation:
    """Create matploltib triangulation from a SensorGrid object.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid.
        alpha_adjust (float, optional): A value to adjust the alpha value of the triangulation. Defaults to 0.1.

    Returns:
        Triangulation: A matplotlib triangulation.
    """

    alpha = sensorgrid_spacing(sensorgrid=sensorgrid) + alpha_adjust
    plane = sensorgrid_to_plane(sensorgrid=sensorgrid)

    if not any([(plane.n.z == 1), (plane.n.z == -1)]):
        warn(
            "The sensorgrid given is planar, but not in the XY plane. You may need to rotate this when visualising it!"
        )

    x, y = np.array(
        [
            plane.xyz_to_xy(Point3D(*sensor.pos)).to_array()
            for sensor in sensorgrid.sensors
        ]
    ).T

    return create_triangulation(x, y, alpha=alpha)


def validate_triangulation_values(triangulation: Triangulation, values: List[float]):
    """Ensure that the triangulation and values are the same length

    Args:
        triangulation (Triangulation): Triangulation object.
        values (List[float]): Set of values to be plotted.

    Raises:
        ValueError: Error if the triangulation and values are not the same length.
    """
    if len(triangulation.x) != len(values):
        raise ValueError(
            "The shape of the triangulations and values given do not match."
        )


def rotate_triangulation(triangulation: Triangulation, angle: float) -> Triangulation:
    """Rotate a matplotlib triangulation about the xy plane by a given angle.

    Args:
        triangulation (Triangulation):
            The triangulation to rotate.
        angle (float): The rotation angle in degrees.
            Positive values rotate counter-clockwise.

    Returns:
        Triangulation: The rotated triangulation.
    """

    x, y, z = triangulation.x, triangulation.y, 0
    plane = Plane(o=Point3D(np.mean(x), np.mean(y), np.mean(z)))
    angle = np.radians(angle)

    u, v, w = plane.n.x, plane.n.y, plane.n.z
    r2 = u**2 + v**2 + w**2
    r = np.sqrt(r2)
    ct = np.cos(angle)
    st = np.sin(angle) / r
    dt = (u * x + v * y + w * z) * (1 - ct) / r2
    qx = u * dt + x * ct + (-w * y + v * z) * st
    qy = v * dt + y * ct + (w * x - u * z) * st
    qz = w * dt + z * ct + (-v * x + u * y) * st

    return Triangulation(
        qx, qy, triangles=triangulation.triangles, mask=triangulation.mask
    )


def reflect_triangulation(triangulation: Triangulation, plane: Plane) -> Triangulation:
    """Reflect a matplotlib triangulation about the given plane.

    Args:
        triangulation (Triangulation): The triangulation to reflect.

    Returns:
        Triangulation: The reflected triangulation.
    """

    x, y, z = triangulation.x, triangulation.y, 0
    d = 2 * (x * plane.n.x + y * plane.n.y + z * plane.n.z)

    qx = x - d * plane.n.x
    qy = y - d * plane.n.y

    return Triangulation(
        qx, qy, triangles=triangulation.triangles, mask=triangulation.mask
    )


def flip_triangulation_x(triangulation: Triangulation) -> Triangulation:
    """Helper function to flip a triangulation about the x axis.

    Args:
        triangulation (Triangulation): The triangulation to flip.

    Returns:
        Triangulation: The flipped triangulation.
    """
    return _reflect_triangulation(
        triangulation, Plane(o=Point3D(0, 0, 0), n=Vector3D(1, 0, 0))
    )


def flip_triangulation_y(triangulation: Triangulation) -> Triangulation:
    """Helper function to flip a triangulation about the y axis.

    Args:
        triangulation (Triangulation): The triangulation to flip.

    Returns:
        Triangulation: The flipped triangulation.
    """
    return _reflect_triangulation(
        triangulation, Plane(o=Point3D(0, 0, 0), n=Vector3D(0, 1, 0))
    )


def move_triangulation(triangulation: Triangulation, vector: Vector3D) -> Triangulation:
    """Move a matplotlib triangulation by a given x and y value.

    Args:
        triangulation (Triangulation): The triangulation to move.
        vector (Vector3D): The vector to move the triangulation by.

    Returns:
        Triangulation: The moved triangulation.
    """

    return Triangulation(
        triangulation.x + vector.x,
        triangulation.y + vector.y,
        triangles=triangulation.triangles,
        mask=triangulation.mask,
    )


def scale_triangulation(
    triangulation: Triangulation, factor: float, origin: Point3D = Point3D()
) -> Triangulation:
    """Scale a matplotlib triangulation by a given factor.

    Args:
        triangulation (Triangulation): The triangulation to scale.
        factor (float): The factor to scale the triangulation by.
        origin (Point3D, optional): The origin point to scale the triangulation from. Defaults to Point3D().

    Returns:
        Triangulation: The scaled triangulation.
    """

    xy = np.array([triangulation.x, triangulation.y]).T
    o = np.array([origin.x, origin.y])

    qx, qy = ((factor * (xy - o)) + o).T

    return Triangulation(
        qx,
        qy,
        triangles=triangulation.triangles,
        mask=triangulation.mask,
    )


# LADYBUG GEOMETRY METHODS - SHOULD PROBABLY BE PUT INTO GEOMETRY HANDLING MODULE #


def mesh3d_isplanar(mesh: Mesh3D) -> bool:
    """Check if a mesh is planar.

    Args:
        mesh (Mesh3D): A ladybug-geometry Mesh3D.

    Returns:
        bool: True if the mesh is planar, False otherwise.
    """

    return len(set(mesh.vertex_normals)) == 1


def mesh3d_get_plane(mesh: Mesh3D) -> Plane:
    """Estimate the plane of a mesh.

    Args:
        mesh (Mesh3D): A ladybug-geometry Mesh3D.

    Returns:
        Plane: The estimated plane of the mesh.
    """

    if not mesh3d_isplanar(mesh=mesh):
        warn(
            "The mesh given is not planar. This method will return a planar mesh based on a selection of 3-points from the first 3-faces of this mesh."
        )

    plane = Plane.from_three_points(
        *[mesh.vertices[j] for j in [i[0] for i in mesh.faces[:3]]]
    )

    if plane.n.z < 0:
        warn(
            "The plane normal is pointing downwards. This method will return a plane with a normal pointing upwards."
        )
        return plane.flip()

    return plane


def sensorgrid_to_patchcollection(
    sensorgrid: SensorGrid, **kwargs
) -> mcollections.PatchCollection:
    """Convert a honeybee-radiance SensorGrid to a matplotlib PatchCollection.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid containing a mesh.

    Returns:
        mcollections.PatchCollection: The matplotlib PatchCollection.
    """

    if sensorgrid.mesh is None:
        raise ValueError(
            "sensorgrid must have a mesh. This would have been assigned when the sensorgrid was created."
        )

    # flatten the mesh to 2D in XY plane, to raise warnings if necessary
    plane = mesh3d_get_plane(mesh=sensorgrid.mesh)

    patches: List[mpatches.Patch] = []
    for face in sensorgrid.mesh.face_vertices:
        patches.append(
            mpatches.Polygon(np.array([i.to_array()[:2] for i in face]), closed=False)
        )

    return mcollections.PatchCollection(patches, **kwargs)


def sensorgrid_plot_values(
    sensorgrid: SensorGrid, values: List[float], ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """Plot a sensorgrid with values.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid.
        values (List[float]): A list of values to plot.

    Returns:
        mcollections.PatchCollection: The matplotlib PatchCollection.
    """

    pc = sensorgrid_to_patchcollection(sensorgrid, **kwargs)
    pc.set_array(values)
    return ax.add_collection(pc)
