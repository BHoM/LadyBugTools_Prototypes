from typing import List, Tuple
from warnings import warn

import matplotlib.path as mpath
import numpy as np
from honeybee_radiance.sensorgrid import SensorGrid
from ladybug_geometry.geometry3d.plane import Plane, Point3D, Vector3D
from ladybugtools_toolkit.plot.utilities import create_triangulation
from matplotlib.patches import PathPatch
from matplotlib.tri.triangulation import Triangulation
from scipy.spatial.distance import cdist
from ladybug_geometry.bounding import bounding_rectangle
from pathlib import Path
from honeybee_radiance_folder.folder import ModelFolder


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
    """Create a list of sensorgrids from a results folder.

    Args:
        model_simulation_folder (Path): A directory containing radiance simulation results from a lbt-recipe.

    Returns:
        Tuple[SensorGrid]: A tuple of honeybee-radiance SensorGrids.
    """
    if not isinstance(model_simulation_folder, Path):
        model_simulation_folder = Path(model_simulation_folder)

    pts_files = find_pts_files(model_simulation_folder)
    sensor_grids = []
    for pts_file in pts_files:
        sensor_grids.append(SensorGrid.from_file(pts_file))

    return tuple(sensor_grids)


def get_sensorgrid_by_name(model_simulation_folder: Path, name: str) -> SensorGrid:
    """Find and return a sensorgrid by name.

    Args:
        model_simulation_folder (Path): The folder containing the sensorgrid.
        name (str): The name of the sensorgrid.

    Returns:
        SensorGrid: A honeybee-radiance SensorGrid.
    """
    if not isinstance(model_simulation_folder, Path):
        model_simulation_folder = Path(model_simulation_folder)

    pts_files = find_pts_files(model_simulation_folder)
    for pts_file in pts_files:
        if name != pts_file.stem:
            continue
        return SensorGrid.from_file(pts_file)

    raise ValueError(f"Sensorgrid with name {name} not found in {pts_files[0].parent}.")


def get_triangulations(model_simulation_folder: Path) -> Tuple[Triangulation]:
    """Create a list of triangulations from a results folder.

    Args:
        model_simulation_folder (Path): A directory contianing radiance simulation results from a lbt-recipe.

    Returns:
        Tuple[Triangulation]: A tuple of matplotlib triangulations.
    """

    grids = get_sensorgrids(model_simulation_folder=model_simulation_folder)
    return tuple([sensorgrid_to_triangulation(grid) for grid in grids])


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
