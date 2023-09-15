from pathlib import Path
from typing import Tuple
from honeybee_radiance.view import View
from honeybee_radiance_folder.folder import ModelFolder


def get_views(results_folder: Path) -> Tuple[View]:
    """Create a list of views from a results folder.

    Args:
        results_folder (Path): A directory contianing radiance simulation results from a lbt-recipe.

    Returns:
        Tuple[SensorGrid]: A tuple of honeybee-radiance Views.
    """

    if not isinstance(results_folder, Path):
        results_folder = Path(results_folder)

    res_dirs = ["point_in_time_view"]
    for res_dir in res_dirs:
        if (results_folder / res_dir).exists():
            return tuple(
                [
                    View.from_file(results_folder / res_dir / i)
                    for i in ModelFolder(
                        project_folder=results_folder / res_dir
                    ).view_files()
                ]
            )

    raise ValueError("No view files found in the given folder.")
