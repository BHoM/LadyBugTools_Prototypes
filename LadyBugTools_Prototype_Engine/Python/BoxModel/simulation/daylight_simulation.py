from honeybee.model import Model
from ladybug.wea import Wea
from lbt_recipes.recipe import Recipe
from lbt_recipes.settings import RecipeSettings
from dataclasses import dataclass


@dataclass
class DaylightSimulation:
    model: Model
    wea: Wea
        
    def __post_init__(self):
        self.recipe = Recipe('annual-daylight')
        self.recipe.input_value_by_name('model', self.model)
        self.recipe.input_value_by_name('wea', self.wea)
        self.recipe.input_value_by_name('north', 0)
        self.recipe.input_value_by_name('thresholds', None)
        self.recipe.input_value_by_name('schedule', None)
        self.recipe.input_value_by_name('grid-filter', None)
        self.recipe.input_value_by_name('radiance-parameters', None)

    def run_annual_daylight_simulation(self, path):
        run_settings = RecipeSettings(folder = path, reload_old=False)
        project_folder = self.recipe.run(run_settings, radiance_check=True, silent=True)
