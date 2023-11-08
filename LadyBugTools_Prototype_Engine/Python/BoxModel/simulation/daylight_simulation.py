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
        #self.recipe.input_value_by_name('visible-transmittance', 0)

    def run_annual_daylight_simulation(self, path):
        run_settings = RecipeSettings(folder = path, reload_old=False)
        project_folder = self.recipe.run(run_settings, radiance_check=True, silent=True)
      

@dataclass
class AnnualGlare:
    model: Model
    wea: Wea
    def __post_init__(self):
        self.recipe = Recipe('imageless-annual-glare') 
        self.recipe.input_value_by_name('model', self.model)
        self.recipe.input_value_by_name('wea', self.wea)
        self.recipe.input_value_by_name('north', 0)
        self.recipe.input_value_by_name('glare-threshold', None)
        self.recipe.input_value_by_name('luminance-factor', None)
        self.recipe.input_value_by_name('schedule', None)
        self.recipe.input_value_by_name('grid-filter', None)
        self.recipe.input_value_by_name('radiance-parameters', None)
        
    def run_annual_glare_simulation(self, path):
        run_settings = RecipeSettings(folder = path, reload_old=False)
        project_folder = self.recipe.run(run_settings, radiance_check=True, silent=True)

@dataclass
class DaylightFactor:
    model: Model
    def __post_init__(self):
        self.recipe = Recipe('daylight-factor')
        self.recipe.input_value_by_name('model', self.model)
        self.recipe.input_value_by_name('grid-filter', None)
        self.recipe.input_value_by_name('radiance-parameters', None)
    
    def run_daylight_factor(self, path):
        run_settings = RecipeSettings(folder = path, reload_old=False)
        project_folder = self.recipe.run(run_settings, radiance_check=True, silent=True)
