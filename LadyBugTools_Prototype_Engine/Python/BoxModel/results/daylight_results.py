
from dataclasses import dataclass, field
from pathlib import Path
from honeybee.model import Model
from honeybee_vtk.legend_parameter import ColorSets
from honeybee_vtk.model import Model as VTKModel, SensorGridOptions, DisplayMode


@dataclass
class DaylightSimResults:
    hb_model: Model= field(init=True)
    results_folder: str = field(init=True)

    def __post_init__(self):
        self.model=VTKModel(hb_model=self.hb_model, grid_options=SensorGridOptions.Mesh)
        self.annual_metrics = [
                    {'folder': 'da', 'extension': 'da', 'name': 'Daylight Autonomy', 'colors': ColorSets.nuanced, 'color_index':1,'shortened': 'DA'},
                    {'folder': 'cda', 'extension': 'cda', 'name': 'Continuous Daylight Autonomy', 'colors': ColorSets.nuanced,'color_index':1, 'shortened': 'cDa'},
                    {'folder': 'udi', 'extension': 'udi', 'name': 'Useful Daylight Illuminance', 'colors': ColorSets.annual_comfort,'color_index':7, 'shortened': 'UDIa'},
                    {'folder': 'udi_lower', 'extension': 'udi', 'name': 'Lower Daylight Illuminance', 'colors': ColorSets.cold_sensation,'color_index':11, 'shortened': 'UDIs'},
                    {'folder': 'udi_upper', 'extension': 'udi', 'name': 'Excessive Daylight Illuminance', 'colors': ColorSets.shade_harm,'color_index':16, 'shortened': 'UDIe'}
                ]

    def load_and_add_results(self):
        for metric in self.annual_metrics:
            results = []
            for grid in self.model.sensor_grids.data:
                res_file = Path(
                    self.results_folder, metric['folder'], f'{grid.identifier}.{metric["extension"]}'
                )
                grid_res = [float(v) for v in res_file.read_text().splitlines()]
                metric['results']=grid_res
                results.append(grid_res)

             #Add results to sensor grids
            self.model.sensor_grids.add_data_fields(results, name=metric['name'], per_face=True,data_range=[0,100], colors=metric['colors'])
            self.model.sensor_grids.color_by= 'Daylight Autonomy'
    
    def set_display_modes(self):
        self.model.sensor_grids.display_mode = DisplayMode.SurfaceWithEdges
        self.model.update_display_mode(DisplayMode.Wireframe)
        self.model.shades.display_mode = DisplayMode.SurfaceWithEdges
        # Not working as expected due to underlying pollination code- for future investigation

@dataclass
class GlareSimResults:
    hb_model: Model= field(init=True)
    results_folder: str = field(init=True)

    def __post_init__(self):
        self.model=VTKModel(hb_model=self.hb_model, grid_options=SensorGridOptions.Mesh)
        self.annual_metrics = [
                    {'folder': 'ga', 'extension': 'ga', 'name': 'Annual Glare Autonomy', 'colors': ColorSets.nuanced, 'color_index':1,'shortened': 'GA'}                    
                ]

    def load_and_add_results(self):
        for metric in self.annual_metrics:
            results = []
            for grid in self.model.sensor_grids.data:
                res_file = Path(
                    self.results_folder, metric['folder'], f'{grid.identifier}.{metric["extension"]}'
                )
                grid_res = [float(v) for v in res_file.read_text().splitlines()]
                metric['results']=grid_res
                results.append(grid_res)

             #Add results to sensor grids
            self.model.sensor_grids.add_data_fields(results, name=metric['name'], per_face=True,data_range=[0,100], colors=metric['colors'])
            self.model.sensor_grids.color_by= 'Annual Glare Autonomy'
    
    def set_display_modes(self):
        self.model.sensor_grids.display_mode = DisplayMode.SurfaceWithEdges
        self.model.update_display_mode(DisplayMode.Wireframe)
        self.model.shades.display_mode = DisplayMode.SurfaceWithEdges
        # Not working as expected due to underlying pollination code- for future investigation

@dataclass
class DaylightFactorResults:
    hb_model: Model= field(init=True)
    results_folder: str = field(init=True)

    def __post_init__(self):
        self.model=VTKModel(hb_model=self.hb_model, grid_options=SensorGridOptions.Mesh)
        self.annual_metrics = [
                    {'folder': 'results', 'extension': 'res', 'name': 'Daylight Factor', 'colors': ColorSets.nuanced, 'color_index':1,'shortened': 'DF'}                    
                ]

    def load_and_add_results(self):
        for metric in self.annual_metrics:
            results = []
            for grid in self.model.sensor_grids.data:
                res_file = Path(
                    self.results_folder, f'{grid.identifier}.{metric["extension"]}'
                )
                grid_res = [float(v) for v in res_file.read_text().splitlines()]
                metric['results']=grid_res
                results.append(grid_res)

             #Add results to sensor grids
            self.model.sensor_grids.add_data_fields(results, name=metric['name'], per_face=True,data_range=[0,100], colors=metric['colors'])
            self.model.sensor_grids.color_by= 'Daylight Factor'
    
    def set_display_modes(self):
        self.model.sensor_grids.display_mode = DisplayMode.SurfaceWithEdges
        self.model.update_display_mode(DisplayMode.Wireframe)
        self.model.shades.display_mode = DisplayMode.SurfaceWithEdges

