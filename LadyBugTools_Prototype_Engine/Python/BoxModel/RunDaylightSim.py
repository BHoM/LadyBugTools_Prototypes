from dataclasses import dataclass, field
from inspect import getcallargs
from pathlib import Path
from honeybee.model import Model
from ladybug.wea import Wea
from .geometry import geom
from .simulation.daylight_simulation import DaylightSimulation, AnnualGlare, DaylightFactor
from pathlib import Path
from .file_utils import make_folder_if_not_exist
import os
from .results.daylight_results import DaylightSimResults, GlareSimResults, DaylightFactorResults
from .results.daylight_plotting import DaylightPlot, generate_zip

#new file for all daylight sim functionality, no streamlit

def RunDaylightSim(model: Model, wea: Wea, path:str):
    # Run daylight simulation
    daylight_sim = DaylightSimulation(model=model, wea=wea)
    daylight_sim.run_annual_daylight_simulation(path)

    results_folder = make_folder_if_not_exist(os.path.join(path,"annual_daylight"),'metrics')

    daylight_results = DaylightSimResults(hb_model=model, results_folder=results_folder)
    daylight_results.load_and_add_results()
    daylight_results.set_display_modes()
    annual_metrics = daylight_results.annual_metrics

    vtk_path = Path(daylight_results.model.to_vtkjs(folder=path, name=model.identifier))
    vtkjs_name = f'{model.identifier}_vtkjs'
    
    #generating plot images
    image_paths=[]
    output_image_folder = os.path.join(path, 'annual_daylight\\results')

    for i in range(len(annual_metrics)):
        metric = annual_metrics[i]
        grids = [model.properties.radiance.sensor_grids]

        plot = DaylightPlot(metric, grids, 0, 100)
        p,fig = plot.generate_fig()
        image_filepath = plot.save_fig(output_image_folder)

    return daylight_results.model, vtk_path, image_filepath

def RunAnnualGlareAutonomy(model: Model, wea: Wea, path: str):
    # Run annual glare simulation
    glare_sim = AnnualGlare(model=model, wea=wea)
    glare_sim.run_annual_glare_simulation(path)
    results_folder = make_folder_if_not_exist(os.path.join(path,"imageless_annual_glare"),'metrics')

    glare_results = GlareSimResults(hb_model=model, results_folder=results_folder)
    glare_results.load_and_add_results()
    glare_results.set_display_modes()
    annual_metrics = glare_results.annual_metrics

    vtk_path = Path(glare_results.model.to_vtkjs(folder=path, name=model.identifier))
    vtkjs_name = f'{model.identifier}_vtkjs'
    
    #generating plot images
    image_paths=[]
    output_image_folder = os.path.join(path, 'imageless_annual_glare\\results')

    for i in range(len(annual_metrics)):
        metric = annual_metrics[i]
        grids = [model.properties.radiance.sensor_grids[0]]

        plot = DaylightPlot(metric, grids, 20, 100)
        p,fig = plot.generate_fig()
        image_filepath = plot.save_fig(output_image_folder)

    return glare_results.model, vtk_path, image_filepath

def RunDaylightFactor(model: Model, path: str):  
    # Run daylight factor
    daylight_factor = DaylightFactor(model=model)
    daylight_factor.run_daylight_factor(path)
    results_folder = make_folder_if_not_exist(os.path.join(path,"daylight_factor"),'results')

    daylightfactor_results = DaylightFactorResults(hb_model=model, results_folder=results_folder)
    daylightfactor_results.load_and_add_results()
    daylightfactor_results.set_display_modes()
    annual_metrics = daylightfactor_results.annual_metrics

    vtk_path = Path(daylightfactor_results.model.to_vtkjs(folder=results_folder, name=model.identifier))
    vtkjs_name = f'{model.identifier}_vtkjs'
    
    #generating plot images
    image_paths=[]
    output_image_folder = os.path.join(path, 'daylight_factor\\results')

    for i in range(len(annual_metrics)):
        metric = annual_metrics[i]
        grids = [model.properties.radiance.sensor_grids[0]]

        plot = DaylightPlot(metric, grids, 0, 10)
        p,fig = plot.generate_fig()
        image_filepath = plot.save_fig(output_image_folder)

    return daylightfactor_results.model, vtk_path, image_filepath