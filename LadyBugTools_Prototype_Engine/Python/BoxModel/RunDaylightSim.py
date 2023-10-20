from dataclasses import dataclass, field
from pathlib import Path

from honeybee.model import Model

from ladybug.wea import Wea

from ladybugtools_toolkit.prototypes.BoxModel.geometry import geom

from ladybugtools_toolkit.prototypes.BoxModel.simulation.daylight_simulation import DaylightSimulation

from pathlib import Path

from ladybugtools_toolkit.prototypes.BoxModel.file_utils import make_folder_if_not_exist

import os

 

from ladybugtools_toolkit.prototypes.BoxModel.results.daylight_results import DaylightSimResults

from results.daylight_plotting import DaylightPlot, generate_zip

 

 

#new file for all daylight sim functionality, no streamlit

 

#set up file

dirname = os.path.dirname(__file__)

path = make_folder_if_not_exist(dirname,'temp')

 

@dataclass

class RunDaylightSim:

    model: Model

    wea: Wea

    grid_size: float

 

    #output data

    def output(vtk_path, image_filepath):

        return vtk_path.read_bytes(), image_filepath

 

    # Generate sensor grid

    sensor_grid = geom.BoxModelSensorGrid(model= model, grid_size=grid_size).sensor_grid

    model.properties.radiance.add_sensor_grid(sensor_grid)

       

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

        grids = [model.properties.radiance.sensor_grids[0]]

 

        plot= DaylightPlot(metric, grids)

        p,fig= plot.generate_fig()

        image_filepath= plot.save_fig(output_image_folder)

 

    output(vtk_path,image_filepath)