import json
import os
import os.path
import pathlib
from pathlib import Path
import sys
from PIL import Image
import zipfile
import io
import streamlit as st

from typing import Dict
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib as mpl
import numpy as np
import pandas as pd

import program as pg
import construction_set as cs
import daylight_plotting as DP
from geometry import create_room, add_glazing


from honeybee.boundarycondition import Outdoors
from honeybee.face import Face
from honeybee.facetype import Floor, Wall
from honeybee.model import Model
from honeybee.room import Room
from honeybee_energy.boundarycondition import Adiabatic
from honeybee_energy.constructionset import ConstructionSet, OpaqueConstruction
from honeybee_energy.programtype import ProgramType
from honeybee_energy.lib.constructionsets import construction_set_by_identifier
from honeybee_energy.lib.programtypes import program_type_by_identifier
from honeybee_energy.properties.model import ModelEnergyProperties
from honeybee_energy.simulation.output import SimulationOutput
from honeybee_energy.simulation.parameter import SimulationParameter
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.run import to_openstudio_osw, run_osw, run_idf
from honeybee_energy.result.osw import OSW
from honeybee_energy.result.loadbalance import LoadBalance
from honeybee_energy.result.eui import eui_from_sql
from honeybee_energy.simulation.runperiod import RunPeriod
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_vtk.model import Model as VTKModel, SensorGridOptions, DisplayMode
from honeybee_vtk.legend_parameter import ColorSets
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.epw import EPW
from ladybug.wea import Wea
from ladybug.color import Color, ColorRange, Colorset
from ladybug.datatype.energy import Energy
from ladybug.dt import Date
from lbt_recipes.recipe import Recipe
from lbt_recipes.settings import RecipeSettings
from pollination_streamlit_viewer import viewer


dirname = os.path.dirname(__file__)
path = os.path.join(dirname, 'temp')
os.makedirs(path, exist_ok=True)

st.set_page_config(page_title='Box Model', layout='wide')

st.header('Box Model App')
st.text(path)

st.subheader('Weather File')
uploaded_epw = st.file_uploader(label="Drag and drop .epw file here",
                    type = ".epw",
                    accept_multiple_files=False)

if uploaded_epw:
    epw_file = Path(f'./data/{uploaded_epw.name}')  # This creates a Path object
    epw_file.parent.mkdir(parents=True, exist_ok=True)
    epw_file.write_bytes(uploaded_epw.read())  # This saves the EPW file to the specified path
    epw_obj = EPW(epw_file)
    st.session_state.epw_file = str(epw_file.resolve())  # Save the resolved absolute path as a string
    st.session_state.wea = Wea.from_epw_file(epw_file, timestep=1)
    st.write("EPW successfully uploaded")

with st.sidebar.form('box-model-geometry'):
    st.subheader('Geometry Definition')
    bay_width = st.slider('Bay Width', min_value=2.0, max_value=10.0, value=3.0, step=0.1)
    bay_count = st.slider('Bay Count', min_value=1, max_value=10, value=3, step=1) 
    room_depth = st.slider('Room Depth', min_value=3.0, max_value=30.0, value=10.0, step=0.5)
    room_height = st.slider('Room Height', min_value=2.0, max_value=5.0, value=2.5, step=0.1)
    glazing_ratio = st.slider('Glazing Ratio', min_value=0.1, max_value=0.9, value=0.4, step=0.05)
    window_height = st.slider('Window Height (glazing ratio takes precedence)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    sill_height = st.slider('Sill Height (window height takes precedence)', min_value=0.0, max_value=2.0, value=0.5, step=0.1)

    geometry_submit_button = st.form_submit_button(
        label='Generate Geometry')

    if geometry_submit_button:
        room_width = bay_width * bay_count
        
        room = Room.from_box(identifier='Test',
                      width= room_width,
                      depth = room_depth,
                      height= room_height)
        
        # Set all faces to adiabatic by default
        for face in room.faces:
            face.boundary_condition = Adiabatic()

        # Assign first wall (north) with outdoor boundary condition and glazing
        room.faces[1].boundary_condition = Outdoors()
        room.faces[1].apertures_by_ratio_rectangle(
            ratio = glazing_ratio,
            aperture_height = window_height,
            sill_height = sill_height,
            horizontal_separation = bay_width)
        
        model = Model(identifier='Test',
                      rooms = [room])
        
        st.session_state.model = model
        st.session_state.room = room
        vtk_path = Path(VTKModel(model).to_vtkjs(folder=path, name=model.identifier))

        vtkjs_name = f'{model.identifier}_vtkjs'
        st.session_state.content = vtk_path.read_bytes()
        st.session_state[vtkjs_name] = vtk_path

if 'content' in st.session_state:
    viewer(
        content=st.session_state.content,
        key='vtkjs-viewer',
        subscribe=False,
        style={
            'height' : '500px'
        }
    )

#Energy simulation code
with st.sidebar.form('box-model-energy'):
    st.subheader('Energy Simulation')
    energy_submit_button = st.form_submit_button(label='Simulate Load Balance')
    st.caption('Requires geometry to be generated and EPW to be uploaded')

    if energy_submit_button:
        room=st.session_state.room #adding the model from the current session
        epw_file= st.session_state.epw_file
        simulation_folder= os.path.join(path, 'simulation')
        os.makedirs(simulation_folder, exist_ok=True)

        #Apply construction set from construction_set.py
        bm_construct_set = cs.BoxModelFabricProperties(epw = epw_obj).construction_set
        room.properties.energy.construction_set = bm_construct_set

        # Apply program type from program.py
        bm_program_type = ProgramType(identifier='BM_program_type',
                                people = pg.PeopleLoad().load,
                                lighting= pg.LightingLoad().load,
                                electric_equipment= pg.ElectricEquipmentLoad().load,
                                infiltration= pg.InfiltrationLoad(hb_room=room, ach = 0.1).load,
                                setpoint= pg.SetpointProgram().setpoint,
                                ventilation= None
                                )
        room.properties.energy.program_type = bm_program_type

        #ideal air system
        ideal_air= IdealAirSystem(identifier='Idealair', economizer_type='NoEconomizer') #HVAC system params
        room.properties.energy.hvac = ideal_air #setting the HVAC system to defined params

        model = Model(identifier='Test',
                      rooms = [room])
        
        #Sim parameters
        sim_output= SimulationOutput()
        sim_output.add_energy_balance_variables(load_type='Sensible')
        sim_output.add_comfort_metrics()
        sim_output.add_hvac_energy_use()
        sim_output.add_surface_temperature()
        sim_output.add_glazing_solar()
        sim_par = SimulationParameter(output=sim_output) #creates a simulation parameter object with EnergyPlus simulation settings#

        # add autocalculated design days
        des_days = [epw_obj.approximate_design_day('WinterDesignDay'),
                    epw_obj.approximate_design_day('SummerDesignDay')]
        sim_par.sizing_parameter.design_days = des_days

        hbjson_path = model.to_hbjson(folder=simulation_folder)
        sim_par_json = sim_par.to_dict()
        sim_par_path = os.path.join(simulation_folder, 'sim_par.json')
        json.dump(sim_par_json, open(sim_par_path, 'w'))
       
        #running the simulation
        osw= to_openstudio_osw(osw_directory=simulation_folder, model_path=hbjson_path,sim_par_json_path=sim_par_path, epw_file = epw_file) 
        osw, idf =run_osw(osw)
        sql, zsz, rdd, html, err = run_idf(idf_file_path = idf, epw_file_path = epw_file)

        #load balance
        sql_path = os.path.join(simulation_folder, 'run\eplusout.sql')
        load_balance = LoadBalance.from_sql_file(model=model, sql_path=sql_path)
        norm_bal_stor = load_balance.load_balance_terms(True,True)
        #term_names = [term.header.metadata['type'] for term in norm_bal_stor] 
        #st.text(term_names)

        #metrics
        outputs = {
        'annual solar gain (kWh/m2)': norm_bal_stor[1].to_unit('kWh/m2').total,
        'peak solar gain (Wh/m2)': norm_bal_stor[1].to_unit('Wh/m2').max,
        'annual heating emand (kWh/m2)': norm_bal_stor[0].to_unit('kWh/m2').total,
        'peak heating demand(Wh/m2)': norm_bal_stor[0].to_unit('Wh/m2').max,
        'annual cooling demand (kWh/m2)': -(norm_bal_stor[8].to_unit('kWh/m2').total),
        'peak cooling demand (Wh/m2)': -(norm_bal_stor[8].to_unit('Wh/m2').min),
        'annual lighting energy (kWh/m2)': norm_bal_stor[3].to_unit('kWh/m2').total,
        'annual external conduction (kWh/m2)': -(norm_bal_stor[6].to_unit('kWh/m2').total + norm_bal_stor[7].to_unit('kWh/m2').total)
        }
        st.session_state.outputs=outputs

        monthly_energy = [term.total_monthly() for term in norm_bal_stor]
        st.session_state.monthly_energy = monthly_energy
        st.success("Successful Energy Simmulation")

            
with st.sidebar.form('box-model-daylight'):
    st.subheader('Daylight Grid')
    #grid_bay_count = st.slider('Bay Count', min_value=1, max_value=5, value=1, step=1)
    grid_size = st.slider('Grid Spacing', min_value=0.1, max_value=1.0, value=0.5, step=0.1) 
    daylight_submit_button = st.form_submit_button(label='Simulate Annual Daylight')
    st.caption('Requires geometry to be generated and EPW to be uploaded')

    if daylight_submit_button:
        model = st.session_state.model
        wea = st.session_state.wea

        room = model.rooms[0]

        for face in room.faces:
            if face.type == Floor():
                floor = face.geometry
        
        identifier='Test'

        sensor_grid = SensorGrid.from_face3d(identifier=identifier,
                                      faces = [floor],
                                      x_dim = grid_size,
                                      offset=0.75,
                                      flip=True)

        model.properties.radiance.add_sensor_grid(sensor_grid)

        recipe = Recipe('annual-daylight')
        recipe.input_value_by_name('model', model)
        recipe.input_value_by_name('wea', wea)
        recipe.input_value_by_name('north', 0)
        recipe.input_value_by_name('thresholds', None)
        recipe.input_value_by_name('schedule', None)
        recipe.input_value_by_name('grid-filter', None)
        recipe.input_value_by_name('radiance-parameters', None)

        run_settings = RecipeSettings(folder = path, reload_old=True)

        project_folder = recipe.run(run_settings, radiance_check=True, silent=True)
     
   
        results_folder= path + '/annual_daylight/metrics'
        os.makedirs(results_folder, exist_ok=True)
        
        hb_model= st.session_state.model
        model=VTKModel(hb_model=hb_model, grid_options=SensorGridOptions.Mesh)
        # load the results for each grid
        # note that we load the results using the order for model to ensure the order will match

        annual_metrics = [
            {'folder': 'da', 'extension': 'da', 'name': 'Daylight Autonomy', 'colors': ColorSets.nuanced, 'color_index':1,'shortened': 'DA'},
            {'folder': 'cda', 'extension': 'cda', 'name': 'Continuous Daylight Autonomy', 'colors': ColorSets.nuanced,'color_index':1, 'shortened': 'cDa'},
            {'folder': 'udi', 'extension': 'udi', 'name': 'Useful Daylight Illuminance', 'colors': ColorSets.annual_comfort,'color_index':7, 'shortened': 'UDIa'},
            {'folder': 'udi_lower', 'extension': 'udi', 'name': 'Lower Daylight Illuminance', 'colors': ColorSets.cold_sensation,'color_index':11, 'shortened': 'UDIs'},
            {'folder': 'udi_upper', 'extension': 'udi', 'name': 'Excessive Daylight Illuminance', 'colors': ColorSets.shade_harm,'color_index':16, 'shortened': 'UDIe'}
        ]

        for metric in annual_metrics:
            results = []
            for grid in model.sensor_grids.data:
                res_file = pathlib.Path(
                    results_folder, metric['folder'], f'{grid.identifier}.{metric["extension"]}'
                )
                grid_res = [float(v) for v in res_file.read_text().splitlines()]
                metric['results']=grid_res
                results.append(grid_res)

            # add the results to sensor grids as a new field
            # per face is set to True since we loaded grids as a mesh
            model.sensor_grids.add_data_fields(results, name=metric['name'], per_face=True,data_range=[0,100], colors=metric['colors'])

            model.sensor_grids.color_by= 'Daylight Autonomy'
        
        # change display mode for sensor grids to be surface with edges
        model.sensor_grids.display_mode = DisplayMode.SurfaceWithEdges
        # update model visualization to wireframe
        model.update_display_mode(DisplayMode.Wireframe)
        # make shades to be shaded with edge
        model.shades.display_mode = DisplayMode.SurfaceWithEdges
        #model.to_html(folder = output_folder,name='Test-With-DaylightResults', show=False)


        vtk_path = Path(model.to_vtkjs(folder=path, name=hb_model.identifier))
        vtkjs_name = f'{hb_model.identifier}_vtkjs'
        st.session_state.content = vtk_path.read_bytes()
        st.session_state[vtkjs_name] = vtk_path

        st.session_state.annual_metrics= annual_metrics

        st.success("Successful Daylight Simmulation")


if 'monthly_energy' in st.session_state:
    monthly_energy = st.session_state.monthly_energy
    outputs= st.session_state.outputs
    st.header("Energy Results")

    col1, col2= st.columns([0.4,0.6])

    with col1:
        #Displaying metrics
        metrics_df = pd.DataFrame.from_dict(outputs,orient='index').round(2)
        metrics_df.columns = [' ']
        st.dataframe(metrics_df, width=500)

    with col2:
        # Dictionary to hold extracted data   
        data = defaultdict(list)
        # Loop through each collection
        for i, monthly_coll in enumerate(monthly_energy):
            load_name = monthly_coll.header.metadata['type']
            monthly_data = monthly_coll.values
            datetimes = monthly_coll.datetimes

            for d, v in zip(datetimes, monthly_data):
                data[d].append(v)
            data['Load'].append(load_name)

        df = pd.DataFrame(data)
        # Melt the DataFrame to have a single 'Month' column and a 'value' column
        df_melted = df.melt(id_vars=['Load'], var_name='Month', value_name='Value')

        #colorset:
        color_set=Colorset()._colors
        cmap= DP.build_custom_continuous_cmap(color_set[19])

        # Create a stacked bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        df_melted.pivot_table(index='Month',columns='Load',values='Value', aggfunc='sum').plot(kind='bar', stacked=True, ax=ax, colormap=cmap, width=0.85, edgecolor= "black")
        ax.set_xlabel('')
        ax.set_ylabel('Energy (kWh)')  
        ax.set_title('Monthly Load Balance')
        ax.tick_params(axis='x', rotation=0)
        ax.set_xticklabels(['Jan','Feb', 'Mar', 'Apr','May','Jun','Jul', 'Aug','Sep','Oct','Nov','Dec'])
        ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
        st.pyplot(fig)

#visualising daylight
if 'annual_metrics' in st.session_state:
    st.header("Daylight Results")
    annual_metrics= st.session_state.annual_metrics
    hb_model= st.session_state.model
    model=VTKModel(hb_model=hb_model, grid_options=SensorGridOptions.Mesh)
    image_paths=[]

    #show legend tickbox
    show_legend = st.checkbox("Show Legend", value=False)

    # Specify the output folder where you want to save the images
    output_image_folder = os.path.join(path, 'annual_daylight//results')
    st.text(output_image_folder)

    col1, col2, col3, col4, col5 = st.columns(5)

    for i in range(len(annual_metrics)):
        metric = annual_metrics[i]
        grids = [hb_model.properties.radiance.sensor_grids[0]]
        mesh_vertices = DP.vertices_from_grids(grids)

        patch_vertices = []
        for grid in mesh_vertices:
            repeated_vertices = DP.add_starting_vertices_to_end(grid)
            patch_vertices.append(repeated_vertices)

        patches_per_grid = DP.vertices_to_patches(patch_vertices)
        patches = DP.flatten(patches_per_grid)

        #colormap
        color_set=Colorset()._colors
        index= metric['color_index']
        rgb=color_set[index]
        cmap= DP.build_custom_continuous_cmap(rgb) #borrowed this from someone

        # Create a PatchCollection
        p = PatchCollection(patches, cmap=cmap, alpha=1)
        p.set_array(metric['results'])

        fig, ax = plt.subplots()
        ax.add_collection(p)
        ax.autoscale(True)
        ax.axis('equal')
        ax.axis('off')
        p.set_clim([0, 100])

        colorbar=fig.colorbar(p)
        if show_legend is True:
            colorbar.ax.set_title(metric['shortened'])
        
        else:
            colorbar.ax.set_title('')  # Remove the colorbar title
            colorbar.remove()

        # Saving graphic
        metric_name = metric['name'].replace(' ', '_')
        image_filepath = os.path.join(output_image_folder, f'{metric_name}.png')
        plt.savefig(image_filepath, bbox_inches='tight', dpi=500)

        # Display the image in the appropriate column
        with open(image_filepath, "rb") as image_file:
            image_data = image_file.read()

        # Place the image in the corresponding column
        if i % 5 == 0:
            col = col1
        elif i % 5 == 1:
            col = col2
        elif i % 5 == 2:
            col = col3
        elif i % 5 == 3:
            col = col4
        else:
            col = col5

        with col:
            st.image(image_data, caption=metric['name'], use_column_width=True)
            image_paths.append(image_filepath)

    zip_filename = 'Daylight_metrics.zip'
    zip_data = DP.generate_zip(image_paths, zip_filename)
    # Provide a download button for the zip file
    st.download_button(label="Download Daylight Metrics", data=zip_data, file_name=zip_filename, mime="application/zip")




def main():
    pass

if __name__ == "__main__":
    main()