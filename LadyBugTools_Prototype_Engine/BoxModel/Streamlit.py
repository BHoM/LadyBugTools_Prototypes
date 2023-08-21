import json
import os
import sys
from pathlib import Path
from PIL import Image
import streamlit as st

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib as mpl

from honeybee.facetype import Floor, Wall
from honeybee.model import Model
from honeybee_energy.programtype import ProgramType
from honeybee_energy.lib.programtypes import program_type_by_identifier
from honeybee_energy.properties.model import ModelEnergyProperties
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.simulation.runperiod import RunPeriod
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_vtk.model import Model as VTKModel, SensorGridOptions, DisplayMode
from honeybee_vtk.legend_parameter import ColorSets
from ladybug.epw import EPW
from ladybug.wea import Wea
from ladybug.color import Color, ColorRange, Colorset
from ladybug.datatype.energy import Energy
from lbt_recipes.recipe import Recipe
from lbt_recipes.settings import RecipeSettings
from pollination_streamlit_viewer import viewer

from file_utils import make_folder_if_not_exist
from geometry.geom import BoxModelGlazing, BoxModelModel, BoxModelRoom
from construction.construction_set import BoxModelFabricProperties
from simulation.energy_sim_setup import SimulationOutputSetup, SimulationParameterSetup
from simulation.energy_sim_run import RunEnergySimulation
from results.energy_results import EnergySimResults
from results.energy_plotting import display_metrics_as_df, LoadBalanceBarPlot
from results.daylight_plotting import build_custom_continuous_cmap, vertices_from_grids, add_starting_vertices_to_end, vertices_to_patches, flatten, generate_zip 
from program.program import PeopleLoad, LightingLoad, ElectricEquipmentLoad, InfiltrationLoad, SetpointProgram

dirname = os.path.dirname(__file__)
path = make_folder_if_not_exist(dirname,'temp')

st.set_page_config(page_title='Box Model', layout='wide')
st.header('Box Model App')
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
        glazing_properties= BoxModelGlazing(glazing_ratio=glazing_ratio,
                                            sill_height=sill_height,
                                            window_height=window_height,
                                            bay_width=bay_width)
        room= BoxModelRoom(bay_width=bay_width,
                           bay_count=bay_count,
                           depth= room_depth,
                           height= room_height,
                           glazing_properties= glazing_properties).get_honeybee_room()
        
        model=BoxModelModel(room).generate_honeybee_model()
        vtk_path = Path(VTKModel(model).to_vtkjs(folder=path, name=model.identifier))

        st.session_state.model = model
        st.session_state.room = room

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
        room=st.session_state.room
        model= st.session_state.model
        epw_file= st.session_state.epw_file

        simulation_folder= make_folder_if_not_exist(path, 'simulation')
        #Apply construction set from construction_set.py
        bm_construct_set = BoxModelFabricProperties(epw = epw_obj).construction_set
        room.properties.energy.construction_set = bm_construct_set

        # Apply program type from program.py
        bm_program_type = ProgramType(identifier='BM_program_type',
                                people = PeopleLoad().load,
                                lighting= LightingLoad().load,
                                electric_equipment= ElectricEquipmentLoad().load,
                                infiltration= InfiltrationLoad(hb_room=room, ach = 0.1).load,
                                setpoint= SetpointProgram().setpoint,
                                ventilation= None
                                )
        room.properties.energy.program_type = bm_program_type

        #ideal air system
        ideal_air= IdealAirSystem(identifier='Idealair', economizer_type='NoEconomizer') #HVAC system params
        room.properties.energy.hvac = ideal_air #setting the HVAC system to defined params
        
        #set up simulation output+ params
        sim_output=SimulationOutputSetup().return_sim_output()
        sim_setup= SimulationParameterSetup(sim_output=sim_output, room=room,
                                            model=model, simulation_folder=simulation_folder, epw_obj=epw_obj)        
        sim_setup.add_autocalculated_design_days()
        hbjson_path= sim_setup.model_to_json()
        sim_par_path= sim_setup.sim_par_to_json()

        #run energy simulation 
        energy_sim=RunEnergySimulation(simulation_folder= simulation_folder, hbjson_path=hbjson_path,
                                     sim_par_path= sim_par_path, epw_file= epw_file)
        
        sql_path= energy_sim.run_simulation()
        load_balance= energy_sim.calculate_load_balance(model, sql_path=sql_path)
        
        #processing results
        energy_sim_processing= EnergySimResults(load_balance=load_balance)
        metrics= energy_sim_processing.metric_dictionary()
        monthly_balance= energy_sim_processing.monthy_balance()

        st.session_state.metrics=metrics
        st.session_state.monthly_balance= monthly_balance
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

        run_settings = RecipeSettings(folder = path, reload_old=False)

        project_folder = recipe.run(run_settings, radiance_check=True, silent=True)
     
        results_folder= make_folder_if_not_exist(os.path.join(path,"annual_daylight"),'metrics')
        
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
                res_file = Path(
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


if 'monthly_balance' in st.session_state:
    monthly_balance= st.session_state.monthly_balance
    metrics= st.session_state.metrics
    st.header("Energy Results")

    col1, col2= st.columns([0.4,0.6])

    with col1:
        #Displaying metrics
        metrics_df = display_metrics_as_df(metrics)
        st.dataframe(metrics_df, width=500)

    with col2:
        fig=LoadBalanceBarPlot(monthly_balance=monthly_balance).save_fig()
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
        mesh_vertices = vertices_from_grids(grids)

        patch_vertices = []
        for grid in mesh_vertices:
            repeated_vertices = add_starting_vertices_to_end(grid)
            patch_vertices.append(repeated_vertices)

        patches_per_grid = vertices_to_patches(patch_vertices)
        patches = flatten(patches_per_grid)

        #colormap
        color_set=Colorset()._colors
        index= metric['color_index']
        rgb=color_set[index]
        cmap= build_custom_continuous_cmap(rgb) #borrowed this from someone

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
    zip_data = generate_zip(image_paths, zip_filename)
    # Provide a download button for the zip file
    st.download_button(label="Download Daylight Metrics", data=zip_data, file_name=zip_filename, mime="application/zip")

def main():
    pass

if __name__ == "__main__":
    main()