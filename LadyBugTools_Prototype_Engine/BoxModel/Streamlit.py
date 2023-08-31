import os
from pathlib import Path
import streamlit as st
from honeybee_energy.programtype import ProgramType
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_vtk.model import Model as VTKModel
from ladybug.epw import EPW
from ladybug.wea import Wea
from pollination_streamlit_viewer import viewer

from file_utils import make_folder_if_not_exist
from geometry.geom import BoxModelGlazing, BoxModelModel, BoxModelRoom, BoxModelSensorGrid
from construction.construction_set import BoxModelFabricProperties
from program.program import PeopleLoad, LightingLoad, ElectricEquipmentLoad, InfiltrationLoad, SetpointProgram
from simulation.energy_simulation import SimulationOutputSetup, SimulationParameterSetup, RunEnergySimulation
from simulation.daylight_simulation import DaylightSimulation
from results.energy_results import EnergySimResults
from results.energy_plotting import display_metrics_as_df, LoadBalanceBarPlot
from results.daylight_results import DaylightSimResults
from results.daylight_plotting import DaylightPlot, generate_zip


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
        sim_par= SimulationParameterSetup(sim_output=sim_output,
                                          epw_file=epw_file)          
        #run energy simulation 
        energy_sim=RunEnergySimulation(sim_par= sim_par, simulation_folder=simulation_folder,
                                       model=model)
        sql_path= energy_sim.run_simulation()

        #processing results
        energy_sim_processing= EnergySimResults(sql_path=sql_path, model=model)
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
        # Generate sensor grid
        sensor_grid= BoxModelSensorGrid(model= model, grid_size=grid_size).sensor_grid
        model.properties.radiance.add_sensor_grid(sensor_grid)
        
        # Run daylight simulation
        daylight_sim= DaylightSimulation(model=model, wea=wea)
        daylight_sim.run_annual_daylight_simulation(path)
        
        results_folder= make_folder_if_not_exist(os.path.join(path,"annual_daylight"),'metrics')
        daylight_results= DaylightSimResults(hb_model=model, results_folder=results_folder)
        daylight_results.load_and_add_results()
        daylight_results.set_display_modes()
        annual_metrics=daylight_results.annual_metrics

        vtk_path = Path(daylight_results.model.to_vtkjs(folder=path, name=model.identifier))
        vtkjs_name = f'{model.identifier}_vtkjs'
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
    model= st.session_state.model

    image_paths=[]
    output_image_folder = os.path.join(path, 'annual_daylight\\results')
    show_legend = st.checkbox("Show Legend", value=False)

    col1, col2, col3, col4, col5 = st.columns(5)

    for i in range(len(annual_metrics)):
        metric = annual_metrics[i]
        grids = [model.properties.radiance.sensor_grids[0]]

        plot= DaylightPlot(metric, grids)
        p,fig= plot.generate_fig()
        image_filepath= plot.save_fig(output_image_folder)

        # Toggling legend-colorbar + title
        # Could potentially move toggle to DaylightPlot class
        colorbar=fig.colorbar(p)
        if show_legend is True:
            colorbar.ax.set_title(metric['shortened'])
        else:
            colorbar.ax.set_title('')  # Remove the colorbar title
            colorbar.remove()

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

    # Provide a download button for the zip file
    zip_filename = os.path.join(output_image_folder,'Daylight_metrics.zip')
    zip_data = generate_zip(image_paths, zip_filename)

    st.download_button(label="Download Daylight Metrics", data=zip_data, file_name='Daylight_metrics.zip', mime="application/zip")

def main():
    pass

if __name__ == "__main__":
    main()