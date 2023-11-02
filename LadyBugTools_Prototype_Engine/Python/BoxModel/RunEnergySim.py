import os
from pathlib import Path
from honeybee_energy.programtype import ProgramType
from honeybee_energy.hvac.idealair import IdealAirSystem

from honeybee_vtk.model import Model

from ladybug.epw import EPW
from .geometry.geom import BoxModelGlazing, BoxModelModel, BoxModelRoom, BoxModelSensorGrid

from .file_utils import make_folder_if_not_exist
from .construction.construction_set import BoxModelFabricProperties
from .program.program import PeopleLoad, LightingLoad, ElectricEquipmentLoad, InfiltrationLoad, SetpointProgram
from .simulation.energy_simulation import SimulationOutputSetup, SimulationParameterSetup, RunEnergySimulation
from .results.energy_results import EnergySimResults

from .results.energy_plotting import display_metrics_as_df, LoadBalanceBarPlot

#New functionality
def Run_EnergySimulation(model: Model, room, epw: EPW, path:str):
    #construction set
    epw_obj = EPW(epw)
    construct_set = BoxModelFabricProperties(epw = epw_obj).construction_set
    room.properties.energy.construction_set = construct_set

    #program type
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
    sim_output = SimulationOutputSetup().return_sim_output()
    sim_par = SimulationParameterSetup(sim_output=sim_output, epw_file=epw)   

    #run energy simulation 
    energy_sim = RunEnergySimulation(sim_par, path, model)
    sql_path = energy_sim.run_simulation()   

    #processing results
    energy_sim_processing= EnergySimResults(sql_path=sql_path, model=model)
    metrics = energy_sim_processing.metric_dictionary()
    monthly_balance = energy_sim_processing.monthy_balance()

    return monthly_balance, metrics

#to call from c#
def energy_sim(model: Model, room, epw: EPW, path:str):
    monthly_balance, metrics = Run_EnergySimulation(model, room, epw, path)
    
    results_folder = make_folder_if_not_exist(path,"results")

    output_image_folder = os.path.join(path, 'results\\result')
    
    metrics_output = display_metrics_as_df(metrics)
    fig_output = LoadBalanceBarPlot(monthly_balance).save_fig()
    fig_output.savefig(output_image_folder)
    
    return metrics_output, fig_output