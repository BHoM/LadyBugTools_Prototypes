import json
import os


import streamlit as st
from ladybug.epw import EPW
from honeybee.model import Model
from honeybee_energy.result.loadbalance import LoadBalance
from dataclasses import dataclass, field
from honeybee_energy.simulation.output import SimulationOutput
from honeybee_energy.simulation.parameter import SimulationParameter

from honeybee_energy.run import to_openstudio_osw, run_osw, run_idf
from honeybee_energy.simulation.output import SimulationOutput
from honeybee_energy.simulation.parameter import SimulationParameter

@dataclass
class SimulationOutputSetup:
    sim_output: SimulationOutput = SimulationOutput()

    def __post_init__(self):
        self.sim_output.add_energy_balance_variables(load_type='Sensible')
        self.sim_output.add_comfort_metrics()
        self.sim_output.add_hvac_energy_use()
        self.sim_output.add_surface_temperature()
        self.sim_output.add_glazing_solar()

    def return_sim_output(self):
        return self.sim_output
        

@dataclass
class SimulationOutputSetup:
    sim_output: SimulationOutput = SimulationOutput()

    def __post_init__(self):
        self.sim_output.add_energy_balance_variables(load_type='Sensible')
        self.sim_output.add_comfort_metrics()
        self.sim_output.add_hvac_energy_use()
        self.sim_output.add_surface_temperature()
        self.sim_output.add_glazing_solar()

    def return_sim_output(self):
        return self.sim_output
        
@dataclass        
class SimulationParameterSetup:
    sim_output: SimulationOutput
    epw_file: str

    def __post_init__(self):
        self.epw_obj = EPW(file_path=self.epw_file)
        self.sim_par = SimulationParameter(output=self.sim_output)
        des_days = [
            self.epw_obj.approximate_design_day('WinterDesignDay'),
            self.epw_obj.approximate_design_day('SummerDesignDay')
        ]
        self.sim_par.sizing_parameter.design_days = des_days

    def sim_par_to_json(self, simulation_folder):
        sim_par_json = self.sim_par.to_dict()
        sim_par_path = os.path.join(simulation_folder, 'sim_par.json')
        with open(sim_par_path, 'w') as json_file:
            json.dump(sim_par_json, json_file)
        return sim_par_path



@dataclass
class RunEnergySimulation:
    sim_par: SimulationParameterSetup
    simulation_folder: str
    model: Model

    def __post_init__(self):
        self.sim_par_path = self.sim_par.sim_par_to_json(simulation_folder=self.simulation_folder)
        self.hbjson_path = self.model.to_hbjson(folder = self.simulation_folder) 

    def run_simulation(self):
        osw_json = to_openstudio_osw(osw_directory=self.simulation_folder,
                                model_path=self.hbjson_path,
                                sim_par_json_path=self.sim_par_path)
        osw, idf = run_osw(osw_json)
        self.sql, self.zsz, self.rdd, self.html, self.err = run_idf(idf_file_path=idf, epw_file_path=str(self.sim_par.epw_file))
        self.sql_path = os.path.join(self.simulation_folder, 'run\eplusout.sql')
        return self.sql_path

#idf is None?