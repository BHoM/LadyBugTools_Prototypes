import json
import os

from ladybug.epw import EPW
from honeybee.model import Model
from honeybee.room import Room
from dataclasses import dataclass, field
from honeybee_energy.simulation.output import SimulationOutput
from honeybee_energy.simulation.parameter import SimulationParameter

from dataclasses import dataclass

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
    room: Room
    model: Model
    simulation_folder:str
    epw_obj: EPW

    def __post_init__(self):
        self.sim_par = SimulationParameter(output=self.sim_output)

    def add_autocalculated_design_days(self):
        des_days = [
            self.epw_obj.approximate_design_day('WinterDesignDay'),
            self.epw_obj.approximate_design_day('SummerDesignDay')
        ]
        self.sim_par.sizing_parameter.design_days = des_days

    def model_to_json(self):
        hbjson_path = self.model.to_hbjson(folder=self.simulation_folder)
        return hbjson_path

    def sim_par_to_json(self):
        sim_par_json = self.sim_par.to_dict()
        sim_par_path = os.path.join(self.simulation_folder, 'sim_par.json')
        with open(sim_par_path, 'w') as json_file:
            json.dump(sim_par_json, json_file)
        return sim_par_path
