import os

from honeybee.model import Model
from honeybee_energy.run import to_openstudio_osw, run_osw, run_idf
from honeybee_energy.result.osw import OSW
from honeybee_energy.result.loadbalance import LoadBalance
from dataclasses import dataclass, field

@dataclass
class RunEnergySimulation:

    simulation_folder: str
    hbjson_path: str
    sim_par_path: str
    epw_file: str

    def run_simulation(self):
        osw = to_openstudio_osw(osw_directory=self.simulation_folder,
                                model_path=self.hbjson_path, sim_par_json_path=self.sim_par_path, epw_file=self.epw_file)
        osw, idf = run_osw(osw)
        self.sql, self.zsz, self.rdd, self.html, self.err = run_idf(idf_file_path=idf, epw_file_path=self.epw_file)
        self.sql_path = os.path.join(self.simulation_folder, 'run\eplusout.sql')
        return self.sql_path

    @staticmethod
    def calculate_load_balance(model, sql_path):
        load_balance = LoadBalance.from_sql_file(model, sql_path)
        return load_balance
        
