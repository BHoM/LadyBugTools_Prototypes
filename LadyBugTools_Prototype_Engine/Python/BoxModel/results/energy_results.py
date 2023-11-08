from dataclasses import dataclass
from honeybee_energy.result.loadbalance import LoadBalance
from honeybee.model import Model


@dataclass
class EnergySimResults:
    sql_path: str
    model: Model

    def __post_init__(self):
        self.load_balance = LoadBalance.from_sql_file(self.model, self.sql_path)
        self.norm_bal_stor = self.load_balance.load_balance_terms(True, True) 

    def metric_dictionary(self):
        print(self.norm_bal_stor)
        try:
            outputs = {
                'annual solar gain (kWh/m2)': self.norm_bal_stor[1].to_unit('kWh/m2').total,
                'peak solar gain (Wh/m2)': self.norm_bal_stor[1].to_unit('Wh/m2').max,
                'annual heating demand (kWh/m2)': self.norm_bal_stor[0].to_unit('kWh/m2').total,
                'peak heating demand (Wh/m2)': self.norm_bal_stor[0].to_unit('Wh/m2').max,
                'annual cooling demand (kWh/m2)': -(self.norm_bal_stor[7].to_unit('kWh/m2').total),
                'peak cooling demand (Wh/m2)': -(self.norm_bal_stor[7].to_unit('Wh/m2').min),
                'annual lighting energy (kWh/m2)': self.norm_bal_stor[3].to_unit('kWh/m2').total,
                'annual external conduction (kWh/m2)': -(self.norm_bal_stor[6].to_unit('kWh/m2').total + self.norm_bal_stor[7].to_unit('kWh/m2').total)
            }
        except:
            raise ValueError("The balance storage:" + str(self.norm_bal_stor))
        return outputs
    
    def monthy_balance(self):
        monthly_energy = [term.total_monthly() for term in self.norm_bal_stor]
        return monthly_energy