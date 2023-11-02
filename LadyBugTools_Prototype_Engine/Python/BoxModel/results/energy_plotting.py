from dataclasses import dataclass
import pandas as pd
from collections import defaultdict
from .daylight_plotter import build_custom_continuous_cmap
import matplotlib.pyplot as plt
from ladybug.color import Colorset

def display_metrics_as_df(metrics):
    metrics_df = pd.DataFrame.from_dict(metrics,orient='index').round(2)
    metrics_df.columns = [' ']
    return metrics_df

@dataclass
class LoadBalanceBarPlot:
    monthly_balance: list

    def save_fig(self):
        data = defaultdict(list)
        for i, monthly_coll in enumerate(self.monthly_balance):
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
        cmap= build_custom_continuous_cmap(color_set[19])

        # Create a stacked bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        df_melted.pivot_table(index='Month',columns='Load',values='Value', aggfunc='sum').plot(kind='bar', stacked=True, ax=ax, colormap=cmap, width=0.85, edgecolor= "black")
        ax.set_xlabel('')
        ax.set_ylabel('Energy (kWh)')  
        ax.set_title('Monthly Load Balance')
        ax.tick_params(axis='x', rotation=0)
        ax.set_xticklabels(['Jan','Feb', 'Mar', 'Apr','May','Jun','Jul', 'Aug','Sep','Oct','Nov','Dec'])
        ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))

        return fig
 

