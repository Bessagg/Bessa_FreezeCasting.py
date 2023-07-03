import pandas as pd
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pio.renderers.default = "browser"

# Load generated df
# import database2dataframe
df = pd.read_pickle('freeze_casting_df_complete.pkl')

# cooling_rate [K/min]. Temperature cold plate [K]

# Mean input averages by filter
filter_fluid = ['water']
filter_solid = ['Al2O3']

for solid in filter_solid:
    filter_sol = df.loc[(df['material'] == solid)]
    #print(filter_sol.value_counts())
    for fluid in filter_fluid:
        filtered_df = df.loc[(df['name_fluid1'] == fluid) & (df['material'] == solid)]

        # Calculate the average of all columns
        average_values = filtered_df.mean()
        print(solid, fluid, "\n", average_values, "\n")

        # plot vf
        plt.figure()
        filtered_df['vf_solid'].plot.hist(bins=30, alpha=0.5)
        plt.title(f'VF solid {solid}/{fluid}')

        # plot total
        plt.figure()
        filtered_df['vf_total'].plot.hist(bins=30, alpha=0.5)
        plt.title(f'VF total {solid}/{fluid}')


        # plot temp cold
        plt.figure()
        filtered_df['temp_cold'].plot.hist(bins=30, alpha=0.5)
        plt.title(f'Temp cold {solid}/{fluid}')

        # print("Press key for next loop")
        # plt.waitforbuttonpress()

# Check this sample
df[df['sample_ID'] == 2050][['material', 'name_fluid1', 'name_part1', 'porosity', 'paper_ID']]

# Check paper
df[df['paper_ID'] == 242][['material', 'vf_part_1', 'vf_fluid_1', 'vf_solid','name_fluid1', 'name_part1', 'porosity', 'paper_ID', 'sample_ID']]