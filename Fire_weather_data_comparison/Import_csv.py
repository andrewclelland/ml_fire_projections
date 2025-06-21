"""
Code to import the three fire processed fire weather csv files before finding correlations/biases/rmses/variance ratios. Only load two csv files at a time to compare them directly.

Ensure this file is in the same folder as `Merge_csv.py` and `Stats_csv.py`.
"""
import pandas as pd

# Only load two files at a time to compare them.
#cems_csv = pd.read_table('/home/users/clelland/Model/NASA_CEMS_comparison/cems_0114.csv', sep=",", index_col=['year', 'month'])
#cems_csv = cems_csv.rename(columns={'build_up_index': 'BUI', 'drought_code': 'DC', 'duff_moisture_code': 'DMC', 'fine_fuel_moisture_code': 'FFMC', 'fire_weather_index': 'FWI', 'initial_fire_spread_index': 'ISI'})

access_csv = pd.read_table('/home/users/clelland/Model/NASA_CEMS_comparison/access_0114.csv', sep=",", index_col=['year', 'month'])

mri_csv = pd.read_table('/home/users/clelland/Model/NASA_CEMS_comparison/mri_0114.csv', sep=",", index_col=['year', 'month'])

#df_all = pd.concat([cems_csv, access_csv, mri_csv], axis=0)