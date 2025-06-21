"""
Code file to import processed CEMS/ACCESS/MRI aggregated fire weather csv files ready to plot.

Keep in same folder as `Make_plot.py`.
"""
import pandas as pd

cems_csv = pd.read_table('/home/users/clelland/Model/NASA_CEMS_comparison/cems_0114.csv', sep=",", index_col=['year', 'month'])
cems_csv = cems_csv.rename(columns={'build_up_index': 'BUI', 'drought_code': 'DC', 'duff_moisture_code': 'DMC', 'fine_fuel_moisture_code': 'FFMC', 'fire_weather_index': 'FWI', 'initial_fire_spread_index': 'ISI'})
cems_csv_single = cems_csv[['ISI', 'latitude', 'longitude']]
cems_csv_reset = cems_csv_single.reset_index()

access_csv = pd.read_table('/home/users/clelland/Model/NASA_CEMS_comparison/access_0114.csv', sep=",", index_col=['year', 'month'])
access_csv_single = access_csv[['ISI', 'latitude', 'longitude']]
access_csv_reset = access_csv_single.reset_index()

mri_csv = pd.read_table('/home/users/clelland/Model/NASA_CEMS_comparison/mri_0114.csv', sep=",", index_col=['year', 'month'])
mri_csv_single = mri_csv[['ISI', 'latitude', 'longitude']]
mri_csv_reset = mri_csv_single.reset_index()

# Merge the dataframes on 'latitude' and 'longitude' columns, keeping only rows with matching lat/lon for the specified year
df_merged = cems_csv_reset.merge(access_csv_reset, on=['latitude', 'longitude', 'year', 'month'], how='inner')
df_merged = df_merged.merge(mri_csv_reset, on=['latitude', 'longitude', 'year', 'month'], how='inner')

# Optional: Reset the index or set 'year' as an index, depending on further usage
df_merged.set_index(['year', 'month', 'latitude', 'longitude'], inplace=True)
df_merged = df_merged.round(3) # Round to 3dp