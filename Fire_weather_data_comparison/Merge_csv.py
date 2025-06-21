"""
Code to merge the imported fire processed fire weather csv files before finding correlations/biases/rmses/variance ratios.

Ensure this file is in the same folder as `Import_csv.py` and `Stats_csv.py`.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from Import_csv import *

# Use same two csv files imported via `Import_csv.py`
#cems_csv_reset = cems_csv.reset_index()
access_csv_reset = access_csv.reset_index()
mri_csv_reset = mri_csv.reset_index()

# Merge the dataframes on 'latitude' and 'longitude' columns, keeping only rows with matching lat/lon for the specified year
#df_merged = cems_csv_reset.merge(access_csv_reset, on=['latitude', 'longitude', 'year', 'month'], how='inner')
#df_merged = cems_csv_reset.merge(mri_csv_reset, on=['latitude', 'longitude', 'year', 'month'], how='inner')
df_merged = access_csv_reset.merge(mri_csv_reset, on=['latitude', 'longitude', 'year', 'month'], how='inner')

#df_merged.set_index(['year', 'month', 'latitude', 'longitude'], inplace=True)

# Optional: Reset the index or set 'year' as an index, depending on further usage
print(df_merged.head())