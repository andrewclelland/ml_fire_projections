"""
Make violin plots of processed fire weather data from csv files. Other plots can be made by editing `sns.violinplot` to any other type of plot (e.g. boxplot)

Ensure `Import_csv_for_plot.csv` is in the same folder.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from Import_csv_for_plot import *

f = plt.figure(figsize=[4,6])

# Extra code to hide outliers
df_plot = df_merged.copy()
df_plot = df_plot.rename(columns={'ISI_x': 'CEMS', 'ISI_y': 'ACCESS', 'ISI': 'MRI'})
Q1 = df_plot['CEMS'].quantile(0.25)
Q3 = df_plot['CEMS'].quantile(0.75)
IQR = Q3 - Q1

upper_whisker = Q3 + 1.5 * IQR
upper_whisker_value = df_plot['CEMS'][df_plot['CEMS'] <= upper_whisker].max()

df_plot = df_plot[(df_plot <= upper_whisker_value).all(axis=1)]

# Normal violin plot
columns_to_plot = ['CEMS', 'ACCESS', 'MRI']

df_melted = df_plot[columns_to_plot].reset_index().melt(id_vars=['year', 'month', 'latitude', 'longitude'], 
                                                        value_vars=columns_to_plot, 
                                                        var_name='ISI Component', 
                                                        value_name='Value')

ax = f.add_subplot(111)
ax = sns.violinplot(x='ISI Component', y='Value', data=df_melted, palette='husl', hue='ISI Component') # <-- edit this line to change type of plot

# Add labels and title
plt.title('ISI ranges')
plt.xlabel('')
plt.ylabel('')

# Show the plot
plt.savefig('isi_range_violin.png', dpi=300, bbox_inches='tight') # <-- Edit output location
plt.show()