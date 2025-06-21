"""
Code to batch combine numpy arrays for a set of variables. Example use is to combine 2001-2020 data with 2021-2023 data.

Edit as necessary.
"""
import numpy as np

#band_names = ['elevation', 'slope', 'aspect', 'land_g1', 'month', 'latitude', 'longitude_sine']
#band_names = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI']
band_names = ['rh', 'pr_sum', 'rlds', 'rsds', 'sfcWind', 't2m', 'mx2t', 'mn2t']

# Loop through each band and combine arrays
for band in band_names:
    # Load the arrays for each band
    array1 = np.load(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north_all/training_north_e5l_0120_{band}_array.npy')
    array2 = np.load(f'/gws/nopw/j04/bas_climate/users/clelland/model/temp_files/training_north_e5l_2123_{band}_array_short.npy')
    
    # Combine the arrays
    combined_array = np.concatenate((array1, array2))
    
    # Save the combined array
    np.save(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north_all/training_north_e5l_0123_{band}_array.npy', combined_array)
    
    # Print a confirmation message
    print(f"Combined array for {band} saved.")