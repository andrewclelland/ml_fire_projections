"""
Code to split a long array up into smaller sections for testing months.

Edit as necessary.
"""
import numpy as np
"""
For reference:
1 month is 1471069 / 1297169 / 177589
1 year is 17652828 / 15566028 / 2131068
14 years is 247139592 / 217924392 / 29834952
17 years is 300098076 / 264622476 / 36228156
20 years is 353056560 / 311320560 / 42621360
23 years is 404543975 / 356721475 / 48836975 (minus Dec 2023)
(all / north / south)
"""

# Define the start and end points for splitting
split_points = [(280188504, 281485673), (281485673, 282782842), (282782842, 284080011), (284080011, 285377180), (285377180, 286674349), (286674349, 287971518), (287971518, 289268687), (289268687, 290565856), (290565856, 291863025), (291863025, 293160194), (293160194, 294457363), (294457363, 295754532)]

# Loop through each band
top_band_names = ['elevation', 'slope', 'aspect', 'month', 'latitude', 'longitude_sine']
for band in top_band_names:
    # Load the array for each band
    array = np.load(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north_all/training_north_0120_{band}_array.npy')
    
    # Loop through each start and end point to split the array
    for i, (start, end) in enumerate(split_points):
        # Extract the segment of the array for the given range
        array_segment = array[start:end]
        
        # Save each segment with an indexed file name
        np.save(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_north/testing_north_2019_{i+1}_{band}_array.npy', array_segment)
        
        # Print a confirmation message
        print(f"Month {i+1} for {band} saved.")

fwi_band_names = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI']
for band in fwi_band_names:
    # Load the array for each band
    array = np.load(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north_all/training_north_cems_0120_{band}_array.npy')
    
    # Loop through each start and end point to split the array
    for i, (start, end) in enumerate(split_points):
        # Extract the segment of the array for the given range
        array_segment = array[start:end]
        
        # Save each segment with an indexed file name
        np.save(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_north/testing_north_cems_2022_{i+1}_{band}_array.npy', array_segment)
        
        # Print a confirmation message
        print(f"Month {i+1} for {band} saved.")

clim_band_names = ['rh', 'pr_sum', 'rlds', 'rsds', 'sfcWind', 't2m', 'mx2t', 'mn2t']
for band in clim_band_names:
    # Load the array for each band
    array = np.load(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_south_all/training_south_e5l_2123_{band}_array.npy')
    
    # Loop through each start and end point to split the array
    for i, (start, end) in enumerate(split_points):
        # Extract the segment of the array for the given range
        array_segment = array[start:end]
        
        # Save each segment with an indexed file name
        np.save(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_south/testing_south_e5l_2023_{i+1}_{band}_array.npy', array_segment)
        
        # Print a confirmation message
        print(f"Month {i+1} for {band} saved.")
