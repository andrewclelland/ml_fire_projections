"""
Code to remove validation months from complete training array.

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

top_band_names = ['elevation', 'slope', 'aspect', 'land_cover_og', 'firecci', 'month', 'latitude', 'longitude_sine']
fwi_band_names = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI']
clim_band_names = ['rh', 'pr_sum', 'rlds', 'rsds', 'sfcWind', 't2m', 'mx2t', 'mn2t']

# Define the start and end points for splitting
split_points = [(108962196, 124528224), (217924392, 233490420), (295754532, 311320560)]

# Loop through each band
for band in top_band_names:
    # Load the array for each band
    array = np.load(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north_all/training_north_0120_{band}_array.npy')

    # Create a mask for indices that are NOT in the specified split ranges
    mask = np.ones(len(array), dtype=bool)
    
    # Loop through each start and end point to split the array
    for start, end in split_points:
        mask[start:end] = False  # Set False for indices in each specified range

    # Extract the remainder of the array
    remainder_array = array[mask]
    
    # Save the remainder array
    np.save(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north/training_north_0120_{band}_array_reduced.npy', remainder_array)
    
    # Print a confirmation message
    print(f"Reduced array for {band} saved.")

for band in fwi_band_names:
    # Load the array for each band
    array = np.load(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north_all/training_north_cems_0120_{band}_array.npy')

    # Create a mask for indices that are NOT in the specified split ranges
    mask = np.ones(len(array), dtype=bool)
    
    # Loop through each start and end point to split the array
    for start, end in split_points:
        mask[start:end] = False  # Set False for indices in each specified range

    # Extract the remainder of the array
    remainder_array = array[mask]
    
    # Save the remainder array
    np.save(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north/training_north_cems_0120_{band}_array_reduced.npy', remainder_array)
    
    # Print a confirmation message
    print(f"Reduced array for {band} saved.")

for band in clim_band_names:
    # Load the array for each band
    array = np.load(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north_all/training_north_e5l_0120_{band}_array.npy')

    # Create a mask for indices that are NOT in the specified split ranges
    mask = np.ones(len(array), dtype=bool)
    
    # Loop through each start and end point to split the array
    for start, end in split_points:
        mask[start:end] = False  # Set False for indices in each specified range

    # Extract the remainder of the array
    remainder_array = array[mask]
    
    # Save the remainder array
    np.save(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north/training_north_e5l_0120_{band}_array_reduced.npy', remainder_array)
    
    # Print a confirmation message
    print(f"Reduced array for {band} saved.")