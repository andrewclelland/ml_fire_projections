"""
Code to split a numpy array at a single point.

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

#band_names = ['elevation', 'slope', 'aspect', 'land_cover', 'firecci', 'mcd', 'month', 'latitude', 'longitude_sine']
#band_names = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI']
#band_names = ['rh', 'pr_sum', 'rlds', 'rsds', 'sfcWind', 't2m', 'mx2t', 'mn2t']
band_names = ['latitude']

# Define the split point
split_point = 1471069

# Loop through each band
for band in band_names:
    # Load the array for each band
    array = np.load(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_data_all/training_0120_{band}_array.npy')
    
    # Find the length of the array
    length = len(array)
    print(f"Length of the array for {band}: {length}")
    
    # Split the array at the specified point
    array1, array2 = np.split(array, [split_point])
    
    # Save the split arrays
    #np.save(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_data/training_e5l_0120_{band}_array.npy', array1)
    np.save('/home/users/clelland/Model/lat_all_single.npy', array1)
    
    # Print a confirmation message
    print(f"Array for {band} split and saved as 0120.")

    # Find the length of the new array
    length1 = len(array1)
    print(f"Length of new array 1 for {band}: {length1}")