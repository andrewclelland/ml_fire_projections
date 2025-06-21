"""
Code to process the GeoTIFF images stored in the Google Cloud Storage bucket to NumPy arrays, suitable for reading into the machine learning models.

This file relates to the historic period for the south Siberia model.

If you choose to use 'chosen_bands' to process separate bands, there are a couple of places where you will have to edit, and these are noted.

Ensure you have permission to access the GCS bucket (ask Trevor) and have a working GEE account linked to a project.
"""
import ee
from calendar import monthrange
import numpy as np
import time

# Record the start time
start_time = time.time()

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='spherical-berm-323321') # <-- Edit as necessary

# Import grid and any regions for clipping
final_grid = ee.FeatureCollection('users/andyc97/model_shapefiles/final_grid')
south_siberia = ee.FeatureCollection('users/andyc97/model_shapefiles/south_siberia_final')

# Import downscaled FWI into an image collection
years = range(2001, 2021)  # 2001 to 2020
months = range(1, 13)  # 1 to 12

# Define the band names
# DO LAND COVER G1 AND PR_SUM SEPARATELY
band_names = ['elevation', 'slope', 'aspect', 'land_cover_og', 'mcd', 'rh', 'pr_sum', 'rlds', 'rsds', 'sfcWind', 't2m', 'mx2t', 'mn2t', 'BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI', 'month', 'latitude', 'longitude_sine'] # all
#chosen_bands = ['pr_sum'] # treat precipitation separately
chosen_bands = ['rlds', 'rsds'] # treat radiation separately

# Google Cloud Storage bucket path template
gcs_template = f'gs://clelland_fire_ml/training_e5l_cems_mcd/cems_e5l_mcd_{year}_{month}.tif' # <-- Ask Trevor for permission

# To be used as a mask
other = ee.Image.loadGeoTIFF('gs://clelland_fire_ml/training_nasa_access_firecci/nasa_access_firecci_2001_1.tif').select('aspect')

# Function to generate an ee.Image from a GCS path
#def create_image(year, month):
def create_image(year, month, selected_bands): # If necessary
    file_path = gcs_template.format(year=year, month=month)
    image = ee.Image.loadGeoTIFF(file_path)
    # Get the last day of the month
    _, last_day = monthrange(year, month)
    
    # Define the start and end dates
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-{last_day:02d}'

    # Rename the bands of the image
    image = image.rename(band_names)

    # Select specific bands (if necessary)
    image = image.select(selected_bands)

    # For pr to pr_sum ONLY - to convert to metres
    #seconds_in_month = last_day * 24 * 60 * 60
    #image = image.multiply(seconds_in_month)

    # For radiation ONLY - to convert to W/m2
    seconds_in_month = last_day * 24 * 60 * 60
    image = image.divide(seconds_in_month)

    # Clip and update the mask to another image
    image = image.clip(south_siberia).updateMask(other)
    
    # Set properties including the start and end dates
    image = image.set({
        'year': year,
        'month': month,
        'system:time_start': ee.Date(start_date).millis(),
        'system:time_end': ee.Date(end_date).millis(),
    })
    return image

# Create an empty list to hold the images
image_list = []

# Loop through scenarios, years, and months, and import the files
for year in years:
    for month in months:
        # For MCD CEMS/E5L - Stop at November if the year is 2023
        if year == 2023 and month > 11:
            break
        try:
            #image = create_image(year, month)
            image = create_image(year, month, chosen_bands) # If necessary
            image_list.append(image)
        except Exception as e:
            print(f"Error loading image for {year}-{month}: {e}")

# Convert the list of images to an Earth Engine ImageCollection
training_images = ee.ImageCollection(image_list)

# Print the image collection size to confirm
print("Image collection created with", training_images.size().getInfo(), "images")

# Iterate over images - n slices
# Function to extract data for each grid cell from each image in the ImageCollection
def extract_data_for_cell(image):
    def extract_data(cell):
        data = image.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=cell.geometry(),
            scale=4000,
            bestEffort=False,
            crs='EPSG:6931',
            maxPixels=1e9)
        return cell.set(data)

    return final_grid.map(extract_data)

# List to hold individual FeatureCollections
feature_collections_list = []
n = 4 # number of slices

# Iterate over each image in the ImageCollection
for i in range(training_images.size().getInfo()):
    # Get the image by index
    image = ee.Image(training_images.toList(training_images.size()).get(i))

    # Extract data for each grid cell for the current image
    extracted_data = extract_data_for_cell(image)

    # Get the size of the extracted data
    fc_size = extracted_data.size().getInfo()

    # Calculate the size of each slice
    slice_size = fc_size // n

    # Create n slices
    for j in range(n):
        # Calculate the start index for the current slice
        start_index = j * slice_size
        # If it's the last slice, include the rest of the data
        if j == n - 1:
            end_index = fc_size
        else:
            end_index = start_index + slice_size

        # Define the current slice
        fc_part = extracted_data.toList(end_index - start_index, start_index)

        # Convert the list back to a FeatureCollection and append to the list
        feature_collections_list.append(ee.FeatureCollection(fc_part))

    print("Iteration over image", i+1, "complete")

def extract_and_concatenate(fc):
    # Retrieve the list of features
    features = fc.getInfo()['features']

    # Extract all property names
    property_names = features[0]['properties'].keys()

    # Initialize a dictionary to hold concatenated arrays for each property
    concatenated_data = {prop: [] for prop in property_names}

    # Iterate over each feature and append property values to the corresponding list in the dictionary
    for feature in features:
        for prop in property_names:
            data = np.array(feature['properties'][prop])
            data = np.where(data == -9999, np.nan, data) # Additional line to deal with NaNs
            concatenated_data[prop].extend(data)

    # Convert lists to NumPy arrays
    for prop in concatenated_data:
        concatenated_data[prop] = np.array(concatenated_data[prop])

    return concatenated_data

# Initialize a dictionary to hold the final concatenated data across all FeatureCollections
final_concatenated_data = {}

# Iterate over each FeatureCollection
for fc in feature_collections_list:
    concatenated_data = extract_and_concatenate(fc)
    for prop, array in concatenated_data.items():
        if prop not in final_concatenated_data:
            final_concatenated_data[prop] = array
        else:
            final_concatenated_data[prop] = np.concatenate((final_concatenated_data[prop], array))

# Iterate through each band
#for band in band_names:
for band in chosen_bands: # If necessary
    # Create the array for each band
    band_array = final_concatenated_data[f'{band}']
    final_band_array = np.reshape(band_array, (band_array.size, 1))
    
    # Print the shape of the reshaped array
    print(f'{band} shape:', final_band_array.shape)
    
    # Save the reshaped array as a .npy file
    print(f"Saving {band} array...")
    np.save(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_south_all/training_south_e5l_0120_{band}_array.npy', final_band_array) # <-- Edit as necessary

# Calculate the time difference
end_time = time.time()
time_difference_seconds = end_time - start_time
time_difference_hours = time_difference_seconds / 3600  # Convert seconds to hours
print(f"Time taken: {time_difference_hours:.2f} hours")