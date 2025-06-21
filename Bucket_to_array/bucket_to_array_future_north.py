"""
Code to process the GeoTIFF images stored in the Google Cloud Storage bucket to NumPy arrays, suitable for reading into the machine learning models.

This file relates to the future period for the Northern model.

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
final_north = ee.FeatureCollection('users/andyc97/model_shapefiles/final_north')

# Define the band names
# FWI
band_names = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'FWI_N15', 'FWI_N30', 'FWI_N45', 'FWI_NP95', 'FWI_Nmid', 'ISI'] # raw FWI
chosen_bands = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI']

# Climate - DO LAND COVER G1, PR AND RADIATION SEPARATELY
#band_names = ['hurs', 'huss', 'pr', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin'] # raw CMIP6
#chosen_bands = ['hurs', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin'] # CMIP6
#chosen_bands = ['pr'] # treat precipitation separately

# Google Cloud Storage bucket path template - ensure Trevor given permission
# CHANGE MODEL AS REQUIRED
gcs_template = 'gs://clelland_fire_ml/FWI_files/ACCESS-CM2_COG/{folder}/ACCESS-CM2_{scenario}_{year}_{month}_cog.tif' # FWI
#gcs_template = 'gs://clelland_fire_ml/CMIP6_files/ACCESS-CM2_COG/{folder}/ACCESS-CM2_{scenario}_{year}_{month}_all_cog.tif' # CMIP6

folders = ['SSP126', 'SSP245', 'SSP370']
scenarios = ['ssp126', 'ssp245', 'ssp370']
years = range(2025, 2101)  # 2025 to 2100
months = range(1, 13)  # 1 to 12

# To be used as a mask
other = ee.Image.loadGeoTIFF('gs://clelland_fire_ml/training_nasa_access_firecci/nasa_access_firecci_2001_1.tif').select('aspect')

# Function to generate an ee.Image from a GCS path
def create_image(folder, scenario, year, month, selected_bands):
    file_path = gcs_template.format(folder=folder, scenario=scenario, year=year, month=month)
    image = ee.Image.loadGeoTIFF(file_path)
    # Get the last day of the month
    _, last_day = monthrange(year, month)

    # Rename the bands of the image and choose selected bands
    image = image.rename(band_names).select(selected_bands)

    # Reproject image
    image = image.reproject(crs='EPSG:6931', scale=4000) 

    # For pr to pr_sum ONLY - to convert to metres
    #seconds_in_month = last_day * 24 * 60 * 60
    #image = image.multiply(seconds_in_month)

    # Fill missing data with -9999, clip and update the mask to another image
    image = image.unmask(-9999, sameFootprint=True).clip(final_north).updateMask(other)
    
    # Set properties including the start and end dates
    image = image.set({
        'year': year,
        'month': month,
    })
    return image

# Iterate over scenarios, years, and months to process images
for folder, scenario in zip(folders, scenarios):
    for year in years:
        for month in months:
            try:
                image = create_image(folder, scenario, year, month, chosen_bands)
                extracted_data = final_grid.map(lambda cell: cell.set(image.reduceRegion(
                    reducer=ee.Reducer.toList(),
                    geometry=cell.geometry(),
                    scale=4000,
                    bestEffort=False,
                    crs='EPSG:6931',
                    maxPixels=1e9
                )))
                
                features = extracted_data.getInfo()['features']
                concatenated_data = {band: [] for band in chosen_bands}
                
                for feature in features:
                    for band in chosen_bands:
                        data = np.array(feature['properties'].get(band, []))
                        data = np.where(data == -9999, np.nan, data)
                        concatenated_data[band].extend(data)
                
                for band in chosen_bands:
                    final_band_array = np.array(concatenated_data[band]).reshape(-1, 1)
                    filename = f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_access_north/{scenario}/testing_access_north_{scenario}_{year}_{month}_{band}_array.npy' # <-- Edit as necessary
                    np.save(filename, final_band_array)
                    print(f"Saved: {filename} with shape {final_band_array.shape}")
            except Exception as e:
                print(f"Error processing {scenario} {year}-{month}: {e}")

# Calculate the time difference
end_time = time.time()
time_difference_seconds = end_time - start_time
time_difference_hours = time_difference_seconds / 3600  # Convert seconds to hours
print(f"Time taken: {time_difference_hours:.2f} hours")
