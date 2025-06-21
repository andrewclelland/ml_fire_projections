"""
Optional step to combine all the inputs into one GeoTIFF image, stored in a Google Cloud Storage bucket.

In reality - it will take forever (~ 2 months) and it is better to skip this step by using bucket_to_array directly.
"""
import ee
from calendar import monthrange
import math
import time

# Record the start time
start_time = time.time()

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='spherical-berm-323321')

# Define shapefile and land mask
final = ee.FeatureCollection('users/andyc97/model_shapefiles/final_shapefile')

tem_map = ee.Image('users/andyc97/model_shapefiles/TEM_LandCover_Map_V4')
tem_map = tem_map.updateMask(tem_map.neq(0)).toFloat()
upscaled_tem = tem_map.reproject(crs='EPSG:6931', scale=4000)
final_LandMask = upscaled_tem.multiply(0).add(1).unmask(0).updateMask(upscaled_tem.neq(0))

# Import base_land layer
base_land = ee.Image('users/andyc97/model_shapefiles/final_baseland')
aspect = base_land.select('aspect')

# ERA5-Land import - to clip final region
era5land = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate('2023-12-01', '2023-12-31')
t2m = era5land.select('temperature_2m').mean().reproject(crs='EPSG:6931', scale=4000).toFloat()

# Consistent Parameters
folders = ['SSP126', 'SSP245', 'SSP370']
scenarios = ['ssp126', 'ssp245', 'ssp370']
years = range(2025, 2101)  # 2025 to 2100
months = range(1, 13)  # 1 to 12

# Import Parameters
climate_band_names = ['hurs', 'huss', 'pr', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin']
climate_selected_bands = ['hurs', 'pr', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin']
fwi_band_names = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'FWI_N15', 'FWI_N30', 'FWI_N45', 'FWI_NP95', 'FWI_Nmid', 'ISI']
fwi_selected_bands = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI']

# Function to generate an ee.Image from a GCS path
def create_image(folder, scenario, year, month, gcs_template, band_names, selected_bands):
    file_path = gcs_template.format(folder=folder, scenario=scenario, year=year, month=month)
    image = ee.Image.loadGeoTIFF(file_path)
    _, last_day = monthrange(year, month)
    
    # Define the start and end dates
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-{last_day:02d}'

    # Rename the bands of the image
    image = image.rename(band_names)

    # Reproject image
    image = image.reproject(crs='EPSG:6931', scale=4000)    
    
    # Select specific bands
    image = image.select(selected_bands)
    
    # Set properties including the start and end dates
    image = image.set({
        'scenario': scenario,
        'year': year,
        'month': month,
        'system:time_start': ee.Date(start_date).millis(),
        'system:time_end': ee.Date(end_date).millis()
    })
    return image

# Process each year and month
for year in years:
    for month in months:
        for folder, scenario in zip(folders, scenarios):
            try:
                # Import FWI images
                fwi_gcs_template = f'gs://clelland_fire_ml/FWI_files/ACCESS-CM2_COG/{folder}/ACCESS-CM2_{scenario}_{year}_{month}_cog.tif' # <-- Edit as necessary
                fwi_image = create_image(folder, scenario, year, month, fwi_gcs_template, fwi_band_names, fwi_selected_bands).unmask(-9999, sameFootprint=True).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask) 
                
                # Import downscaled climate data
                climate_gcs_template = f'gs://clelland_fire_ml/CMIP6_files/ACCESS-CM2_COG/{folder}/ACCESS-CM2_{scenario}_{year}_{month}_all_cog.tif' # <-- Edit as necessary
                climate_image = create_image(folder, scenario, year, month, climate_gcs_template, climate_band_names, climate_selected_bands).unmask(-9999, sameFootprint=True).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask) 

                # Clip to final region and mask
                lat = ee.Image.pixelLonLat().select('latitude').updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)
                lon = ee.Image.pixelLonLat().select('longitude').updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)
                lon_rad = lon.multiply(math.pi / 180)
                lon_sin = lon_rad.sin().rename("longitude_sine")
                
                coords_clipped = lat.addBands(lon_sin).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask).toFloat()
                month_clipped = ee.Image.constant(month).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask).toFloat().rename("month")
                
                # Create the final image
                final_image = base_land.addBands(climate_image).addBands(fwi_image).addBands(month_clipped).addBands(coords_clipped).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)
                print(f"Creating final_{scenario}_{year}_{month}")
                
                # Export to cloud storage
                task = ee.batch.Export.image.toCloudStorage(
                    image=final_image,
                    description=f'final_{scenario}_{year}_{month}',
                    bucket='clelland_fire_ml', # <-- Edit as necessary
                    fileNamePrefix=f'ACCESS-CM2_future/{scenario}/access_{scenario}_{year}_{month}', # <-- Edit as necessary
                    region=final.geometry(),
                    scale=4000,
                    crs='EPSG:6931',
                    maxPixels=1e9,
                    formatOptions={'cloudOptimized': True}
                )
                task.start()
                
                time.sleep(250) # Add delay
                
            except Exception as e:
                print(f"Error processing {scenario} {year}-{month}: {e}")

# Calculate the time difference
end_time = time.time()
time_difference_seconds = end_time - start_time
time_difference_hours = time_difference_seconds / 3600  # Convert seconds to hours
print(f"Time taken: {time_difference_hours:.2f} hours")