"""
Script to combine the inputs into GeoTIFF files for model training and store in a Google Cloud Storage bucket.

The code can be edited to include different inputs, e.g. FireCCI instead of MCD64A1, and NASA-downsclaed CMIP6 and fire weather data rather than ERA5-Land and CEMS.

This step can be skipped by using bucket_to_array directly, however it doesn't take too long to run for this historic period (a few days).
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

# Consistent Parameters
years = range(2001, 2024)  # 2001 to 2023
months = range(1, 13)  # 1 to 12

# Import Parameters
climate_band_names = ['hurs', 'huss', 'pr', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin']
climate_selected_bands = ['hurs', 'pr', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin']
fwi_band_names = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'FWI_N15', 'FWI_N30', 'FWI_N45', 'FWI_NP95', 'FWI_Nmid', 'ISI']
fwi_selected_bands = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI']

# Function to generate an ee.Image from a GCS path
def create_image(year, month, gcs_template, band_names, selected_bands):
    file_path = gcs_template.format(year=year, month=month)
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
        'scenario': 'historical',
        'year': year,
        'month': month,
        'system:time_start': ee.Date(start_date).millis(),
        'system:time_end': ee.Date(end_date).millis()
    })
    return image

# Process each year and month
for year in years:
    for month in months:
        try:
            # ERA5-Land monthly means
            _, last_day = monthrange(year, month)
            era5land = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(f'{year}-{month:02d}-01', f'{year}-{month:02d}-{last_day:02d}')
            pr = era5land.select('total_precipitation_sum').mean().reproject(crs='EPSG:6931', scale=4000).divide(last_day * 24 * 60 * 60) # Convert to seconds per month
            strd = era5land.select('surface_thermal_radiation_downwards_sum').mean().reproject(crs='EPSG:6931', scale=4000)
            ssrd = era5land.select('surface_solar_radiation_downwards_sum').mean().reproject(crs='EPSG:6931', scale=4000)
            t2m = era5land.select('temperature_2m').mean().reproject(crs='EPSG:6931', scale=4000)
            mx2t = era5land.select('temperature_2m_max').max().reproject(crs='EPSG:6931', scale=4000)
            mn2t = era5land.select('temperature_2m_min').min().reproject(crs='EPSG:6931', scale=4000)

            def calculate_rh(image):
                temp = image.select('temperature_2m').subtract(273.15)
                dewpoint = image.select('dewpoint_temperature_2m').subtract(273.15)
                # Calculate saturation vapor pressure for temperature and dewpoint, then rh
                e_t = image.expression('6.112 * exp((17.67 * temp) / (temp + 243.5))', {'temp': temp})
                e_td = image.expression('6.112 * exp((17.67 * dewpoint) / (dewpoint + 243.5))', {'dewpoint': dewpoint})
                rh = e_td.divide(e_t).multiply(100).clamp(0, 100)  # Clamp to range [0, 100]
                return rh

            def calculate_wsp(image):
                u10 = image.select('u_component_of_wind_10m')
                v10 = image.select('v_component_of_wind_10m')
                wspe5l = (u10.pow(2).add(v10.pow(2))).sqrt()
                return wspe5l
            
            hurs = era5land.map(calculate_rh).mean().reproject(crs='EPSG:6931', scale=4000).multiply(100).round().divide(100).rename("relative_humidity")
            wsp = era5land.map(calculate_wsp).mean().reproject(crs='EPSG:6931', scale=4000).multiply(100).round().divide(100)

            climate_image = hurs.addBands(pr).addBands(strd).addBands(ssrd).addBands(wsp).addBands(t2m).addBands(mx2t).addBands(mn2t).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask).toFloat()
            
            # Import downscaled climate data
            #climate_gcs_template = f'gs://clelland_fire_ml/CMIP6_files/ACCESS-CM2_COG/Historical/ACCESS-CM2_historical_{year}_{month}_all_cog.tif' # Edit model
            #climate_image = create_image(year, month, climate_gcs_template, climate_band_names, climate_selected_bands).unmask(-9999, sameFootprint=True).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)

            # CEMS FWI data - make monthly means
            _, last_day = monthrange(year, month)
            cemsfwi = ee.ImageCollection("projects/climate-engine-pro/assets/ce-cems-fire-daily-4-1").filterDate(f'{year}-{month:02d}-01', f'{year}-{month:02d}-{last_day:02d}')
            buildupindex = cemsfwi.select('build_up_index').mean().reproject(crs='EPSG:6931', scale=4000)
            droughtcode = cemsfwi.select('drought_code').mean().reproject(crs='EPSG:6931', scale=4000)
            duffmoisturecode = cemsfwi.select('duff_moisture_code').mean().reproject(crs='EPSG:6931', scale=4000)
            finefuelmoisturecode = cemsfwi.select('fine_fuel_moisture_code').mean().reproject(crs='EPSG:6931', scale=4000)
            fireweatherindex = cemsfwi.select('fire_weather_index').mean().reproject(crs='EPSG:6931', scale=4000)
            initialspreadindex = cemsfwi.select('initial_fire_spread_index').mean().reproject(crs='EPSG:6931', scale=4000)

            # Deal with missing CEMS data if necessary
            bui_filled = buildupindex.unmask(-9999, sameFootprint=True).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)
            dc_filled = droughtcode.unmask(-9999, sameFootprint=True).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)
            dmc_filled = duffmoisturecode.unmask(-9999, sameFootprint=True).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)
            ffmc_filled = finefuelmoisturecode.unmask(-9999, sameFootprint=True).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)
            fwi_filled = fireweatherindex.unmask(-9999, sameFootprint=True).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)
            isi_filled = initialspreadindex.unmask(-9999, sameFootprint=True).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)

            fwi_image =  bui_filled.addBands(dc_filled).addBands(dmc_filled).addBands(ffmc_filled).addBands(fwi_filled).addBands(isi_filled).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)  
            
            # Import NASA FWI images
            #fwi_gcs_template = f'gs://clelland_fire_ml/FWI_files/ACCESS-CM2_COG/Historical/ACCESS-CM2_historical_{year}_{month}_cog.tif' # Edit model
            #fwi_image = create_image(year, month, fwi_gcs_template, fwi_band_names, fwi_selected_bands).unmask(-9999, sameFootprint=True).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)

            # Fire CCI data (filter by date)
            #_, last_day = monthrange(year, month)
            #firecci = ee.ImageCollection("ESA/CCI/FireCCI/5_1").filterDate(f'{year}-{month:02d}-01', f'{year}-{month:02d}-{last_day:02d}')
            #julianDay = firecci.select('BurnDate').min()
            #confLevel = firecci.select('ConfidenceLevel').max()
            #CLmask = confLevel.gte(ee.Number(80))
            #JDmasked = julianDay.updateMask(CLmask)
            #upscaled_firecci = JDmasked.reproject(crs='EPSG:6931', scale=4000)
            #clipped_fire = upscaled_firecci.multiply(0).add(1).unmask(0).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask).toFloat()

            # MCD64A1 data (filter by date)
            _, last_day = monthrange(year, month)
            mcd64a1 = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(f'{year}-{month:02d}-01', f'{year}-{month:02d}-{last_day:02d}')
            burnDate = mcd64a1.select('BurnDate').min()
            upscaled_mcd64a1 = burnDate.reproject(crs='EPSG:6931', scale=4000)
            clipped_fire = upscaled_mcd64a1.multiply(0).add(1).unmask(0).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask).toFloat()
            
            # Clip to final region and mask
            lat = ee.Image.pixelLonLat().select('latitude').updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)
            lon = ee.Image.pixelLonLat().select('longitude').updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)
            lon_rad = lon.multiply(math.pi / 180)
            lon_sin = lon_rad.sin().rename("longitude_sine")
            
            coords_clipped = lat.addBands(lon_sin).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask).toFloat()
            month_clipped = ee.Image.constant(month).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask).toFloat().rename("month")
            
            # Create the final image
            final_hist = base_land.addBands(clipped_fire).addBands(climate_image).addBands(fwi_image).addBands(month_clipped).addBands(coords_clipped).updateMask(aspect).updateMask(t2m).clip(final).updateMask(final_LandMask)
            print(f"Creating final_hist_{year}_{month}")
            
            # Export to cloud storage
            task = ee.batch.Export.image.toCloudStorage(
                image=final_hist,
                description=f'nasa_access_firecci_{year}_{month}',
                bucket='clelland_fire_ml', # <-- Edit as necessary
                fileNamePrefix=f'training_nasa_access_firecci/nasa_access_firecci_{year}_{month}', # <-- Edit as necessary
                region=final.geometry(),
                scale=4000,
                crs='EPSG:6931',
                maxPixels=1e9,
                formatOptions={'cloudOptimized': True}
            )
            task.start()
            
            time.sleep(240) # Add delay
            
        except Exception as e:
            print(f"Error processing {year}-{month}: {e}")

# Calculate the time difference
end_time = time.time()
time_difference_seconds = end_time - start_time
time_difference_hours = time_difference_seconds / 3600  # Convert seconds to hours
print(f"Time taken: {time_difference_hours:.2f} hours")