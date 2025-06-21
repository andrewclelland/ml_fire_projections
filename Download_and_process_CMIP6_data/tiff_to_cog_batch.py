"""
Code to convert GeoTIFF files to Cloud Optimized GeoTIFFs (COGs) for Google Cloud Storage buckets.
"""
import subprocess
import time

# Record the start time
start_time = time.time()

# Define the scenarios, years, and months
#scenarios = ['historical']
#years = range(2001, 2015)
#folders = ['Historical']
scenarios = ['ssp126', 'ssp245', 'ssp370']
years = range(2015, 2101)
folders = ['SSP126', 'SSP245', 'SSP370']
months = range(1, 13)

# Define the base input and output paths (edit as necessary)
in_base = '/gws/nopw/j04/bas_climate/users/clelland/CMIP6/combined/MRI-ESM2-0_{scenario}_{year}_{month}_all.tif'
out_base = '/gws/nopw/j04/bas_climate/users/clelland/CMIP6/MRI-ESM2-0_COG/{folder}/MRI-ESM2-0_{scenario}_{year}_{month}_all_cog.tif'

# Loop through all folders, scenarios, years, and months
for folder, scenario in zip(folders, scenarios):
    for year in years:
        for month in months:
            # Format the input and output file paths
            in_image = in_base.format(scenario=scenario, year=year, month=month)
            out_image = out_base.format(folder=folder, scenario=scenario, year=year, month=month)
            
            # Construct the gdal_translate command
            command = [
                'gdal_translate', in_image, out_image,
                '-of', 'COG',
                '-projwin_srs', 'EPSG:4326'            
            ]
            
            # Run the gdal_translate command
            subprocess.run(command, check=True)

# Calculate the time difference
end_time = time.time()
time_difference_seconds = end_time - start_time
time_difference_hours = time_difference_seconds / 3600  # Convert seconds to hours
print(f"Time taken: {time_difference_hours:.2f} hours")