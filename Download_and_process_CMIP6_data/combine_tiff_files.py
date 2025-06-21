"""
Python file to combine single-band processed tif files for each CMIP6 variable to one tif file with multiple bands
"""
from rasterio import open as rio_open

# Scenarios and year range
#scenarios = ["historical"]
scenarios = ["ssp126", "ssp245", "ssp370"]
#years = range(2001, 2015)
years = range(2015, 2101)
variables = ["hurs", "huss", "pr", "rlds", "rsds", "sfcWind", "tas", "tasmax", "tasmin"] # Edit as required
months = range(1, 13)

def combine_tifs_for_all(output_dir, variables, scenarios, years, months):
    for scenario in scenarios:
        for year in years:
            for month in months:
                input_tifs = []
                
                # Build list of input TIFF files for each variable
                for variable in variables:
                    input_tif = f"/gws/nopw/j04/bas_climate/users/clelland/CMIP6/{variable}/MRI-ESM2-0_{variable}_{scenario}_{year}_{month}.tif"
                    print(f'Adding {input_tif}')
                    input_tifs.append(input_tif)
                
                # Output file path
                output_tif = f'{output_dir}/MRI-ESM2-0_{scenario}_{year}_{month}_all.tif'
                
                # Read the first input file to get metadata
                with rio_open(input_tifs[0]) as src:
                    meta = src.meta.copy()
                
                # Update the metadata for the new multiband file
                meta.update({
                    'count': len(variables)  # Set the number of bands
                })
                
                # Create the output file with all bands
                with rio_open(output_tif, 'w', **meta) as dst:
                    for i, input_tif in enumerate(input_tifs, start=1):
                        with rio_open(input_tif) as src:
                            dst.write(src.read(1), i)

# Output directory
output_dir = '/gws/nopw/j04/bas_climate/users/clelland/CMIP6/combined'

# Combine TIFFs for all variables, scenarios, years, and months
combine_tifs_for_all(output_dir, variables, scenarios, years, months)

print('Combination of TIFFs completed.')