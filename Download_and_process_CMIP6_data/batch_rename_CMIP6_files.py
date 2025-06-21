"""
Rename the downloaded files to have consistent filenames
"""
import os

# v1.1 - hurs and pr
# v1.2 - tas and tasmin
source_directory = "/gws/nopw/j04/bas_climate/users/clelland/model/CMIP6/tas/" # <-- Edit as necessary
years = range(2001, 2101)

# Loop through all the files in the source directory
for file_name in os.listdir(source_directory):
    for year in years:
        # Check if the file ends with '.nc' for the given year
        if file_name.endswith(f"{year}_v1.1.nc"):
        #if file_name.endswith(f"{year}_v1.2.nc"):
            # Construct the full source file path
            source_file_path = os.path.join(source_directory, file_name)
    
            # Create the new file name
            new_file_name = file_name.replace(f"_{year}_v1.1.nc", f"_{year}.nc")
            #new_file_name = file_name.replace(f"_{year}_v1.2.nc", f"_{year}.nc")
                
            # Construct the full new file path
            new_file_path = os.path.join(source_directory, new_file_name)
    
            # Rename the file
            os.rename(source_file_path, new_file_path)

print("All matching files have been renamed.")