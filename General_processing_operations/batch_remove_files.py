"""
Code file to batch remove files.

Edit as necessary.
"""
import os

# Source directory where the files are located
source_directory = "/home/users/clelland/Model/FWI_files/"
#years = range(2001, 2015)
#years = range(2015, 2101)

# Loop through all the files in the source directory
for file_name in os.listdir(source_directory):
#    for year in years:
        # Check if the file name starts with "ACCESS-CM2_ssp370"
#    if file_name.startswith("testing"):
        # Check if the file ends with '.nc'
    if file_name.endswith("_2023_v2.nc"):
            # Construct the full file path
        source_file_path = os.path.join(source_directory, file_name) # Indent if 'ends with' is used as well

            # Remove the file
        os.remove(source_file_path) # Indent as above

print("All matching files have been deleted.")