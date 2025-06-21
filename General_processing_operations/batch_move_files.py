"""
Code file to batch move files from one location to another.

Edit as necessary.
"""
import os
import shutil

# Source directory where the files are located
source_directory = "/gws/nopw/j04/bas_climate/users/clelland/model/output_mri_north"

# Destination directory where you want to move the .tif files
destination_directory = "/gws/nopw/j04/bas_climate/users/clelland/model/output_v1"

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Loop through all the files in the source directory
for file_name in os.listdir(source_directory):
    # Check if the starts ends with...
#    if file_name.startswith("preds_xgb_south"):
#    # Check if the file ends with '.tif'
    if file_name.endswith("fwi.nc"):
            # Construct full file paths
        source_file_path = os.path.join(source_directory, file_name)
        destination_file_path = os.path.join(destination_directory, file_name)
    
            # Move the file
        shutil.move(source_file_path, destination_file_path)

print("All files have been moved.")