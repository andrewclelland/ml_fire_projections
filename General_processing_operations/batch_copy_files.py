"""
Code file to batch copy files from one location to another.

Edit as necessary.
"""
import os
import shutil

# Source directory where the .tif files are located
source_directory = "/gws/nopw/j04/bas_climate/users/clelland/model/testing_access_south/"

# Destination directory where .tif files will be copied
destination_directory = "/gws/nopw/j04/bas_climate/users/clelland/model/testing_mri_south/"

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

def copy_tif_files(source_dir, dest_dir):
    # Loop through all the files in the source directory
    for file_name in os.listdir(source_dir):
        # Check if the file ends with .tif
        if file_name.endswith(".npy"):
            # Construct the full file paths
            source_file_path = os.path.join(source_dir, file_name)
            dest_file_path = os.path.join(dest_dir, file_name)
            
            # Copy the file to the destination directory
            shutil.copy2(source_file_path, dest_file_path)

# Copy all .tif files from the source to the destination
copy_tif_files(source_directory, destination_directory)
print("All .npy files have been copied.")