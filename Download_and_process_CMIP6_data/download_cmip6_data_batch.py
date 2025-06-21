"""
Code to batch download CMIP6 data from remote server and store locally
"""
import os
import requests
import time

# Record the start time
start_time = time.time()

# Base URL of the remote server
BASE_URL = "https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/NEX-GDDP-CMIP6/ACCESS-CM2/{scenario}/r1i1p1f1/{variable}" # <-- Change for each model

# Scenarios and year range
#scenarios = ["historical"]
scenarios = ["ssp126", "ssp245", "ssp370"]
#years = range(2001, 2015)
years = range(2015, 2101)
variables = ["huss", "rlds", "rsds", "sfcWind", "tasmax"] # no version - edit as necessary
#variables = ["hurs", "pr"] # v1.1
#variables = ["tas", "tasmin"] # v1.2

# Function to create local directory and ensure it exists
def create_directory(variable):
    local_directory = f"/gws/nopw/j04/bas_climate/users/clelland/CMIP6/{variable}" # <-- Edit local directory location
    os.makedirs(local_directory, exist_ok=True)
    return local_directory

# Function to download a file
def download_file(variable, scenario, year, local_directory):
    # Try both file name formats (with and without v1.1)
    for version in [""]:
#    for version in ["_v1.1"]:
#    for version in ["_v1.2"]:
        # Construct the file name and URL - change for each model
        file_name = f"{variable}_day_ACCESS-CM2_{scenario}_r1i1p1f1_gn_{year}{version}.nc"
        file_url = f"{BASE_URL}/{file_name}".format(scenario=scenario, variable=variable)
        
        # Local file path
        local_file_path = os.path.join(local_directory, file_name)
        
        # Try downloading the file
        try:
            print(f"Attempting to download {file_name}...")
            response = requests.get(file_url, stream=True)
            response.raise_for_status()  # Check if the download was successful

            # Save the file to the local directory
            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded {file_name} to {local_file_path}")
            return  # Exit the function if the download is successful

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {file_name}: {e}")
            # Continue to try the next version if the current one fails

# Loop through each scenario and year to download all files
for variable in variables:
    # Create local directory for each variable
    local_directory = create_directory(variable)
    
    for scenario in scenarios:
        for year in years:
            download_file(variable, scenario, year, local_directory)

# Calculate the time difference
end_time = time.time()
time_difference_seconds = end_time - start_time
time_difference_hours = time_difference_seconds / 3600  # Convert seconds to hours
print(f"Time taken: {time_difference_hours:.2f} hours")