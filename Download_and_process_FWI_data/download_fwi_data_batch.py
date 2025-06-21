"""
Code to batch download FWI data from remote server and store locally
"""
import os
import requests
import time

# Record the start time
start_time = time.time()

# Base URL of the remote server
BASE_URL = "https://data.nas.nasa.gov/gddpimpact/gddpimpactdata/FWI/Monthly/ACCESS-CM2" # <-- Change for each model

# Scenarios and year range
#scenarios = ["historical"]
scenarios = ["ssp126", "ssp245", "ssp370"]
#years = range(2001, 2015)
years = range(2015, 2101)

# Local directory to store the files - edit as necessary
local_directory = "/gws/nopw/j04/bas_climate/users/clelland/FWI"

# Ensure the local directory exists
os.makedirs(local_directory, exist_ok=True)

def download_file(scenario, year):
    # Construct the file name and URL
    file_name = f"ACCESS-CM2_{scenario}_fwi_metrics_monthly_{year}.nc" # <-- Change for each model
    file_url = f"{BASE_URL}/{file_name}"
    
    # Local file path
    local_file_path = os.path.join(local_directory, file_name)

    # Download the file
    try:
        print(f"Downloading {file_name}...")
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  # Check if the download was successful

        # Save the file to the local directory
        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded {file_name} to {local_file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file_name}: {e}")

# Loop through each scenario and year to download all files
for scenario in scenarios:
    for year in years:
        download_file(scenario, year)

# Calculate the time difference
end_time = time.time()
time_difference_seconds = end_time - start_time
time_difference_hours = time_difference_seconds / 3600  # Convert seconds to hours
print(f"Time taken: {time_difference_hours:.2f} hours")