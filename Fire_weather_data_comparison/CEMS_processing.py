"""
Code to convert CEMS fire weather data stored in the GEE Community Catalog to analysis-ready monthly aggregated means over a given region.

Ensure you have an account linked to a project in Google Earth Engine.
"""
import ee
ee.Authenticate()
ee.Initialize(project='spherical-berm-323321') # <-- Edit project as necessary
import calendar
import numpy as np
import pandas as pd

# Consistent Parameters - edit as necessary
final = ee.FeatureCollection('users/andyc97/model_preprocessed/final_shapefile') # shapefile of study region
quarter_degree_grid = ee.FeatureCollection('users/andyc97/model_preprocessed/quarter_degree_grid') # grid over study region

# Import downscaled FWI into an image collection
years = list(range(2001, 2015))  # 2001 to 2014 - for direct comparison with FWI data derived from CMIP6 data
months = list(range(1, 13))  # 1 to 12

# Google Cloud Storage bucket path template
gcs_template = 'gs://ce-cems-fire-daily-4-1/{year}{month}{day}.tif'

# Function to generate monthly mean image for a given year and month
def create_monthly_mean_image(year, month):
    # Get the number of days in the month
    _, last_day = calendar.monthrange(year, month)
    
    # Initialize an empty ImageCollection for the month
    monthly_images = ee.ImageCollection([])

    # Loop through each day of the month
    for day in range(1, last_day + 1):
        # Format the file path
        file_path = gcs_template.format(year=year, month=f'{month:02d}', day=f'{day:02d}')
        
        try:
            # Load the GeoTIFF image from GCS
            daily_image = ee.Image.loadGeoTIFF(file_path).clip(final)
                  
            # Handle missing pixels for BUI and FWI bands
            bui_filled = daily_image.select('build_up_index').unmask(-9999, sameFootprint=True).updateMask(daily_image.select('fine_fuel_moisture_code'))
            fwi_filled = daily_image.select('fire_weather_index').unmask(-9999, sameFootprint=True)
            
            # Replace the original BUI and FWI with the filled versions
            daily_image = daily_image.addBands([bui_filled.rename('build_up_index'), fwi_filled.rename('fire_weather_index')], overwrite=True)
            
            # Add the daily image to the collection
            monthly_images = monthly_images.merge(ee.ImageCollection([daily_image]))
        
        except Exception as e:
            # Handle missing files or errors in loading
            print(f'Failed to load {file_path}: {e}')
            continue

    # Calculate the mean of the monthly images
    print(f'Processing month {year}_{month}')
    monthly_mean_image = monthly_images.mean()
    
    # Get coordinates and add to the image
    coords = ee.Image.pixelLonLat().clip(final).updateMask(monthly_mean_image.select('fine_fuel_moisture_code'))
    monthly_mean_image = monthly_mean_image.addBands(coords)

    # Set properties including the start and end dates
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-{last_day:02d}'
    
    monthly_mean_image = monthly_mean_image.set({
        'scenario': 'historical',
        'year': year,
        'month': month,
        'system:time_start': ee.Date(start_date).millis(),
        'system:time_end': ee.Date(end_date).millis()
    })

    return monthly_mean_image

# Create an empty list to hold the images
image_list = []

# Loop through scenarios, years, and months, and import the files
for year in years:
    for month in months:
        try:
            image = create_monthly_mean_image(year, month)
            image_list.append(image)
        except Exception as e:
            print(f"Error loading image for {year}-{month}: {e}")

# Convert the list of images to an Earth Engine ImageCollection
monthly_mean_collection = ee.ImageCollection(image_list)

# Print the image collection size to confirm
print("Image collection created with", monthly_mean_collection.size().getInfo(), "images")

## Iterate over images
image_collection = monthly_mean_collection

# Function to extract data for all grid cells from each image in the ImageCollection
def extract_data_for_all_cells(image):
    # Use reduceRegions to process all grid cells at once, instead of iterating over them
    data = image.reduceRegions(
        collection=quarter_degree_grid,
        reducer=ee.Reducer.toList(),
        scale=27829.87269831839,
        crs='EPSG:4326'
    )
    return data

# Iterate over images and extract data for all grid cells at once
extracted_data_list = image_collection.map(extract_data_for_all_cells).toList(image_collection.size())

# Function to process the extracted data and concatenate arrays for each property
def extract_and_concatenate_from_featurecollection(fc):
    fc_dict = fc.getInfo()
    
    features = fc_dict['features']
    property_names = features[0]['properties'].keys()

    # Create a list of lists to hold property values for each feature
    all_property_values = {prop: [] for prop in property_names}

    for feature in features:
        for prop in property_names:
            # Only append values if they are not NaN
            if isinstance(feature['properties'][prop], list):
                all_property_values[prop].append(feature['properties'][prop])
            else:
                all_property_values[prop].append([feature['properties'][prop]])

    # Now concatenate arrays while ensuring they are of the same length
    concatenated_data = {}
    for prop in property_names:
        # Convert to NumPy array and filter out missing values
        valid_data = np.concatenate(all_property_values[prop])
        # Filter out NaN values, if any
        valid_data = valid_data[~np.isnan(valid_data)]
        concatenated_data[prop] = valid_data

    return concatenated_data

# Initialize a dictionary to hold the final concatenated data across all FeatureCollections, including 'month'
final_concatenated_data = {}
years_list = []
months_list = []

# Iterate over each FeatureCollection in the extracted data list
for i in range(image_collection.size().getInfo()):
    # Get the current image
    image = ee.Image(image_collection.toList(image_collection.size()).get(i))
    print(f'Extracting image: {i + 1}')

    # Extract the 'year' and 'month' properties from the image
    year = image.get('year').getInfo()
    month = image.get('month').getInfo()
    
    # Get the current FeatureCollection
    fc = ee.FeatureCollection(extracted_data_list.get(i))

    # Extract and concatenate the data for this FeatureCollection
    concatenated_data = extract_and_concatenate_from_featurecollection(fc)

    # Update the final concatenated data and store the 'year' and 'month' value for each image
    years_list.append(years)
    months_list.append(month)
    
    for prop, array in concatenated_data.items():
        if prop not in final_concatenated_data:
            final_concatenated_data[prop] = array
        else:
            final_concatenated_data[prop] = np.concatenate((final_concatenated_data[prop], array))

def find_array_lengths(data_dict):
    array_lengths = {}
    
    for key, value in data_dict.items():
        if hasattr(value, '__len__'):  # Check if the value has a length (e.g., an array)
            array_lengths[key] = len(value)
    
    return array_lengths

array_lengths = find_array_lengths(final_concatenated_data)
print(array_lengths)

# Create a DataFrame from the dictionary
df = pd.DataFrame.from_dict(final_concatenated_data)
df = df[['build_up_index', 'drought_code', 'duff_moisture_code', 'fine_fuel_moisture_code', 'fire_weather_index', 'initial_fire_spread_index', 'latitude', 'longitude']]
df['month'] = np.repeat(months_list, len(df) // len(months_list))
df.set_index('month', inplace=True)

print(df)

# Export to CSV
# Specify the CSV file path
csv_file_path = 'cems_0114.csv' # <-- Edit as necessary
df_reset = df.reset_index()

# Export the DataFrame to CSV
df_reset.to_csv(csv_file_path, index=False)