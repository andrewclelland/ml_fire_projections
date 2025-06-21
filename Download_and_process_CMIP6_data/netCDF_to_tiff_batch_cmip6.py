"""
Code to convert the downloaded netCDF files to GeoTIFF format.
"""
import rasterio
from rasterio.transform import from_origin
import pyproj
import xarray as xr
import numpy as np
import pandas as pd
import time
import warnings

# Hide FutureWarning warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Record the start time
start_time = time.time()

# Define the scenarios and years
variables = ["hurs", "huss", "pr", "rlds", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]
#scenarios = ["historical"]
#years = range(2001, 2015)  # From 2001 to 2014

scenarios = ["ssp126", "ssp245", "ssp370"]
years = range(2015, 2101)  # From 2015 to 2100

# Define a function to save DataFrame to a TIFF file
def save_df_to_tiff(df, filename, lat_col='lat', lon_col='lon', data_cols=None, resolution=0.25):
    """
    Save DataFrame to a TIFF file.
    
    Parameters:
    - df: pandas DataFrame containing 'lat', 'lon', and data columns.
    - filename: path to the output TIFF file.
    - lat_col: name of the latitude column.
    - lon_col: name of the longitude column.
    - data_cols: list of columns to be written to the TIFF file.
    - resolution: spatial resolution (size of each pixel in degrees).
    """
    if data_cols is None:
        data_cols = df.columns.difference([lat_col, lon_col])
    
    # Extract unique latitudes and longitudes
    lats = np.sort(df[lat_col].unique())[::-1]  # Sort latitudes in descending order
    lons = np.sort(df[lon_col].unique())
    
    lat_index = {lat: i for i, lat in enumerate(lats)}
    lon_index = {lon: i for i, lon in enumerate(lons)}
    
    # Initialize arrays for each data column
    data_arrays = {col: np.full((len(lats), len(lons)), np.nan) for col in data_cols}
    
    for _, row in df.iterrows():
        lat_idx = lat_index[row[lat_col]]
        lon_idx = lon_index[row[lon_col]]
        for col in data_cols:
            data_arrays[col][lat_idx, lon_idx] = row[col]
    
    # Define the transform and metadata
    transform = from_origin(lons.min(), lats.max(), resolution, resolution)
    
    with rasterio.open(filename, 'w', driver='GTiff',
                       height=data_arrays[data_cols[0]].shape[0],
                       width=data_arrays[data_cols[0]].shape[1],
                       count=len(data_cols),
                       dtype='float32',
                       crs=pyproj.CRS('EPSG:4326'),
                       transform=transform) as dst:
        for i, col in enumerate(data_cols, start=1):
            dst.write(data_arrays[col], i)
    
    print(f"Saved TIFF file: {filename}")

# Process each scenario and year
for variable in variables:
    for scenario in scenarios:
        for year in years:
            netcdf_path = f"/gws/nopw/j04/bas_climate/users/clelland/CMIP6/{variable}/{variable}_day_ACCESS-CM2_{scenario}_r1i1p1f1_gn_{year}.nc" # <-- Edit local location of netCDF files
            try:
                # Open the dataset and convert to DataFrame
                ds = xr.open_dataset(netcdf_path)
                ds = ds.resample(time='1M').mean()
                df = ds.to_dataframe().reset_index()

                # Convert the 'time' column to datetime format
                df['month'] = df['time'].apply(lambda x: x.month)
                df.set_index('month', inplace=True)
                df.drop('time', axis=1, inplace=True)

                # Apply the function to the DataFrame
                df['lon'] = df['lon'].apply(lambda x: x if x <= 180 else x - 360)

                # Remove unnecessary data to reduce file size - specific for Arctic-boreal region
                filtered_df = df[((df['lon'] > 130) & (df['lat'] >= 42)) | 
                ((100 < df['lon']) & (df['lon'] <= 130) & (df['lat'] >= 48)) | 
                ((54 < df['lon']) & (df['lon'] <= 100) & (df['lat'] >= 50)) | 
                ((3 < df['lon']) & (df['lon'] <= 54) & (df['lat'] >= 55)) | 
                ((-20 < df['lon']) & (df['lon'] <= 3) & (df['lat'] >= 59)) | 
                ((-95 < df['lon']) & (df['lon'] <= -20) & (df['lat'] >= 45)) | 
                ((df['lon'] <= -95) & (df['lat'] >= 49))]

                # Specify the columns to check for NaN values
                columns_to_check = [f'{variable}']

                # Drop rows where all specified columns are NaN
                df_cleaned = filtered_df.dropna(subset=columns_to_check, how='all')

                # Group by the month index and create separate DataFrames
                grouped = df_cleaned.groupby(level='month')
                month_dfs = {month: group for month, group in grouped}

                # Fill missing longitude values with NaN
                for i in range(1, 13):  # Loop through each month's dataframe
                    if i in month_dfs:  # Check if month data exists
                        df = month_dfs[i]
                    
                        # Find unique longitude values in the current dataframe
                        unique_lon_values = df['lon'].unique()

                        # Generate the expected range of longitude values from -179.875 to 179.875 with 0.250 steps
                        expected_lon_values = np.arange(-179.875, 180, 0.250)

                        # Find missing values
                        missing_lon_values = set(expected_lon_values) - set(unique_lon_values)

                        # If there are missing longitude values, add new rows with lat=0 and NaN for other variables
                        if missing_lon_values:
                            # Create a DataFrame for the missing values
                            missing_rows = pd.DataFrame({
                                'lon': list(missing_lon_values),
                                'lat': 0,  # Set latitude to 0
                            })
                        
                            # Add NaNs for all other columns in the dataframe
                            for col in df.columns:
                                if col not in ['lon', 'lat']:
                                    missing_rows[col] = np.nan
                        
                            # Append the missing rows to the current dataframe
                            month_dfs[i] = pd.concat([df, missing_rows], ignore_index=True)

                # Process each month DataFrame and save as TIFF
                for month, month_df in month_dfs.items():
                    # Define the output TIFF file name
                    filename = f"/gws/nopw/j04/bas_climate/users/clelland/CMIP6/{variable}/ACCESS-CM2_{variable}_{scenario}_{year}_{month}.tif" # <-- Edit local location of output .tif files
                
                    # Save DataFrame to TIFF file
                    save_df_to_tiff(month_df, filename)
        
            except FileNotFoundError:
                print(f"File not found: {netcdf_path}")
            except Exception as e:
                print(f"Error processing {netcdf_path}: {e}")

# Calculate the time difference
end_time = time.time()
time_difference_seconds = end_time - start_time
time_difference_hours = time_difference_seconds / 3600  # Convert seconds to hours
print(f"Time taken: {time_difference_hours:.2f} hours")