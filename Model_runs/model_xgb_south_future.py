"""
South Siberian model for making future projections under both CMIP6 models and all scenarios. Output will be saved as a netCDF file with prediction plots as png.

Edit as necessary, but keep model parameters consistent throughout.
"""
import numpy as np
import dask.array as da
import dask
from dask import delayed
import xgboost as xgb
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
import netCDF4 as nc
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Record the start time
start_time = time.time()

# Load numpy training arrays from .npy files
top_band_names = ['elevation', 'slope', 'aspect', 'land_g1', 'month', 'latitude', 'longitude_sine']
fwi_band_names = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI']
clim_band_names = ['rh', 'pr_sum', 'rlds', 'rsds', 'sfcWind', 't2m', 'mx2t', 'mn2t']

# Function to load arrays lazily with Dask and ensure 2D shape
@delayed
def load_npy_file(filepath):
    data = np.load(filepath)
    return data.reshape(-1, 1)  # Ensure each array is 2D with a single feature column

# Load training arrays - Edit as necessary
loaded_train_arrays = [
    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_south_all/training_south_0120_{band}_array.npy'), shape=(42621360, 1), dtype=np.float32)
    for band in top_band_names
] + [
    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_south_all/training_south_cems_0120_{band}_array.npy'), shape=(42621360, 1), dtype=np.float32)
    for band in fwi_band_names
] + [
    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_south_all/training_south_e5l_0120_{band}_array.npy'), shape=(42621360, 1), dtype=np.float32)
    for band in clim_band_names
]

# Concatenate along axis 1 to stack columns into X_train
X_train = da.concatenate(loaded_train_arrays, axis=1)
print('X_train shape:', X_train.shape)

# Load and ravel the training fire array
train_firecci_array = load_npy_file('/gws/nopw/j04/bas_climate/users/clelland/model/training_south_all/training_south_0120_firecci_array.npy') # <-- Edit as necessary
y_train = da.from_delayed(train_firecci_array, shape=(42621360,), dtype=np.float32).ravel()
print('y_train shape:', y_train.shape)

# Compute training arrays, ravel and ensure target labels are integers for classification
X_train, y_train = dask.compute(X_train, y_train)
y_train = y_train.ravel()
y_train = y_train.astype(int)

# Stratified sampling
burned_indices = np.where(y_train == 1)[0]
unburned_indices = np.where(y_train == 0)[0]

# Use all available indices
burned_sample_indices = np.random.choice(burned_indices, 45788, replace=False)
unburned_sample_indices = np.random.choice(unburned_indices, 42575572, replace=False)

# Combine the sampled indices
sample_indices = np.concatenate([burned_sample_indices, unburned_sample_indices])

# Create the sampled dataset
X_sampled = X_train[sample_indices]
y_sampled = y_train[sample_indices]

# Convert to XGBoost DMatrix
dtrain = xgb.DMatrix(X_sampled, label=y_sampled)

# **Custom loss function to penalize False Negatives more**
def weighted_log_loss(y_pred, dtrain):
    y_true = dtrain.get_label()
    
    # Convert raw scores to probabilities
    pred_prob = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Define weight for False Negatives
    fn_weight = 10.0  
    
    # Compute gradient and hessian
    grad = pred_prob - y_true
    grad[y_true == 1] *= fn_weight  # Increase gradient for positives
    
    hess = pred_prob * (1 - pred_prob)
    hess[y_true == 1] *= fn_weight  # Increase hessian for positives
    
    return grad, hess

# Define model parameters
params = {
    "objective": "binary:logistic",  # Standard binary classification
    "learning_rate": 0.01,
    "random_state": 42,
    "device": "cpu",
    "scale_pos_weight": 500,
    "max_depth": 6
}

# Train model with the custom loss function
gbr = xgb.train(
    params,
    dtrain,
    num_boost_round=11500,
    obj=weighted_log_loss  # Custom loss function
)

print("\nModel training complete.")

# --- **Initialise NetCDF file** ---
# Load and process latitudes and longitudes
lat_single = np.load('/home/users/clelland/Model/lat_south_single.npy')
lon_single = np.load('/home/users/clelland/Model/lon_south_single.npy')

latitudes = np.round(lat_single, decimals=2).flatten()
longitudes = np.round(lon_single, decimals=2).flatten()

# Extract unique latitudes and longitudes
latitudes_sorted = sorted(set(latitudes))
longitudes_sorted = sorted(set(longitudes))

# --- **Testing Phase** ---
models = ['access', 'mri']
models_long = ['ACCESS-CM2', 'MRI-ESM2-0']
folders = ['SSP126', 'SSP245', 'SSP370']
scenarios = ['ssp126', 'ssp245', 'ssp370']
years = range(2025, 2101)
months = range(1, 13)

constant_variables = ['elevation', 'slope', 'aspect', 'land_cover']
latlon_variables = ['latitude', 'longitude_sine']
variable_names = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI', 'hurs', 'pr', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin']

for model, model_long in zip(models, models_long):
    for folder, scenario in zip(folders, scenarios):
        # Create a single netCDF4 Dataset with multiple time steps
        fn = f'/gws/nopw/j04/bas_climate/users/clelland/model/output_{model}_south/output_{model}_south_{scenario}_2025_2100_v2.nc' # <-- Edit as necessary
        ds = nc.Dataset(fn, 'w', format='NETCDF4')
        
        # Add global metadata
        ds.description = f"NetCDF file containing predictions of burned area for the Arctic-boreal zone for the years 2025-2100 and SSP scenario {folder}, using downscaled {model_long} future climate data"
        ds.spatial_resolution = "4000 metres"
        ds.projected_coordinate_system = "WGS 1984 NSIDC EASE-Grid 2.0 North"
        ds.projection = "Lambert Azimuthal Equal Area"
        ds.wkid = "6931"
        ds.authority = "EPSG"
        ds.false_easting = "0.0"
        ds.false_northing = "0.0"
        ds.latitude_of_projection_origin = "90.0"
        ds.longitude_of_projection_origin = "0.0"
        ds.netCDF4 = "netCDF4 1.7.2, released 2024/10/22"
        
        # Define dimensions
        ds.createDimension('time', None)
        ds.createDimension('lat', len(latitudes_sorted))
        ds.createDimension('lon', len(longitudes_sorted))
        
        # Create coordinate variables
        times = ds.createVariable('time', 'f4', ('time',))
        lats = ds.createVariable('lat', 'f4', ('lat',))
        lons = ds.createVariable('lon', 'f4', ('lon',))
        preds = ds.createVariable('predictions', 'f4', ('time', 'lat', 'lon',), fill_value=np.nan, compression='zlib')
        years_var = ds.createVariable('year', 'i4', ('time',))
        months_var = ds.createVariable('month', 'i4', ('time',))
        
        # Assign values to coordinate variables
        lats[:] = latitudes_sorted
        lons[:] = longitudes_sorted
        
        # Set variable metadata
        times.units = "days since 2000-01-01 00:00:00"
        times.calendar = "standard"
        lats.long_name = "latitude"
        lats.units = "degrees_north"
        lons.long_name = "longitude"
        lons.units = "degrees_east"
        preds.long_name = "probabilistic predictions of burned area"
        preds.compress = "zlib"
        years_var.long_name = "year"
        months_var.long_name = "month"
        
        # Initialize time step index
        time_index = 0
        
        # Initialize data array with NaNs
        data_array = np.full((len(latitudes_sorted), len(longitudes_sorted)), np.nan, dtype=np.float32)
                    
        # Fill the data array with values from final_dict
        lat_index = {lat: i for i, lat in enumerate(latitudes_sorted)}
        lon_index = {lon: i for i, lon in enumerate(longitudes_sorted)}
        
        print(f"\nnetCDF file for {model} {folder} initialised")
        
        for year in years:
            for month in months:
                print("\nYear: ", year)
                print("Month: ", month)
        
                # Load testing arrays - Edit as necessary
                loaded_test_arrays = [
                    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_{model}_south/single_south_{band}_array.npy'), shape=(177589, 1), dtype=np.float32)
                    for band in constant_variables
                ] + [
                    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_{model}_south/single_south_month_{month}_array.npy'), shape=(177589, 1), dtype=np.float32)
                ] + [
                    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_{model}_south/single_south_{band}_array.npy'), shape=(177589, 1), dtype=np.float32)
                    for band in latlon_variables
                ] + [       
                    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_{model}_south/{scenario}/testing_{model}_south_{scenario}_{year}_{month}_{band}_array.npy'), shape=(177589, 1), dtype=np.float32)
                    for band in variable_names
                ]
                
                # Concatenate along axis 1 to stack columns into X_test
                X_test = da.concatenate(loaded_test_arrays, axis=1)
                print('\nX_test shape:', X_test.shape)
                
                # Compute test array and convert to DMatrix
                X_test = dask.compute(X_test)[0]
                dtest = xgb.DMatrix(X_test)
                
                # Make predictions
                y_prob = gbr.predict(dtest)
                
                # Check if either predictions or actual values contain non-zero values
                if np.any(y_prob):
        
                    # Select latitude and longitude cols - Edit as necessary
                    lat_single = np.load('/home/users/clelland/Model/lat_south_single.npy')
                    lon_single = np.load('/home/users/clelland/Model/lon_south_single.npy')
                
                    # Create a dictionary with the extracted data
                    final_dict = {
                        'latitude': lat_single,
                        'longitude': lon_single,
                        'preds': y_prob}
                    
                    # Limit final_dict entries to 2dp
                    def limit_decimal_places(data, decimals=2):
                        for key, value in data.items():
                            if isinstance(value, np.ndarray):
                                # Round the numpy array values to the specified decimal places
                                data[key] = np.round(value, decimals)
                        return data
                    final_dict = limit_decimal_places(final_dict)
        
                    # ---**Save as netCDF** ---
                    # Convert to numpy arrays
                    values = np.array(final_dict['preds']).flatten()  # Ensure 1D
        
                    # Compute time value for the current month
                    time_value = nc.date2num(datetime(year, month, 1), times.units, times.calendar)
                    times[time_index] = time_value
                    years_var[time_index] = year
                    months_var[time_index] = f'{month:02d}'
        
                    # Reset data array to NaNs for each timestep
                    data_array.fill(np.nan)
                    
                    # Fill the data array with values from final_dict
                    for lat, lon, val in zip(latitudes, longitudes, values):
                        i, j = lat_index[lat], lon_index[lon]
                        data_array[i, j] = val
                    
                    # Assign data to the variable for this time step
                    preds[time_index, :, :] = data_array
        
                    # Increment time index
                    time_index += 1
                    
                    # ---**Plot the output** ---
                    # Create a scatter plot
                    plt.figure(figsize=(20, 8))
                    ax = plt.axes(projection=ccrs.PlateCarree())
                    
                    # Add map features
                    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
                    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
                    ax.add_feature(cfeature.LAND, facecolor='white')
                    ax.add_feature(cfeature.BORDERS, linestyle=':')
                    
                    # Plot predicted values
                    sc1 = plt.scatter(final_dict['longitude'], final_dict['latitude'], c=final_dict['preds'], 
                                      cmap='viridis', s=1, marker='D', edgecolor='none', transform=ccrs.PlateCarree(), label='Predicted')
                    
                    # Add a color bar
                    cbar = plt.colorbar(sc1, orientation='vertical', pad=0.05, shrink=0.7)
                    cbar.set_label('Prediction Value')
                    
                    # Set the title and labels
                    plt.title(f'XGBoost {model_long} {folder} South Predictions for {year}_{month:02d}')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    
                    # Save the plot
                    plt.savefig(f'/home/users/clelland/Model/Final_plots/South v2/{model_long}/{folder}/preds_{model}_south_{scenario}_{year}_{month:02d}.png', dpi=300, bbox_inches='tight', transparent=True) # <-- Edit as necessary
                    
        # Close the netCDF dataset
        ds.close()
        print(f"\nSaved netCDF file for {model} {folder}")     

# Retrieve feature importances
feature_importances = gbr.get_score(importance_type='gain')
        
# Define feature names
feature_names = ['elevation', 'slope', 'aspect', 'land_cover', 'month', 'latitude', 'longitude_sine', 'BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI', 'rh', 'pr', 'rlds', 'rsds', 'sfcWind', 't2m', 'mx2t', 'mn2t']

# Ensure all features are included, defaulting to 0 if missing
importances = np.array([feature_importances.get(f"f{i}", 0.0) for i in range(len(feature_names))])

# Normalize to sum to 1 (if any feature has importance > 0)
if importances.sum() > 0:
    importances /= importances.sum()

# Print feature importances
print("\nFeature Importances:")
for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance:.4f}")

# Record time
end_time = time.time()
time_taken = (end_time - start_time) / 3600
print(f"\nTime taken: {time_taken:.2f} hours")
