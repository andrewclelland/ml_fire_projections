"""
South Siberia model for validation/testing on the historic period. Probability of burn to be considered for analysis - currently 50%.

When loading training data - the first line (currently hashed out) is the reduced training set used for validation, whereas the second line is the full training set used for testing.

Edit as necessary, but keep model parameters consistent throughout.
"""
import numpy as np
import dask.array as da
import dask
from dask import delayed
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_curve, auc, accuracy_score, confusion_matrix, 
    classification_report, jaccard_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
import pandas as pd
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
    #da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_south/training_south_0120_{band}_array_reduced.npy'), shape=(36228156, 1), dtype=np.float32)
    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_south_all/training_south_0120_{band}_array.npy'), shape=(42621360, 1), dtype=np.float32)
    for band in top_band_names
] + [
    #da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_south/training_south_cems_0120_{band}_array_reduced.npy'), shape=(36228156, 1), dtype=np.float32)
    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_south_all/training_south_cems_0120_{band}_array.npy'), shape=(42621360, 1), dtype=np.float32)
    for band in fwi_band_names
] + [
    #da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_south/training_south_e5l_0120_{band}_array_reduced.npy'), shape=(36228156, 1), dtype=np.float32)
    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_south_all/training_south_e5l_0120_{band}_array.npy'), shape=(42621360, 1), dtype=np.float32)
    for band in clim_band_names
]

# Concatenate along axis 1 to stack columns into X_train
X_train = da.concatenate(loaded_train_arrays, axis=1)
print('X_train shape:', X_train.shape)

# Load and ravel the training fire array - Edit as necessary
#train_firecci_array = load_npy_file('/gws/nopw/j04/bas_climate/users/clelland/model/training_south/training_south_0120_firecci_array_reduced.npy')
train_firecci_array = load_npy_file('/gws/nopw/j04/bas_climate/users/clelland/model/training_south_all/training_south_0120_firecci_array.npy')
#y_train = da.from_delayed(train_firecci_array, shape=(36228156,), dtype=np.float32).ravel()
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
    obj=weighted_log_loss # Custom loss function
)

print("\nModel training complete.")

# --- **Testing Phase** ---
#years = [2008, 2015, 2020] # validation years
#months = range(1, 13)  # 1 to 12
years = [2021, 2022, 2023] # testing years
months = range(1, 13)  # 1 to 12
#years = range(2001, 2024) # entire historic period
#months = range(1, 13)  # 1 to 12

for year in years:
    for month in months:
        print("\nYear: ", year)
        print("Month: ", month)

        # Load testing arrays - Edit as necessary
        loaded_test_arrays = [
            da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_south/testing_south_{year}_{month}_{band}_array.npy'), shape=(177589, 1), dtype=np.float32)
            for band in top_band_names
        ] + [
            da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_south/testing_south_cems_{year}_{month}_{band}_array.npy'), shape=(177589, 1), dtype=np.float32)
            for band in fwi_band_names
        ] + [
            da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_south/testing_south_e5l_{year}_{month}_{band}_array.npy'), shape=(177589, 1), dtype=np.float32)
            for band in clim_band_names
        ]
        
        # Concatenate along axis 1 to stack columns into X_test
        X_test = da.concatenate(loaded_test_arrays, axis=1)
        print('\nX_test shape:', X_test.shape)
        
        # Load and ravel the testing fire array - Edit as necessary
        test_firecci_array = load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_south/testing_south_{year}_{month}_mcd_array.npy') # firecci for 2001-2020, mcd for 2021-2023
        y_test = da.from_delayed(test_firecci_array, shape=(177589,), dtype=np.float32).ravel()
        print('y_test shape:', y_test.shape)
        
        # Compute test arrays, ravel and ensure target labels are integers for classification
        X_test, y_test = dask.compute(X_test, y_test)
        y_test = y_test.ravel()
        y_test = y_test.astype(int)
        
        # Convert test data to DMatrix
        dtest = xgb.DMatrix(X_test)
        
        # Make predictions
        y_prob = gbr.predict(dtest)

        # Set prediction probability and convert to binary class
        prediction_probability = 0.5 # <-- Edit as necessary
        y_pred = (y_prob > prediction_probability).astype(int)
        
        # Ensure arrays are 1D
        y_prob = y_prob.ravel()
        y_pred = y_pred.ravel()
        y_test = y_test.ravel()
        
        # --- **Model Evaluation** ---
        # Check if either predictions or actual values contain non-zero values
        if np.any(y_test) or np.any(y_pred):
        
            # Evaluate common metrics
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            iou = jaccard_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Print metrics
            print(f"\nAccuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"IOU: {iou:.4f}")
            
            # Confusion matrix
            print("\nConfusion Matrix:")
            print(conf_matrix)
            
            # Detailed classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Compute ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Print AUC
            print(f"AUC Score: {roc_auc:.4f}")
            
            # Plot ROC Curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
            plt.title(f"XGB South ROC curve for {year}_{month:02d}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.grid()
            plt.savefig(f'/home/users/clelland/Model/xgb_south_{year}_{month:02d}_roc.png', dpi=300, bbox_inches='tight', transparent=True) # <-- Edit as necessary

            # Select latitude and longitude cols - Edit as necessary
            lat_single = np.load('/home/users/clelland/Model/lat_south_single.npy')
            lon_single = np.load('/home/users/clelland/Model/lon_south_single.npy')
            
            # Add actual data
            actual_fire = np.load(f'/gws/nopw/j04/bas_climate/users/clelland/model/testing_south/testing_south_{year}_{month}_mcd_array.npy') # <-- Edit as necessary - firecci for 2001-2020, mcd for 2021-2023
            
            # Create a dictionary with the extracted data
            final_dict = {
                'latitude': lat_single,
                'longitude': lon_single,
                'actual': actual_fire,
                'preds': y_pred,
                'probs': y_prob}
            
            # Limit final_dict entries to 3dp
            def limit_decimal_places(data, decimals=3):
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        # Round the numpy array values to the specified decimal places
                        data[key] = np.round(value, decimals)
                return data
            final_dict = limit_decimal_places(final_dict)

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
            sc1 = plt.scatter(final_dict['longitude'], final_dict['latitude'], c=final_dict['probs'], 
                              cmap='viridis', s=1, marker='D', edgecolor='none', transform=ccrs.PlateCarree(), label='Predicted probability')
            
            # Masking the actual values (only plot where actual == 1)
            mask = np.array(final_dict['actual']) == 1
            
            # Plot actual values in red, only where actual == 1
            sc2 = plt.scatter(np.array(final_dict['longitude'])[mask], np.array(final_dict['latitude'])[mask], 
                              c='red', s=1, marker='D', edgecolor='none', transform=ccrs.PlateCarree(), label='Actual', alpha=0.5)
            
            # Add a legend
            #plt.legend(loc='lower left')
            ax.set_extent([final_dict['longitude'].min() - 1, final_dict['longitude'].max() + 1, final_dict['latitude'].min() - 1, final_dict['latitude'].max() + 1], crs=ccrs.PlateCarree())

            # Add a color bar - optional
            #cbar = plt.colorbar(sc1, orientation='vertical', pad=0.05, shrink=0.7)
            #cbar.set_label('Probability Value')
            
            # Set the title and labels
            plt.title(f'Map Visualization of XGB South Predictions for {year}_{month:02d}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            
            # Show the plot
            plt.savefig(f'/home/users/clelland/Model/preds_xgb_south_{year}_{month:02d}.png', dpi=300, bbox_inches='tight', transparent=True) # <-- Edit as necessary
            
            def meters_to_degrees(grid_size_meters, lat):
                """Converts grid size from meters to degrees."""
                grid_size_degrees_lat = grid_size_meters / 111320  # Approximate
                grid_size_degrees_lon = grid_size_meters / (111320 * np.cos(np.radians(lat)))  # Adjust for longitude
                return grid_size_degrees_lat, grid_size_degrees_lon
            
            def create_grid_meters(lon_min, lon_max, lat_min, lat_max, grid_size_meters):
                """Creates a grid in meters instead of degrees."""
                mid_lat = (lat_min + lat_max) / 2
                grid_size_lat, grid_size_lon = meters_to_degrees(grid_size_meters, mid_lat)
                
                lon_bins = np.arange(lon_min, lon_max, grid_size_lon)
                lat_bins = np.arange(lat_min, lat_max, grid_size_lat)
                
                return lon_bins, lat_bins
            
            def assign_to_grid(lons, lats, lon_bins, lat_bins):
                """Assigns each point to a grid cell."""
                lon_indices = np.digitize(lons, lon_bins) - 1
                lat_indices = np.digitize(lats, lat_bins) - 1
                return lon_indices, lat_indices
            
            def compute_grid_percentages(pred_lons, pred_lats, actual_lons, actual_lats, grid_size_meters):
                """Computes the percentage of predictions and actual values within each grid cell."""
                lon_min = 59.50568279537051
                lon_max = 141.74005609794693
                lat_min = 45.53657531738281
                lat_max = 63.013065338134766
                
                lon_bins, lat_bins = create_grid_meters(lon_min, lon_max, lat_min, lat_max, grid_size_meters)
                
                pred_lon_idx, pred_lat_idx = assign_to_grid(pred_lons, pred_lats, lon_bins, lat_bins)
                actual_lon_idx, actual_lat_idx = assign_to_grid(actual_lons, actual_lats, lon_bins, lat_bins)
                
                df_pred = pd.DataFrame({'lon_idx': pred_lon_idx, 'lat_idx': pred_lat_idx})
                df_actual = pd.DataFrame({'lon_idx': actual_lon_idx, 'lat_idx': actual_lat_idx})
                
                pred_counts = df_pred.value_counts().reset_index(name='pred_count')
                actual_counts = df_actual.value_counts().reset_index(name='actual_count')
                
                grid_counts = pd.merge(pred_counts, actual_counts, on=['lon_idx', 'lat_idx'], how='outer').fillna(0)
                grid_counts['total'] = grid_counts['pred_count'] + grid_counts['actual_count']
                grid_counts['pred_percent'] = (grid_counts['pred_count'] / grid_counts['total']) * 100
                grid_counts['actual_percent'] = (grid_counts['actual_count'] / grid_counts['total']) * 100
            
                # Filter out grid cells where both percentages are 0
                grid_counts = grid_counts[(grid_counts['pred_percent'] != 0) | (grid_counts['actual_percent'] != 0)]
                
                return grid_counts, lon_bins, lat_bins
            
            # Masking the actual values (only plot where actual == 1)
            actual_mask = np.array(final_dict['actual']) == 1
            preds_mask = np.array(final_dict['preds']) > prediction_probability
            
            # Example usage
            pred_lons = np.array(final_dict['longitude'])[preds_mask].ravel()
            pred_lats = np.array(final_dict['latitude'])[preds_mask].ravel()
            actual_lons = np.array(final_dict['longitude'])[actual_mask].ravel()
            actual_lats = np.array(final_dict['latitude'])[actual_mask].ravel()
            
            grid_size_meters = 278300  # grid cell size in metres - 2.5 degrees
            grid_counts, lon_bins, lat_bins = compute_grid_percentages(pred_lons, pred_lats, actual_lons, actual_lats, grid_size_meters)
                     
            # Create a new plot with a color bar for prediction vs actual percentage difference
            plt.figure(figsize=(20, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            # Add map features
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.LAKES, facecolor='lightblue')
            ax.add_feature(cfeature.LAND, facecolor='white')
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            
            # Set the extent to zoom into the region of interest
            ax.set_extent([final_dict['longitude'].min() - 1, final_dict['longitude'].max() + 1, final_dict['latitude'].min() - 1, final_dict['latitude'].max() + 1], crs=ccrs.PlateCarree())
            
            # Compute difference between predicted and actual percentage
            grid_counts['diff'] = grid_counts['pred_percent'] - grid_counts['actual_percent']
            
            # Convert grid indices back to coordinate values
            lon_centers = lon_bins[:-1] + np.diff(lon_bins) / 2  # Grid cell centers (longitude)
            lat_centers = lat_bins[:-1] + np.diff(lat_bins) / 2  # Grid cell centers (latitude)

            # Create a 2D grid for coloring
            diff_matrix = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)  # Initialize matrix with NaNs
            for _, row in grid_counts.iterrows():
                lat_idx = min(int(row['lat_idx']), diff_matrix.shape[0] - 1)
                lon_idx = min(int(row['lon_idx']), diff_matrix.shape[1] - 1)
                diff_matrix[lat_idx, lon_idx] = row['diff']

            # Plot using pcolormesh to color each grid cell
            mesh = ax.pcolormesh(lon_bins, lat_bins, diff_matrix, cmap='coolwarm', shading='auto', vmin=-100, vmax=100, transform=ccrs.PlateCarree())

            # Add a color bar
            cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.7)
            cbar.set_label('Prediction % - Actual %')
            
            plt.title(f"Difference Between Predicted and Actual Percentages for {year}_{month:02d}")
            plt.savefig(f'/home/users/clelland/Model/xgb_south_{year}_{month:02d}_colour_grid.png', dpi=300, bbox_inches='tight', transparent=True) # <-- Edit as necessary

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
