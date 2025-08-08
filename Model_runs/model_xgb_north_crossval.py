"""
Script to conduct a cross-validation for the Northern model of several learning rate, max depth and num boosting round values to find the optimal combination of parameters.

All training data is included. Adjust as necessary.
"""
import numpy as np
import dask.array as da
import dask
from dask import delayed
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import time
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

# Load training arrays - edit as necessary
loaded_train_arrays = [
    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north_all/training_north_0123_{band}_array.npy'), shape=(332688305, 1), dtype=np.float32)
    for band in top_band_names
] + [
    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north_all/training_north_cems_0123_{band}_array.npy'), shape=(332688305, 1), dtype=np.float32)
    for band in fwi_band_names
] + [
    da.from_delayed(load_npy_file(f'/gws/nopw/j04/bas_climate/users/clelland/model/training_north_all/training_north_e5l_0123_{band}_array.npy'), shape=(332688305, 1), dtype=np.float32)
    for band in clim_band_names
]

# Concatenate along axis 1 to stack columns into X_train
X_train = da.concatenate(loaded_train_arrays, axis=1)
print('X_train shape:', X_train.shape)

# Load and ravel the training fire array - edit as necessary
train_fire_array = load_npy_file('/gws/nopw/j04/bas_climate/users/clelland/model/training_north_all/training_north_0123_fire_array.npy')
y_train = da.from_delayed(train_fire_array, shape=(332688305,), dtype=np.float32).ravel()
print('y_train shape:', y_train.shape)

# Compute training arrays, ravel and ensure target labels are integers for classification
X_train, y_train = dask.compute(X_train, y_train)
y_train = y_train.ravel()
y_train = y_train.astype(int)

# Stratified sampling
burned_indices = np.where(y_train == 1)[0]
unburned_indices = np.where(y_train == 0)[0]

# Randomly select 72m indices for unburned pixels
burned_sample_indices = np.random.choice(burned_indices, 72074, replace=False) # Use all available burned pixels
unburned_sample_indices = np.random.choice(unburned_indices, 72001926, replace=False)

# Combine the sampled indices
sample_indices = np.concatenate([burned_sample_indices, unburned_sample_indices])

# Create the sampled dataset
X_sampled = X_train[sample_indices]
y_sampled = y_train[sample_indices]

# **Custom loss function to penalize False Negatives more**
def weighted_log_loss(y_true, y_pred):    
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

# Define parameters for testing - edit as necessary
xgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'n_estimators': [10000, 12500, 15000],
}

# All these values can be changed if necessary
xgb_model = xgb.XGBClassifier(objective=weighted_log_loss, device='cpu')
xgb_grid = GridSearchCV(xgb_model, param_grid=xgb_param_grid, cv=3, n_jobs=-1, scoring='roc_auc')
xgb_grid.fit(X_sampled, y_sampled)

print("\nModel training complete.")
print(f"\nBest parameters for XGBoost: {xgb_grid.best_params_}")

# Record time
end_time = time.time()
time_taken = (end_time - start_time) / 3600
print(f"\nTime taken: {time_taken:.2f} hours")
