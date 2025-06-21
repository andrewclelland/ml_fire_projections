"""
Compute SHAP values and make plots for the south Siberia model.

Edit as necessary, but keep model parameters consistent throughout.
"""
import numpy as np
import dask.array as da
import dask
from dask import delayed
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Record the start time
start_time = time.time()

# Load numpy training arrays from .npy files
top_band_names = ['elevation', 'slope', 'aspect', 'land_g1', 'month', 'latitude', 'longitude_sine'] # Exclude fire
fwi_band_names = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI']
clim_band_names = ['rh', 'pr_sum', 'rlds', 'rsds', 'sfcWind', 't2m', 'mx2t', 'mn2t']

# Function to load arrays lazily with Dask and ensure 2D shape
@delayed
def load_npy_file(filepath):
    data = np.load(filepath)
    return data.reshape(-1, 1)  # Ensure each array is 2D with a single feature column

# Load training arrays
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
train_firecci_array = load_npy_file('/gws/nopw/j04/bas_climate/users/clelland/model/training_south_all/training_south_0120_firecci_array.npy')
y_train = da.from_delayed(train_firecci_array, shape=(42621360,), dtype=np.float32).ravel()
print('y_train shape:', y_train.shape)

# Compute training arrays, ravel and ensure target labels are integers for classification
X_train, y_train = dask.compute(X_train, y_train)
y_train = y_train.ravel()
y_train = y_train.astype(int)

# Stratified sampling
burned_indices = np.where(y_train == 1)[0]
unburned_indices = np.where(y_train == 0)[0]

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

# Names have to be in the same order as you load the training variables
feature_names = ['elevation', 'slope', 'aspect', 'land cover', 'month', 'latitude', 'longitude (sine)',
                 'BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI', 'rel. humidity', 'precipitation', 'longwave rad.', 'shortwave rad.', 'wind speed', 'mean temp.', 'max temp.', 'min temp.']

X_sampled_df = pd.DataFrame(X_sampled, columns=feature_names)

# Convert trained XGBoost model to SHAP TreeExplainer
# 1 million iterations can be adjusted
sampled_df = X_sampled_df.sample(n=1000000, random_state=42)
explainer = shap.TreeExplainer(gbr)

# Compute SHAP values for the original (non-DMatrix) input data
shap_values = explainer(sampled_df)

plt.figure()  # start a new figure
shap.summary_plot(shap_values, sampled_df, show=False, max_display=len(sampled_df.columns))
plt.tight_layout()
plt.savefig('/home/users/clelland/Model/xgb_south_shap.png', dpi=300) # <-- Edit as necessary
plt.close()

# Record time
end_time = time.time()
time_taken = (end_time - start_time) / 3600
print(f"\nTime taken: {time_taken:.2f} hours")