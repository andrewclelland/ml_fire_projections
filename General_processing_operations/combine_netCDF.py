"""
Code to combine two netCDF files.
"""
import xarray as xr
import numpy as np

# Open the two NetCDF files - edit location as necessary
ds1 = xr.open_dataset("/gws/nopw/j04/bas_climate/users/clelland/model/output_access_north/output_access_north_ssp126_2025_2100.nc")
ds2 = xr.open_dataset("/gws/nopw/j04/bas_climate/users/clelland/model/output_access_south/output_access_south_ssp126_2025_2100.nc")

# Ensure both datasets have the same grid and time steps
ds1_join, ds2_join = xr.align(ds1, ds2, join="outer")

# Replace NaNs with very low values so that np.fmax can handle them
combined_ds = xr.apply_ufunc(np.fmax, ds1_join, ds2_join, dask="allowed")

# Copy global attributes (metadata) from the first dataset
combined_ds.attrs = ds1.attrs

# Copy variable attributes (units, long name, etc.) from ds1
for var in ds1.variables:
    if var in combined_ds.variables:  # Ensure variable exists in the combined dataset
        combined_ds[var].attrs = ds1[var].attrs

# Specify zlib compression for the 'predictions' variable
compression_settings = {"predictions": {"zlib": True, "complevel": 9}}

# Save to a new NetCDF file - edit location as necessary
combined_ds.to_netcdf("/gws/nopw/j04/bas_climate/users/clelland/model/output_access_combined/output_access_combined_ssp126_2025_2100.nc", encoding=compression_settings)