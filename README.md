# ml_fire_projections
Repository for making machine learning wildfire monthly projections in the Arctic-boreal zone under different SSP scenarios from 2025-2100. Two models are used - one specific for South Siberia - to capture the different fire regime, the other covering the remainder of the region. 21 input variables are considered, containing topographic, land cover, climate and fire danger indices.

The analysis code is stored in a separate repository: [linked here](https://github.com/andrewclelland/analyse_ml_fire_projections).

Preliminary:
*  Ensure you have a Google Earth Engine account linked to a project
*  It is strongly recommended to have access to a Google Cloud Storage bucket
*  Use of a supercomputer or HPC is encouraged for running the models

Order for running scripts:
1.  `Download_and_process` the CMIP6 and FWI data and convert them to GeoTIFF format.
    *  For the CMIP6 data you will then have to `combine` the files and `rename` them if you use the latest download version.
    *  It is advised that the data be bias corrected (run the next step as to why) before being uploaded to the bucket.
    *  Finally convert the GeoTIFFs to COGs. It is then advised to copy the COGs to a Google Cloud Storage bucket.
2.  Conduct a `Fire_weather_data_comparison` to understand the biases between the NASA-downscaled fire weather data and those from CEMS.
    *  First, you need to `Process` the data to csv format - also in this Notebook you can create a grid to iterate over the whole study region, which will be useful later. The `CEMS_processing.py` file is a suitable alternative to the Jupyter Notebook and can be adapted for processing the NASA-downscaled data.
    *  Then, using the `Stats_csv.py` and `Make_plot.py` files you can analyse the data. The comparison Notebook is also an option, however this requires careful management of the csv files to avoid crashing.
    *  This code can be adapted to compare the ERA5-Land and NASA-downscaled climate data.
    *  Other than making the iterative grid, this step is optional.
3.  `Preprocess` the images for the `static_input` (i.e. topographic, land cover, lat-lon and month variables) and the historic (`hist_batch`) images. It is useful to have these stored in the Google Cloud Storage bucket.
    *  The `preprocess_future_batch.py` step is optional and may not be worth your time - the values can be extracted automatically using adjustments at the next step...
4.  Store the COG data locally for both the historic and future periods (all scenarios) as NumPy arrays by running `Bucket_to_array` files.
    *  Take extra care with the precipitation and radiation data - the CMIP6 precipitation data has to be converted to metres to match the ERA5-Land training data, whereas the ERA5-Land shortwave and longwave radiation values have to be converted to W/m2 to be consistent with the CMIP6 data.
5.  Once the data are processed in the correct form, `run` the models.
    *  First conduct a cross-validation (`crossval`) to find the optimal combination of model parameters for each model.
    *  Then validate and test the models in the `historic` period. Validation years are 2008, 2015 and 2020; testing years are 2021-2023, although the model can be tested on 2001-2023 if desired.
    *  Run the model for the `future` period across all scenarios. The output will be saved locally as netCDF files which can then be analysed (see next repository).
    *  Part of the analysis includes SHAP plots, which can be run using the `model_xgb_region_shap.py` scripts.
  
The `General_processing_operations` folder contains scripts relating to all aspects of processing files, especially NumPy arrays.
