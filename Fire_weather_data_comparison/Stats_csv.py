"""
Code to find correlations/biases/rmses/variance ratios between processed fire weather csv files (either CEMS, ACCESS or MRI).

VARIABLE_x and VARIABLE_y will be in the order you imported them.

Ensure this file is in the same folder as `Import_csv.py` and `Merge_csv.py`.
"""
import pandas as pd
from Import_csv import *
from Merge_csv import *
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Get the unique years in the dataframe
unique_years = df_merged['year'].unique()

# Iterate over each year
for year in unique_years:
    print(f'\nYear: {year}')
    
    # Filter the dataframe for the current year
    df_year = df_merged[df_merged['year'] == year]
    
    # Correlation
    buicorr = df_year['BUI_x'].corr(df_year['BUI_y'])
    dccorr = df_year['DC_x'].corr(df_year['DC_y'])
    dmccorr = df_year['DMC_x'].corr(df_year['DMC_y'])
    ffmccorr = df_year['FFMC_x'].corr(df_year['FFMC_y'])
    fwicorr = df_year['FWI_x'].corr(df_year['FWI_y'])
    isicorr = df_year['ISI_x'].corr(df_year['ISI_y'])
    
    print(f'BUI corr: {buicorr}')
    print(f'DC corr: {dccorr}')
    print(f'DMC corr: {dmccorr}')
    print(f'FFMC corr: {ffmccorr}')
    print(f'FWI corr: {fwicorr}')
    print(f'ISI corr: {isicorr}')
    print('--------------------------------------')

    # Bias
    df_year['buibias'] = df_year['BUI_x'] - df_year['BUI_y']
    buibias = df_year['buibias'].mean()
    
    df_year['dcbias'] = df_year['DC_x'] - df_year['DC_y']
    dcbias = df_year['dcbias'].mean()
    
    df_year['dmcbias'] = df_year['DMC_x'] - df_year['DMC_y']
    dmcbias = df_year['dmcbias'].mean()
    
    df_year['ffmcbias'] = df_year['FFMC_x'] - df_year['FFMC_y']
    ffmcbias = df_year['ffmcbias'].mean()
    
    df_year['fwibias'] = df_year['FWI_x'] - df_year['FWI_y']
    fwibias = df_year['fwibias'].mean()
    
    df_year['isibias'] = df_year['ISI_x'] - df_year['ISI_y']
    isibias = df_year['isibias'].mean()
    
    print(f'BUI bias: {buibias}')
    print(f'DC bias: {dcbias}')
    print(f'DMC bias: {dmcbias}')
    print(f'FFMC bias: {ffmcbias}')
    print(f'FWI bias: {fwibias}')
    print(f'ISI bias: {isibias}')
    print('--------------------------------------')

    # RMSE
    df_mergedbui = df_year[df_year['BUI_y'].notna()]
    df_mergeddc = df_year[df_year['DC_y'].notna()]
    df_mergeddmc = df_year[df_year['DMC_y'].notna()]
    df_mergedffmc = df_year[df_year['FFMC_y'].notna()]
    df_mergedfwi = df_year[df_year['FWI_y'].notna()]
    df_mergedisi = df_year[df_year['ISI_y'].notna()]
    df_mergedbui = df_mergedbui[df_mergedbui['BUI_x'].notna()]
    df_mergeddc = df_mergeddc[df_mergeddc['DC_x'].notna()]
    df_mergeddmc = df_mergeddmc[df_mergeddmc['DMC_x'].notna()]
    df_mergedffmc = df_mergedffmc[df_mergedffmc['FFMC_x'].notna()]
    df_mergedfwi = df_mergedfwi[df_mergedfwi['FWI_x'].notna()]
    df_mergedisi = df_mergedisi[df_mergedisi['ISI_x'].notna()]
    
    if not df_mergedbui.empty:
        buirmse = np.sqrt(mean_squared_error(df_mergedbui['BUI_x'], df_mergedbui['BUI_y']))
        print(f'BUI rmse: {buirmse}')
    if not df_mergeddc.empty:
        dcrmse = np.sqrt(mean_squared_error(df_mergeddc['DC_x'], df_mergeddc['DC_y']))
        print(f'DC rmse: {dcrmse}')
    if not df_mergeddmc.empty:
        dmcrmse = np.sqrt(mean_squared_error(df_mergeddmc['DMC_x'], df_mergeddmc['DMC_y']))
        print(f'DMC rmse: {dmcrmse}')
    if not df_mergedffmc.empty:
        ffmcrmse = np.sqrt(mean_squared_error(df_mergedffmc['FFMC_x'], df_mergedffmc['FFMC_y']))
        print(f'FFMC rmse: {ffmcrmse}')
    if not df_mergedfwi.empty:
        fwirmse = np.sqrt(mean_squared_error(df_mergedfwi['FWI_x'], df_mergedfwi['FWI_y']))
        print(f'FWI rmse: {fwirmse}')
    if not df_mergedisi.empty:
        isirmse = np.sqrt(mean_squared_error(df_mergedisi['ISI_x'], df_mergedisi['ISI_y']))
        print(f'ISI rmse: {isirmse}')
    print('--------------------------------------')

        # Variance Ratio
    if df_year['BUI_y'].var() != 0:
        buivr = df_year['BUI_x'].var() / df_year['BUI_y'].var()
        print(f'BUI vr: {buivr}')
    if df_year['DC_y'].var() != 0:
        dcvr = df_year['DC_x'].var() / df_year['DC_y'].var()
        print(f'DC vr: {dcvr}')
    if df_year['DMC_y'].var() != 0:
        dmcvr = df_year['DMC_x'].var() / df_year['DMC_y'].var()
        print(f'DMC vr: {dmcvr}')
    if df_year['FFMC_y'].var() != 0:
        ffmcvr = df_year['FFMC_x'].var() / df_year['FFMC_y'].var()
        print(f'FFMC vr: {ffmcvr}')
    if df_year['FWI_y'].var() != 0:
        fwivr = df_year['FWI_x'].var() / df_year['FWI_y'].var()
        print(f'FWI vr: {fwivr}')
    if df_year['ISI_y'].var() != 0:
        isivr = df_year['ISI_x'].var() / df_year['ISI_y'].var()
        print(f'ISI vr: {isivr}')