"""
BASIC script to create bias correction images between the 2015-2023 historic observed and projected data for each variable and each month. The corrections are then applied to the future images.

Rename image bands to match the variable names beforehand.

Edit as necessary. Ensure Google Cloud Storage permission ok.
"""
import ee
import calendar

ee.Authenticate()
ee.Initialize(project='spherical-berm-323321')  # <-- Edit as necessary

# Load ecoregions
ecoRegions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
selected_regions = ecoRegions.filter(
    ee.Filter.And(
        ee.Filter.inList('BIOME_NUM', [6, 11]),
        ee.Filter.inList('REALM', ['Nearctic', 'Palearctic'])
    )
)

# Variables
hist_vars = ['rh', 'pr_sum', 'rlds', 'rsds', 'sfcWind', 't2m', 'mx2t', 'mn2t']
future_vars = ['hurs', 'pr', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin']
rename_map = {'hurs': 'rh', 'pr': 'pr_sum', 'tas': 't2m', 'tasmax': 'mx2t', 'tasmin': 'mn2t'}
fwi_vars = ['BUI', 'DC', 'DMC', 'FFMC', 'FWI', 'ISI']
all_vars = hist_vars + fwi_vars

models = ['ACCESS-CM2', 'MRI-ESM2-0']
scenarios = ['SSP126', 'SSP245', 'SSP370']

# Helper: get seconds in month
def get_seconds_in_month(year, month):
    days = calendar.monthrange(year, month)[1]
    return days * 24 * 60 * 60

# Load historical image collection (2001–2023, excluding Dec 2023)
def load_historic_images():
    image_list = []
    for year in range(2001, 2024):
        for month in range(1, 13):
            if year == 2023 and month == 12:
                continue
            path = f'gs://clelland_fire_ml/training_e5l_cems_mcd/cems_e5l_mcd_{year}_{month}.tif'
            img = ee.Image.loadGeoTIFF(path)
            seconds = get_seconds_in_month(year, month)
            img = img.select(all_vars)
            img = img.set({'system:time_start': ee.Date.fromYMD(year, month, 1).millis()})
            # Adjust radiative bands
            img = img.addBands(
                img.select(['rlds']).divide(seconds).rename('rlds'),
                overwrite=True
            ).addBands(
                img.select(['rsds']).divide(seconds).rename('rsds'),
                overwrite=True
            )
            image_list.append(img)
    return ee.ImageCollection(image_list)

# Load future climate image collection
def load_future_climate_images(model, scenario):
    image_list = []
    for year in range(2015, 2101):
        for month in range(1, 13):
            path = f'gs://clelland_fire_ml/CMIP6_files/{model}_COG/{scenario}/{model}_{scenario.lower()}_{year}_{month}_all_cog.tif'
            img = ee.Image.loadGeoTIFF(path)
            img = img.select([v for v in future_vars if v != 'huss'])
            renamed = [rename_map.get(b, b) for b in img.bandNames().getInfo()]
            img = img.rename(renamed)
            img = img.set({'system:time_start': ee.Date.fromYMD(year, month, 1).millis()})
            image_list.append(img)
    return ee.ImageCollection(image_list)

# Load future FWI image collection
def load_future_fwi_images(model, scenario):
    image_list = []
    for year in range(2015, 2101):
        for month in range(1, 13):
            path = f'gs://clelland_fire_ml/FWI_files/{model}_COG/{scenario}/{model}_{scenario.lower()}_{year}_{month}_cog.tif'
            img = ee.Image.loadGeoTIFF(path)
            img = img.select(fwi_vars)
            img = img.set({'system:time_start': ee.Date.fromYMD(year, month, 1).millis()})
            image_list.append(img)
    return ee.ImageCollection(image_list)

# Combine future climate and FWI
def load_future_combined_image(model, scenario, year, month):
    path_clim = f'gs://clelland_fire_ml/CMIP6_files/{model}_COG/{scenario}/{model}_{scenario.lower()}_{year}_{month}_all_cog.tif'
    path_fwi = f'gs://clelland_fire_ml/FWI_files/{model}_COG/{scenario}/{model}_{scenario.lower()}_{year}_{month}_cog.tif'
    img_clim = ee.Image.loadGeoTIFF(path_clim)
    img_clim = img_clim.select([v for v in future_vars if v != 'huss'])
    renamed = [rename_map.get(b, b) for b in img_clim.bandNames().getInfo()]
    img_clim = img_clim.rename(renamed)
    img_fwi = ee.Image.loadGeoTIFF(path_fwi).select(fwi_vars)
    return img_clim.addBands(img_fwi)

# Compute per-pixel mean over overlapping years
def compute_mean_image(images, var, is_radiative=False, sec_in_month=None):
    collection = ee.ImageCollection(images)
    if is_radiative:
        collection = collection.map(lambda img: img.select(var).divide(sec_in_month).rename(var))
    else:
        collection = collection.map(lambda img: img.select(var))
    return collection.mean().rename(var)

# Create bias correction images per month
def generate_bias_images_per_region(model, scenario):
    bias_images = []
    historic_ic = load_historic_images()
    future_ic = load_future_climate_images(model, scenario).merge(load_future_fwi_images(model, scenario))

    for month in range(1, 13):
        def compute_bias_for_region(region):
            month_filter = ee.Filter.calendarRange(month, month, 'month')

            date_filter = ee.Filter.date('2015-01-01', '2023-12-01')
            hist_monthly = historic_ic.filter(ee.Filter.And(month_filter, date_filter)).map(lambda img: img.clip(region.geometry()))
            fut_monthly = future_ic.filter(ee.Filter.And(month_filter, date_filter)).map(lambda img: img.clip(region.geometry()))

            hist_mean = hist_monthly.mean().select(all_vars)
            fut_mean = fut_monthly.mean().select(all_vars)

            diff = fut_mean.subtract(hist_mean)

            band_values = [
                diff.select(var).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region.geometry(),
                    scale=4000,
                    maxPixels=1e13
                ).values().get(0)
                for var in all_vars
            ]

            bias_img = ee.Image.constant(band_values).rename(all_vars).clip(region.geometry())
            return bias_img.set({
                'month': month,
                'scenario': scenario,
                'model': model,
                'region': region.get('ECO_NAME')
            })

        monthly_biases = selected_regions.map(compute_bias_for_region)
        mosaic = ee.ImageCollection(monthly_biases).mosaic().set({
            'month': month,
            'scenario': scenario,
            'model': model
        })
        bias_images.append(mosaic)

    return ee.ImageCollection(bias_images)

# Apply bias correction to future images
def apply_bias_to_future(model, scenario, bias_ic):
    for year in range(2025, 2101):
        for month in range(1, 13):
            original = load_future_combined_image(model, scenario, year, month)
            bias = bias_ic.filter(ee.Filter.eq('month', month)).first()

            corrected = original.select(all_vars).add(bias)
            corrected = corrected.set({
                'model': model,
                'scenario': scenario,
                'year': year,
                'month': month,
                'system:time_start': ee.Date.fromYMD(year, month, 1).millis()
            })

            print(f'Corrected image ready for {model}, {scenario}, {year}-{month:02d}')
            # Add export logic here

# Activate the bias correction workflow
bias_collections = {}
for model in models:
    for scenario in scenarios:
        print(f"\n=== Processing model: {model}, scenario: {scenario} ===")
        bias_ic = generate_bias_images_per_region(model, scenario)
        bias_collections[(model, scenario)] = bias_ic
        apply_bias_to_future(model, scenario, bias_ic)