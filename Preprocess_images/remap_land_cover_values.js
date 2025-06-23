/**
* JavaScript script to be used in Google Earth Engine's Code Editor for making broader land cover classes for the machine learning model input based on the TEM land cover map.
*/
// Load image and shapefile
var image = ee.Image('users/andyc97/model_shapefiles/TEM_LandCover_Map_V3');
var shpfile = ee.FeatureCollection('users/andyc97/model_shapefiles/final_shapefile');

// Define the mapping of old values to new values
var fromValues = [0, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 35, 50, 51, 52, 60, 61, 62, 63, 64, 65, 70, 98, 99];
var toValues = [100, 110, 110, 120, 120, 110, 120, 120, 120, 120, 130, 110, 110, 120, 120, 120, 110, 120, 110, 120, 140, 140, 140, 150, 160, 170, 170, 170, 180, 180, 180, 180, 180, 160, 190, 160, 190];

// Replace values using remap
var groupedImage = image.remap(fromValues, toValues);

// Define visualization parameters (optional)
var visParams = {
  min: 100,
  max: 190,
  palette: ['red', 'blue'] // Customize colors
};

// Add the grouped image to the map
Map.addLayer(groupedImage, visParams, 'Grouped Values');

// Export the resultant image as an asset
Export.image.toAsset({
  image: groupedImage,
  description: 'GroupedImage',
  assetId: 'users/andyc97/model_shapefiles/TEM_LandCover_Map_V3_Grouped1',
  region: shpfile.geometry(),
  scale: 1000,
  maxPixels: 1e8
});
