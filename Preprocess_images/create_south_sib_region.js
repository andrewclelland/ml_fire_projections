/**
JavaScript script using the Google Earth Engine Code Editor to create the final south Siberia region.
*/
var table = ee.FeatureCollection('users/andyc97/Shapefiles/Soja_ecoregions_Dissolved'); // <-- Use previously created dissolved ecoregion
var table2 = ee.FeatureCollection('users/andyc97/model_shapefiles/final_north'); // <-- Used to check for no overlap with northern region
var table3 = ee.FeatureCollection('users/andyc97/model_shapefiles/final_shapefile'); // <-- Used to show whole, previously-created shapefile, again to ensure new region is within boundaries

var soja = table.filter(
  ee.Filter.inList('ecoregion', ['Mixed Forest', 'Forest Steppe', 'Southern Taiga',
  'Montane Sub-Boreal', 'Montane Boreal'])
);
Map.addLayer(soja, {}, 'Soja');

// Define a list of unique colors (one for each ecoregion)
var colors = [
  'red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta',
  'brown', 'pink', 'lime', 'teal', 'navy', 'gray', 'black', 'gold', 'violet', 'olive'
];

// Get the unique ecoregions
var uniqueEcoregions = table.aggregate_array('ecoregion').distinct();

// Create a dictionary to map ecoregions to colors
var ecoregionColorDict = ee.Dictionary.fromLists(uniqueEcoregions, colors);

// Map over the FeatureCollection to assign colors based on the ecoregion
var coloredFeatures = soja.map(function(feature) {
  var ecoregion = feature.get('ecoregion');  // Get the ecoregion of the feature
  var color = ecoregionColorDict.get(ecoregion);  // Look up the color for this ecoregion
  return feature.set('style', { color: color, width: 1 });  // Add styling information
});

// Style the FeatureCollection using the 'style' property
var styledFeatureCollection = coloredFeatures.style({ styleProperty: 'style' });

// Add the styled FeatureCollection to the map
Map.addLayer(styledFeatureCollection, {}, 'Ecoregions');

// Print the list to the console
print('Unique Ecoregion Categories:', uniqueEcoregions);

Map.addLayer(table2.draw("red"), {}, 'North');

// Check for whole extent within ABZ
var ecoRegions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017');

// Filter features where the property equals 6 or 11
var filteredFeatures = ecoRegions.filter(
  ee.Filter.or(
    ee.Filter.eq('BIOME_NUM', 6),
    ee.Filter.eq('BIOME_NUM', 11)
  )
);

Map.addLayer(filteredFeatures, {}, 'Dinerstein');

// Compute the intersection of the FeatureCollections
var intersection = soja.map(function(feature1) {
  return table3.map(function(feature2) {
    return feature1.intersection(feature2.geometry(), ee.ErrorMargin(1));
  });
}).flatten(); // Flatten the nested FeatureCollection

// Reduce the intersection to a single geometry (union of all features)
var intersectionGeometry = intersection
  .geometry()
  .dissolve(); // Dissolve all features into a single outer geometry
  
Map.addLayer(ee.Feature(intersectionGeometry), {color: 'red'}, 'Outer Intersection Geometry');
