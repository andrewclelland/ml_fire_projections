/**
*  JavaScript Google Earth Engine Code Editor script to dissolve Soja et al (2004) ecoregions for use as part of our southern Siberia region in the ML models.
*/

// Load the feature collection - Edit as necessary
ecos = ee.FeatureCollection('users/andyc97/Shapefiles/Soja_ecosystems');

// Function to dissolve features by 'ecoregion'
var dissolveByEcoregion = function(featureCollection, attribute) {
  // Create a list of unique attribute values
  var uniqueValues = featureCollection.distinct(attribute).aggregate_array(attribute);
  
  // Map over the unique values and union features for each
  var dissolved = uniqueValues.map(function(value) {
    // Filter the collection for the current attribute value
    var filtered = featureCollection.filter(ee.Filter.eq(attribute, value));
    // Union the geometries
    var unioned = filtered.union(100); // Adjust maxError as needed
    // Add the attribute back to the resulting geometry
    return unioned.map(function(feature) {
      return feature.set(attribute, value);
    });
  });
  
  // Flatten the resulting collections into one
  return ee.FeatureCollection(dissolved).flatten();
};

// Dissolve features by 'ecoregion'
var dissolvedEcos = dissolveByEcoregion(ecos, 'ecoregion');

//remove rivers, floodlands and deltas
var dissolvedEcos = dissolvedEcos.filter(ee.Filter.neq('ecoregion', 'Rivers, Floodlands and Deltas'));

// Print and visualize
print('Dissolved Ecosystems:', dissolvedEcos);
Map.addLayer(dissolvedEcos, {}, 'Dissolved Ecoregions');
