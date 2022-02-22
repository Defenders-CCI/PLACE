// Script to export training data and imagery from GEE. This
// file uses Bastrop county, TX as an example

// Imports
var counties = ee.FeatureCollection("TIGER/2018/Counties"),
    s2 = ee.ImageCollection("COPERNICUS/S2"),
    impervious = /* color: #d63000 */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Point([-97.75906958913608, 30.22489950950131]),
            {
              "class_idx": 0,
              "system:index": "0"
            })]),
    forest = /* color: #1a9300 */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Point([-97.74706296663048, 30.23106289178293]),
            {
              "class_idx": 1,
              "system:index": "0"
            })]),
    water = /* color: #0b4a8b */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Point([-97.73839406709435, 30.252075135913966]),
            {
              "class_idx": 2,
              "system:index": "0"
            })]),
    field = 
    /* color: #ffc82d */
    /* shown: false */
    ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Point([-97.6385729763961, 30.19637869749364]),
            {
              "class_idx": 3,
              "system:index": "0"
            })]),
    bare = 
    /* color: #785213 */
    /* shown: false */
    ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Point([-97.64888760653973, 30.204981844569176]),
            {
              "class_idx": 4,
              "system:index": "0"
            })]);

//filter to specific county + state
 
var bastrop = counties.filterMetadata('NAME', 'equals', 'Bastrop').filterMetadata('STATEFP', 'equals', '48')
Map.addLayer(bastrop)

function basicQA(img){
    /*
    Mask clouds in a Sentinel-2 image using builg in quality assurance band
    Parameters:
        img (ee.Image): Sentinel-2 image with QA band
    Returns:
        ee.Image: original image masked for clouds and cirrus
    */
    //print('basicQA:', img)
    var qa = img.select('QA60').int16()
    // print('qa:', type(qa))
    // qa = img.select(['QA60']).int16()
    //print('qa:', qa.getInfo())
    // Bits 10 and 11 are clouds and cirrus, respectively.
    var cloudBitMask = 1024 // math.pow(2, 10)
    var cirrusBitMask = 2048 //math.pow(2, 11)
    // Both flags should be set to zero, indicating clear conditions.
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(qa.bitwiseAnd(cirrusBitMask).eq(0))
    var dated = img.updateMask(mask)
    //dated = img.addBands(img.metadata('system:time_start', 'date')).updateMask(mask)
    return dated
}

// filter S2 imagery to the years and county of interest

var bastrop_images = s2.filterBounds(bastrop).map(basicQA)

/*
var bastrop_s2_2016 = bastrop_images.filterDate('2016-05-01', '2016-09-01')
.median()
.clip(bastrop)
*/

/*
var bastrop_s2_2017 = bastrop_images.filterDate('2017-05-01', '2017-09-01')
.median()
.clip(bastrop)
*/

/*
var bastrop_s2_2018 = bastrop_images.filterDate('2018-05-01', '2018-09-01')
.median()
.clip(bastrop)
*/

/*
var bastrop_s2_2019 = bastrop_images.filterDate('2019-05-01', '2019-09-01')
.median()
.clip(bastrop)
*/

var bastrop_s2_2020 = bastrop_images.filterDate('2020-05-01', '2020-09-01')
.median()
.clip(bastrop)

// Visualize imagery. This can then be used to create training data points
Map.addLayer(bastrop_s2_2020, {'bands':['B4', 'B3', 'B2'], 'min':250, 'max':2500})

// merging training points into single feature collection
var classes = impervious.merge(forest).merge(water).merge(field).merge(bare)

//export training data points as geojson to Google Drive;
Export.table.toDrive({
  collection: classes,
  description: 'bastrop_2016',
  fileFormat: 'GeoJSON'
})

// exporting images as .tif files to Google Drive 
Export.image.toDrive({
  image: bastrop_s2_2016.select(['B4', 'B3', 'B2', 'B8']),
  description: 'bastrop_2016_test',
  region: geometry,
  scale: 10,
  maxPixels: 1e13
})