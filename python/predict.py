from azureml.core import Workspace, Dataset, Datastore
from azure.storage.fileshare import ShareFileClient

import os
from os.path import join
import json
from joblib import dump, load
import sys

import tensorflow as tf
import tensorflow.keras as keras

import rasterio as rio
import numpy as np

import argparse

#import custom modules
sys.path.append(join(os.getcwd(), 'utils'))

from LandcoverProg import LandcoverProg
from ModelSessionKerasExample import KerasDensePredictor
from DataLoader import InMemoryRaster

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, help='The path to the config file')
parser.add_argument('-m', type=str, help='The path to the models json file')
# parser.add_argument('-i', type=str, help='The path to the image to classify')

args = parser.parse_args()

with open(args.m, 'r') as f:
    models = json.load(f)

# LandcoverProg class takes path to config file
classifier = LandcoverProg(config = args.c, models = args.m)

ds = Datastore.get(
    workspace = classifier._workspace,
    datastore_name = classifier._fileshare['datastore_name'])
    
root_path = (ds, '')
share_files = Dataset.File.from_files(path = [root_path])

print('getting data from fileShare')
# get our files from sharepoint
with share_files.mount('data') as mount:
    mount_point = mount.mount_point
    img_path = join(mount_point, classifier._data['img_path'])
    underlying_model_file_path, augment_model_file_path, augment_x_train_file_path, augment_y_train_file_path = classifier.load_model_files_json()
    augment_model = load(join(mount_point, augment_model_file_path))
    augment_x_train = np.load(join(mount_point, augment_x_train_file_path), allow_pickle = True)
    augment_y_train = np.load(join(mount_point, augment_y_train_file_path), allow_pickle = True)
    underlying_model = keras.models.load_model(
        join(mount_point, underlying_model_file_path),
        compile=False,
        custom_objects={
            "jaccard_loss":keras.metrics.mean_squared_error, 
            "loss":keras.metrics.mean_squared_error
        }
    )

    extent = classifier.define_extent(img_path)
    input_raster = classifier.get_data_from_extent(img_path, extent)

print('running predictions')
try:
    predictor = KerasDensePredictor(underlying_model, augment_model, augment_x_train, augment_y_train)
    output = predictor.run(input_raster.data, True)
    assert input_raster.shape[0] == output.shape[0] and input_raster.shape[1] == output.shape[
        1], "ModelSession must return an np.ndarray with the same height and width as the input"

    output_raster = InMemoryRaster(
        output,
        input_raster.crs,
        input_raster.transform,
        input_raster.bounds)    

    class_list = classifier.load_classes_json()

    color_list = classifier.load_colors_json()

    output_hard, output_raster, img_hard = classifier.image_post_processing(
        input_raster,
        output_raster,
        color_list)

    print('Writing to temporary local .tif file')
    file_path = classifier.write_to_geotiff('./tmp.tif', output_hard, output_raster, img_hard)
    print('Moving file to fileshare')
    classifier.move_file_to_fileshare(file_path)
finally:
    print('finished')