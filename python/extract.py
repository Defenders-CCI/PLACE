# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:24:03 2021

@author: MEvans
"""
from azureml.core import Workspace, Dataset, Datastore
from azure.storage.fileshare import ShareClient

import os
import shutil
import glob
from os.path import join
from sys import path
import json

import tensorflow as tf
import tensorflow.keras as keras

import argparse

import numpy as np
import rasterio as rio
import numpy as np
import geopandas as gpd

from utils.ModelSessionKerasExample import KerasDenseFineTune

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, help='The path to the job config file')
parser.add_argument('-m', type=str, help='The path to the models json file')

args = parser.parse_args()

# read the county-year config json file
with open(args.c, 'r') as f:
    config = json.load(f)

# read the model config json file
with open(args.m, 'r') as f:
    models = json.load(f)

print('successfully read config and model files')

# extract relevant key-values
model = models[config['model']]
fs = config['fileShare']
dat = config['data']
wksp = config['workspace']

# load workspace configuration from the config.json file in the current folder.
ws = Workspace(subscription_id = wksp["subscription_id"], workspace_name = wksp["workspace_name"], resource_group = wksp["resource_group"])

# access our registered data share containing image data in this workspace
datastore = Datastore.get(workspace = ws, datastore_name = fs['datastore_name'])

print(datastore)
root_path = (datastore, '')
share_files = Dataset.File.from_files(path = [root_path])

# Connect to fileshare and read imagery, geojson, and model data
with share_files.mount('data') as mount:
    mount_point = mount.mount_point
    # get imagery data
    img_path = join(mount_point, dat['img_path'])
    with rio.open(img_path) as src:
        tile = np.moveaxis(src.read(), 0, -1)
        affine = src.transform
        crs = src.crs
    # get geojson points data
    pts_path = join(mount_point, dat['pts_path'])
    gdf = gpd.read_file(pts_path, driver = 'GeoJSON')
    # get the pre-trained model
    model_path = join(mount_point, model['underlying_model_file_path'])
    m = keras.models.load_model(model_path, compile=False, custom_objects={
        "jaccard_loss":keras.metrics.mean_squared_error, 
        "loss":keras.metrics.mean_squared_error
        })

print(tile.shape, 'image loaded into memory as nump array')

print(len(gdf), 'points loaded into memory as geodataframe')

# create a uner object to extract pre-trained model output
tuner = KerasDenseFineTune(m, gdf)

# generate output features
output = tuner.run(tile)
# sample output features at points
tuner.sample(affine)
tuner.retrain()
#save_directory = join(root, 'data')
print('saving model locally')
saved = tuner.save_state_to('.')
print('saved?', saved)

# connect to fileshare and save extracted data numpy files to relevant directories
share_client = ShareClient.from_connection_string(fs['connection_string'], share_name = fs['file_share_name'])

# TODO: can we use BytesIO to write directly to fileshare, rather than saving local copies?
# with io.BytesIO() as buffer:
#     np.save(buffer, tuner.)

with open('augment_model.p', 'rb') as f:
    file_client = share_client.get_file_client(model['augment_model_file_path'])
    file_client.upload_file(f)

with open('augment_y_train.npy', 'rb') as f:
    file_client = share_client.get_file_client(model['augment_y_train_file_path'])
    file_client.upload_file(f)

with open('augment_x_train.npy', 'rb') as f:
    file_client = share_client.get_file_client(model['augment_x_train_file_path'])
    file_client.upload_file(f)

print('model data moved to fileshare')
