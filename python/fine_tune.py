from azureml.core import Workspace, Dataset, Datastore
from azure.storage.fileshare import ShareClient

import os
import shutil
import glob
from os.path import join
from sys import path
import json
import joblib

import tensorflow as tf
import tensorflow.keras as keras
import sklearn.base
from sklearn.ensemble import RandomForestClassifier

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

# read annual config file
with open(args.c, 'r') as f:
    config = json.load(f)

# read model confg file
with open(args.m, 'r') as f:
    models = json.load(f)

print('successfully read config and model files')

# access relevant key values
year = config['model']
model = models[year]
fs = config['fileShare']
dat = config['data']
wksp = config['workspace']

# load workspace configuration from the config.json file in the current folder.
ws = Workspace(subscription_id = wksp["subscription_id"], workspace_name = wksp["workspace_name"], resource_group = wksp["resource_group"])

# access our registered data share containing image data in this workspace
datastore = Datastore.get(workspace = ws, datastore_name = fs['datastore_name'])

print(datastore)
root_path = (datastore, 'THC')
share_files = Dataset.File.from_files(path = [root_path])

with share_files.mount() as mount:
    mount_point = mount.mount_point
    # get numpy data
    npy_files = []
    for root, dirs, files in os.walk(mount_point): 
        for f in files:
            if '.npy' in f and f'{year}' in root:
                npy_files.append(join(root, f))

    x_files = [f for f in npy_files if 'x_train' in f]
    y_files = [f for f in npy_files if 'y_train' in f]
    xs = [np.load(f, allow_pickle = True) for f in x_files]
    ys = [np.load(f, allow_pickle = True) for f in y_files]

x_train = np.concatenate(xs, axis = 0)
y_train = np.concatenate(ys, axis = 0)
print('x train shape', x_train.shape)
print('y train shape', y_train.shape)

augment_model = RandomForestClassifier()

# remove nan values
goodx = ~np.isnan(x_train).any(axis = 1)
goody = y_train != ''
goodrows = goodx&goody

#fine-tune the model
augment_model.fit(x_train[goodrows, :], y_train[goodrows].astype(int))
score = augment_model.score(x_train[goodrows, :], y_train[goodrows].astype(int))
print("Fine-tuning accuracy: %0.4f" % (score))
augment_model_trained = True
print('saving model locally')
joblib.dump(augment_model, "./augment_model.p")
np.save('./augment_y_train.npy', y_train[goodrows].astype(int))
np.save('./augment_x_train.npy', x_train[goodrows, :])

# save fine tuned model to fileshare
share_client = ShareClient.from_connection_string(fs['connection_string'], share_name = fs['file_share_name'])

with open('augment_model.p', 'rb') as f:
    file_client = share_client.get_file_client(model['augment_model_file_path'])
    file_client.upload_file(f)

with open('augment_y_train.npy', 'rb') as f:
    file_client = share_client.get_file_client(model['augment_y_train_file_path'])
    file_client.upload_file(f)

with open('augment_x_train.npy', 'rb') as f:
    file_client = share_client.get_file_client(model['augment_x_train_file_path'])
    file_client.upload_file(f)

print('trained model moved to fileshare')
