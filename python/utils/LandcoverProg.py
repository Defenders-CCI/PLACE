from joblib import dump, load
#import re
import azureml.core
from azureml.core import Experiment, Workspace, Dataset, Datastore, ScriptRunConfig
from azure.storage.fileshare import ShareFileClient
from os.path import join
#import matplotlib
#import matplotlib.pyplot as plt
#import glob
import fiona
import shapely
#import gdal
#import osr
import numpy as np
import os
import rasterio
from DataLoader import DataLoaderCustom, InMemoryRaster, warp_data_to_3857
from Utils import setup_logging, get_random_string, class_prediction_to_img
from ModelSessionKerasExample import KerasDensePredictor
#from Models import _load_model, load_models
#from Datasets import load_datasets
#import numpy
#import tensorflow as tf
#from tensorflow import keras
#import sklearn
import json
import cv2

#from ModelSessionAbstract import ModelSession


class LandcoverProg():
    """
    A class used to represent the landcover-ai programmatic process
    ...

    Methods
    -------
    load_workspace(self)
        loads the workspace

    show_datastores(self, ws)
        prints the datastores

    load_data_share_info(self)
        loads the data share information

    register_data_share(self,account_key, ws, datastore_name, file_share_name, account_name)
        registers the data share

    get_data_share(self, ws, datastore_name)
        gets the datashare

    start_mount(self, datastore, path)
        starts mount and creates dataset mount folder

    walk_directory(self, dataset_mount_folder)
        walks through files in dataset mount folder and creates a list of file paths

    get_geometry(self, file_name)
        gets the geometry of the image 
        Note: `geometry` is expected
        to be a GeoJSON polygon (as a dictionary) in the EPSG:4326 coordinate system.

    get_data_from_geometry()
        returns the data from the input raster source

    pred_tile(self,input_raster)
        runs predictions on the input raster

    load_classes_json(self, file_name)
        loads classes from json and create a list

    load_colors_json(self, file_name)
        load colors from json and create a list

    image_post_processing(self, input_raster, output_raster, color_list)
        post processes image using predictions

    write_to_geotiff(self, file_path, output_hard, output_raster)
        writes classified image to geotiff 

    """
    def __init__(self, **kwargs):
        
        config_file = kwargs['config']
        with open(config_file, 'r') as f:
            js = json.load(f)

        models_file = kwargs['models']
        with open(models_file, 'r') as f:
            ms = json.load(f)
    
        self._modelName = js['model']
        self._model = ms[self._modelName]
        self._workspace = None
        self._data = None
        self._fileshare = None
        if 'workspace' in js.keys():
            config = js['workspace']
            ws = Workspace(subscription_id = config['subscription_id'],
               resource_group = config['resource_group'],
               workspace_name = config['workspace_name']
               )
            self._workspace = ws
            
        if 'data' in js.keys():
            self._data = js['data']
        if 'fileShare' in js.keys():
            self._fileshare = js['fileShare']
        
    @property
    def workspace(self):
        return self._workspace
        
    @workspace.setter
    def workspace(self, **kwargs):
        ws = Workspace(subscription_id = kwargs['subscription_id'],
               resource_group = kwargs['resource_group'],
               workspace_name = kwargs['workspace_name']
               )
        self._workspace = ws
        
    @property
    def fileShare(self):
        return self._fileShare
    
    @fileShare.setter
    def fileShare(self, **kwargs):
        if self._workspace:
            ds = Datastore.get(workspace = self._workspace, datastore_name = kwargs['datastore_name'])
            self._datastore = ds
        else:
            print('An Azure workspace must be associated with this object to set a datastore')
    
#    def load_workspace(self):
#        """Loads workspace from the config.json file in the current folder
#
#        Returns
#        -------
#        sets _ws attribute
#        """
#        ws = Workspace.from_config()
#
#        self._ws = ws

    def show_datastores(self):
        """Loads workspace from the config.json file in the current folder

        Parameters
        ----------
        ws
            workspace object
        """
        if self._workspace:
            # get metadata about the workspace
            print('workspace metadata', self._workspace.name, self._workspace.location, self._workspace.resource_group, sep='\t')
            # list the registered datastores
            print('workspace datastores', self._workspace.datastores)
        else:
            print('Associate an Azure workspace to get datastore info')
            
#    def load_data_share_info(self):
#        """Loads data share info from user edited datastore.json file 
#
#        Returns
#        -------
#        account_key
#            key for the account, usually a long string of letters, numbers, and special characters
#
#        datastore_name
#            name of the datastore
#
#        file_share_name
#            name of the file share
#
#        account_name
#            name of the account that holds the file share
#
#        file_path
#            file path inside of file_share
#        """
#        datastore_info = json.load(open('datastore.json', "r"))
#        account_key = datastore_info[0]["account_key"]
#        datastore_name = datastore_info[1]["datastore_name"]
#        file_share_name = datastore_info[2]["file_share_name"]
#        account_name = datastore_info[3]["account_name"]
#        file_path = datastore_info[4]["file_path"]
#
#        return account_key, datastore_name, file_share_name, account_name, file_path

#    def register_data_share(self, account_key, ws, datastore_name, file_share_name, account_name):
#        """Registers the data share using the specifications provided
#
#        Parameters
#        ----------
#        account_key
#            key for the account, usually a long string of letters, numbers, and special characters
#
#        ws
#            workspace object
#
#        datastore_name
#            name of the datastore
#
#        file_share_name
#            name of the file share
#
#        account_name
#            name of the account that holds the file share
#
#        Returns
#        -------
#        registered_datastore
#            registered data share with image data
#        """
#        registered = Datastore.register_azure_file_share(
#            account_key=account_key,
#            workspace=ws,
#            datastore_name=datastore_name,
#            file_share_name=file_share_name,
#            account_name=account_name)
#
#        return registered

# Deprecating - equivalent to datastore.setter
        
#    def get_data_share(self, datastore_name):
#        """Gets the datashare 
#
#        Parameters
#        ----------
#        ws
#            workspace object
#
#        datastore_name
#            name of the datastore
#
#        Returns
#        -------
#        datastore
#            the datastore object
#
#        """
#        # access the datashare with model checkpoint and imagery
#        datastore = Datastore.get(workspace=self._workspace, datastore_name=datastore_name)
#
#        return datastore

    # def create_dataset_mount_folder(self, dataset_mount):
    #     """Creates dataset mount folder

    #        Parameters
    #        ----------
    #        ws
    #            workspace object

    #        datastore_name
    #            name of the datastore

    #        Returns
    #        -------
    #        dataset_mount_folder
    #            the mounted dataset folder

    #        """

    #     dataset_mount_folder = dataset_mount.mount_point

    #     return dataset_mount_folder


    # def start_mount(self):
    #     """Starts the datastore mount
    
    #    Parameters
    #    ----------
    #    ws
    #        workspace object
    
    #    datastore_name
    #        name of the datastore
    
    #    Returns
    #    -------
    #    str: dataset mount point folder
    
    #    """
    #     if self._datastore:
    #        file_path = (self._datastore, self._data["file_path"])
    #        dataset = Dataset.File.from_files(path=[file_path])
    #        dataset_mount = dataset.mount()
    #        dataset_mount.start()
            
    #        return dataset_mount.mount_point
    #     else:
    #        print('An Azure datastore must be associated to mount a fileshare')

    # def stop_mount(self, dataset_mount):
    #     dataset_mount.stop()
    
    
    def walk_directory(self, dataset_mount_folder):
        """Walks through dataset mount folder and creates a list of the file paths
    
        Parameters
        ----------
        dataset_mount_folder
            the mounted dataset folder
    
        Returns
        -------
        files_list
            a list with the mounted file paths
    
        """
        import glob
        files = glob.glob(dataset_mount_folder + '/*')
        files_list = []
        for file in files:
            files_list.append(file)
    
        return files_list
    
    
    def get_geometry(self):
        """Gets the geometry of the image
        Note: `geometry` is expected
        to be a GeoJSON polygon (as a dictionary) in the EPSG:4326 coordinate system.
    
        Parameters
        ----------
        file_name
            the name of the geojson that contains the geometry of the image
    
        Returns
        -------
        geom
            the geometry of the image
    
        """
        geom = json.load(open("geom.geojson", "r"))
    
        return geom


    def get_file_path_from_file(self, files_list):
        """Gets the file path based on a given index, indexes files_list 

        Parameters
        ----------
        files_list
            the list of paths generated earlier from the dataset_mount_folder
    
        Returns
        -------
        file_path
            str of the file_path of desired img
    
        """
        file_path = None
        img_index = None
        for i, file in enumerate(files_list):
            print(i, file)
        while img_index is None:
            try:
                img_index = int(input('Please enter the index of the file you wish to run classification on\n'))
                if img_index <= len(files_list):
                    file_path = img_index
                else:
                    print('That value is out of range')
                break
            except ValueError:
                print("Oops!  That was no valid number.  Try again...")

        return file_path


    def define_extent(self, img):
        """defines the extent of an image given the image path
    
        Parameters
        ----------
        img
            the path of the image of interest
    
        Returns
        -------
        extent
            the extent of the image in the format {'xmax': -8547225, 'xmin': -8547525, 'ymax': 4709841, 'ymin': 4709541, 'crs': 'epsg:3857'}
        """
        with rasterio.open(img) as image:
            bounds = list(image.bounds)
            extent = {}
            extent['xmax'] = bounds[2]
            extent['xmin'] = bounds[0]
            extent['ymax'] = bounds[3]
            extent['ymin'] = bounds[1]
            extent['crs'] = image.crs
    

        return extent



    def extent_to_transformed_geom(self, extent, dst_crs):
        """This function takes an extent in the the format {'xmax': -8547225, 'xmin': -8547525, 'ymax': 4709841, 'ymin': 4709541, 'crs': 'epsg:3857'}
    and converts it into a GeoJSON polygon, transforming it into the coordinate system specificed by dst_crs.
    
    Args:
        extent (dict): A geographic extent formatted as a dictionary with the following keys: xmin, xmax, ymin, ymax, crs
        dst_crs (str): The desired coordinate system of the output GeoJSON polygon as a string (e.g. epsg:4326)

    Returns:
        geom (dict): A GeoJSON polygon
        """
        left, right = extent["xmin"], extent["xmax"]
        top, bottom = extent["ymax"], extent["ymin"]
        src_crs = extent["crs"]

        geom = {
        "type": "Polygon",
        "coordinates": [[(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]]
    }

        if src_crs == dst_crs: # TODO(Caleb): Check whether this comparison makes sense for CRS objects
            return geom
        else:
            return fiona.transform.transform_geom(src_crs, dst_crs, geom)
    """
    /***************************************************************************************
    *    Title: Landcover (get_data_from_extent() from DataLoader.py)
    *    Author: Microsoft (C. Robinson et al)
    *    Date: 7/14/2021
    *    Code version: ?
    *    Availability: https://github.com/microsoft/landcover
    *
    *    Official Documentation:
    *    
    *    Returns the data from the class' data source corresponding to a *buffered* version of the input 
    *    extent. Buffering is done by `self.padding` number of units (in terms of the source 
    *    coordinate system).
    *
    *    
    
        Args:
            extent (dict): A geographic extent formatted as a dictionary with the following keys: xmin, xmax, ymin, ymax, crs
    
        Returns:
            output_raster (InMemoryRaster): A raster cropped to a *buffered* version of the input extent.  
    *
    ***************************************************************************************/
    """
    def get_data_from_extent(self, img, extent):
        with rasterio.open(img, "r") as f:
            src_crs = f.crs.to_string()
            transformed_geom = self.extent_to_transformed_geom(extent, src_crs)
            transformed_geom = shapely.geometry.shape(transformed_geom)

            buffed_geom = transformed_geom.buffer(0)
            buffed_geojson = shapely.geometry.mapping(buffed_geom)

            #buffed = shapely.geometry.mapping(shapely.geometry.box(*buffed_geom.bounds))
            src_image, src_transform = rasterio.mask.mask(f, [buffed_geojson], crop=True, all_touched=True, pad=False) # NOTE: Used to buffer by geom, haven't tested this.

        src_image = np.rollaxis(src_image, 0, 3)
        return InMemoryRaster(src_image, src_crs, src_transform, buffed_geom.bounds)

    
    def load_model_files_json(self):
        """loads the model files based on provided paths in the model_files.json document
    
        Returns
        -------
        underlying_model_file_path
            the underlying unsupervised model that is applied before the Random Forest Classifier

        augment_model_file_path
            the path of the augment model (the Random Forest Classifier)

        augment_x_train_file_path
            the path of the augment x train file
 
        augment_y_train_file_path
            the path of the augment y train file

        """
        model = self._model
        underlying_model_file_path = model["underlying_model_file_path"]
        augment_model_file_path = model["augment_model_file_path"]
        augment_x_train_file_path = model["augment_x_train_file_path"]
        augment_y_train_file_path = model["augment_y_train_file_path"]
        
        return underlying_model_file_path, augment_model_file_path, augment_x_train_file_path, augment_y_train_file_path




    def pred_tile(self, input_raster, model_name):
        """runs prediction on input_raster (I made some changes to the original code)
    /***************************************************************************************
    *    Title: Landcover (server.py)
    *    Author: Microsoft (C. Robinson et al)
    *    Date: 7/14/2021
    *    Code version: ?
    *    Availability: https://github.com/microsoft/landcover
    ***************************************************************************************/
    """
        underlying_model_file_path, augment_model_file_path, augment_x_train_file_path, augment_y_train_file_path = self.load_model_files_json(model_name)
        predictor = KerasDensePredictor(underlying_model, augment_model, augment_x_train, augment_y_train)
        output = predictor.run(input_raster.data, True)
        assert input_raster.shape[0] == output.shape[0] and input_raster.shape[1] == output.shape[
            1], "ModelSession must return an np.ndarray with the same height and width as the input"
    
        return InMemoryRaster(output, input_raster.crs, input_raster.transform, input_raster.bounds)
    
    
    def load_classes_json(self):
        """loads the classes from the classes.json file provided. This is in the /tmp folder after you
        save a model checkpoint from landcover-ai tool or you can provide your own
    
        Parameters
        ----------
        file_name
            the name of the file (usually classes.json since that is what comes with the checkpoint)
    
        Returns
        -------
        class_list
            a list of all of the prediction classes
        """
        model = self._model
        classes = model['classes']
        # with open(model['classes'], "r") as f:
        #     classes = json.load(f)
        
        class_list = [cl['name'] for cl in classes]
    
        return class_list
    
    
    def load_colors_json(self):
        """loads the colors from the classes.json file provided. This is in the /tmp folder after you
        save a model checkpoint from landcover-ai tool
    
        Parameters
        ----------
        file_name
            the name of the file (usually classes.json since that is what comes with the checkpoint)
    
        Returns
        -------
        color_list
            a list of the colors assoficated with the prediction classes
    
        """
        model = self._model
        classes = model['classes']
        # with open(model['classes'], "r") as f:
        #     classes = json.load(f)
        
        color_list = [cl['color'] for cl in classes]
    
        return color_list
    
    
    def image_post_processing(self, input_raster, output_raster, color_list):
        """ processes image to prepare to write to a file
    /***************************************************************************************
    *    Title: Landcover (server.py)
    *    Author: Microsoft (C. Robinson et al)
    *    Date: 7/14/2021
    *    Code version: ?
    *    Availability: https://github.com/microsoft/landcover
    ***************************************************************************************/
    Return:
        output_hard (array): classified array image
        output_raster (inMemoryRaster): classified raster
        img_hard (array): 3D array with rgba in dimension 2
    """
        # Everything below here is from landcover's server.py document
        # returning the index with the maximum value in respect to both axes
        output_hard = output_raster.data.argmax(axis=2)
        nodata_mask = np.sum(np.isnan(input_raster.data), axis = 2) == input_raster.shape[2]
        # nodata_mask = np.sum(input_raster.data == 0, axis=2) == input_raster.shape[2]
        output_hard[nodata_mask] = 255
        class_vals, class_counts = np.unique(output_hard[~nodata_mask], return_counts=True)
        # create an 3 channel RGB image
        img_hard = class_prediction_to_img(output_raster.data, True, color_list)
        img_hard = cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGRA)
        img_hard[nodata_mask] = [0, 0, 0, 0]
        output_raster.data = img_hard
        output_raster.shape = img_hard.shape
    
        return output_hard, output_raster, img_hard
    
    
    def write_to_geotiff(self, file_path, output_hard, output_raster, img_hard):
        """writes image to geotiff (I made some changes to the original code)
    /***************************************************************************************
    *    Title: Landcover (server.py)
    *    Author: Microsoft (C. Robinson et al)
    *    Date: 7/14/2021
    *    Code version: ?
    *    Availability: https://github.com/microsoft/landcover
    ***************************************************************************************/
    """
        array = os.path.splitext(file_path)
        name = os.path.basename(array[0])
        # Everything below here is from landcover's server.py document
        new_profile = {}
        new_profile['driver'] = 'GTiff'
        new_profile['dtype'] = 'uint8'
        new_profile['compress'] = "lzw"
        new_profile['count'] = 1
        new_profile['transform'] = output_raster.transform
        new_profile['height'] = output_hard.shape[0]
        new_profile['width'] = output_hard.shape[1]
        new_profile['crs'] = output_raster.crs
        
        file_path = "%s_classified.tif" % (name)
        
        f = rasterio.open(file_path, 'w', **new_profile)
        f.write(output_hard.astype(np.uint8), 1)
        f.close()

        return file_path

    def move_file_to_fileshare(self, local_file):
        """moves the written tif file to the fileshare
    
        Parameters
        ----------
        file_path
            the path of the classified image
        """

        connection_string = self._fileshare['connection_string']
        # connection_string = "DefaultEndpointsProtocol=https;AccountName=changedetectio8527471924;AccountKey=Dku+0TqE3wzDk0vpS72stllllxRpWbSqK0qjDblVX3pSha2Qhiq2/E8wW15KcuSThZ24WGmttkSNjgIGdkBzDA==;EndpointSuffix=core.windows.net"
        share_name = self._fileshare['file_share_name']
        share_path = self._data['out_path']
        
        file_client = ShareFileClient.from_connection_string(
            conn_str= connection_string,
            share_name=share_name,
            file_path=share_path)
        
        with open(local_file, 'rb') as f:
            file_client.upload_file(f)

        os.remove(local_file)

    

        

    
    
    
