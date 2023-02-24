# PLACE
Predicting Landscapes to Advance Conservation Effectiveness

The PLACE project seeks to develop predictive models of anthropogenic land cover change based upon underlying geographic, regulatory, economic, and social data. As a proof of concept, we are conducting case studies in two focal areas: A 12-county area in central Texas known as 'Texas Hill Country', and Mecklenberg County, NC.
This work was funded by National Geographic and the Blumenthal Foundation.

## Python code
Based on the Microsoft Landcover AI repository, this directory contains files, modules, and scripts for fine-tuning and running land cover classification models using Sentinel-2 data and the pre-trained sentinel-demo model.

## R code
This directory will contain files, modules, and scripts for creating and fitting predictive models

## GEE code
This directory contains javascript files used to generate and export training data and satellite imagery for land cover mapping from Google Earth Engine.

## Setup
After cloning this repo, use `conda env create -f conda env create -f environment_precise_landcover_prog.yml` to create a conda virtual environment with all the necessary libraries to run landcover AI training.

## Instructions
We fine-tune the Microsoft Landcover AI model to predict 5 land cover classes, from which changes between years can be identified:
Impervious Surface (0), Forest (1), Water (2), Field (3), and Bare (4). Manually curated training data points representing each desired output class are generated, and output features from the pre-trained unsuspervised sentinel U-Net model are collected at these points. This training data is passed to a Random Forest Classifier to produce a model that can predict the desired classes. We train models for each year using training data from all counties in Texas Hill Country. This method consists of Landcover-AI models consists of 6 steps.
1. Export training data points and imagery using GEE and move to Azure FileShare/Blob Storage
2. From within the PLACE repo, activate the landcover-ai virtual environment `conda activate landcover-ai`
3. Create config files in python/jobs for each county-year
4. Update models.json to include a model for the relevant year
5. Run 'extract.py'to collect pre-trained model output features for each county `python/python3 extract.py -c {county-year config file} -m models.json`
6. Run 'fine_tune.py' for the relevant year: `python/python3 fine_tune.py -c {year config file} -m models.json`
7. Run 'predict.py' with the relevant county config file and models.json: `python/python3 predict.py -c {county-year config file} -m models.json`

The result of this process will be two npy files containing pre-trained model outputs extracted at all points for a given year, and an augement.p trained model file
# PLACE
