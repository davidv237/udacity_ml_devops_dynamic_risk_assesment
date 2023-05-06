from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import datetime
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle():
    model_file_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    latest_score_file_path = os.path.join(config['output_model_path'], 'latestscore.txt')
    ingested_files_file_path = os.path.join(config['output_folder_path'], 'ingestedfiles.txt')

    # Get the modification time of the trained model file
    model_modification_time = os.path.getmtime(model_file_path)

    # Calculate the time difference between the current time and the modification time
    time_difference = datetime.datetime.now() - datetime.datetime.fromtimestamp(model_modification_time)

    # Set the threshold value in hours
    threshold_hours = 24

    # Check if the model is the latest within the specified threshold
    if time_difference <= datetime.timedelta(hours=threshold_hours):
        # Copy the trained model file
        shutil.copy(model_file_path,
                    os.path.join(prod_deployment_path, 'trainedmodel.pkl'))

        # Copy the latestscore.txt file
        shutil.copy(latest_score_file_path,
                    os.path.join(prod_deployment_path, 'latestscore.txt'))

        # Copy the ingestedfiles.txt file
        shutil.copy(ingested_files_file_path,
                    os.path.join(prod_deployment_path, 'ingestedfiles.txt'))
    else:
        print("The model is not the latest. Please retrain the model before deploying.")

        
        
if __name__ == '__main__':
    store_model_into_pickle()


