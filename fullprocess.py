

import training
import scoring
import deployment
import diagnostics
import reporting

import subprocess

import ingestion

import json
import os

import pickle
from sklearn.preprocessing import LabelEncoder

from scoring import score_model
import pandas as pd
from sklearn.metrics import f1_score

import sys


##################Check and read new data
#first, read ingestedfiles.txt

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

def check_and_read_new_data():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Read ingestedfiles.txt
    prod_deployment_path = config['prod_deployment_path']
    ingested_files_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')

    with open(ingested_files_path, 'r') as f:
        ingested_files = f.read().splitlines()

    print(ingested_files)

    # Check for new files in the input_folder_path
    input_folder_path = config['input_folder_path']
    all_files = os.listdir(input_folder_path)

    new_files = [file for file in all_files if file not in ingested_files and file.endswith('.csv')]

    if not new_files:
        print("No new files found. Ending the process.")
        return False

    # If new data is found, proceed to the next step (checking for model drift)
    print(f"New files found and ingested: {', '.join(new_files)}")

    return new_files


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

def check_for_model_drift(new_files, config):
    # Read the score from latestscore.txt
    prod_deployment_path = config['prod_deployment_path']
    latest_score_file = os.path.join(prod_deployment_path, 'latestscore.txt')

    with open(latest_score_file, 'r') as f:
        latest_score = float(f.read().strip())

    # Load the trained model
    model_file = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Make predictions on the new data
    input_folder_path = config['input_folder_path']
    new_data = []
    for file in new_files:
        file_path = os.path.join(input_folder_path, file)
        data = pd.read_csv(file_path)
        new_data.append(data)

    new_data = pd.concat(new_data, axis=0)


    test_data = new_data
    # Preprocess the data by encoding categorical features
    le = LabelEncoder()
    test_data['corporation'] = le.fit_transform(test_data['corporation'])

    # Define the features (X_test) and target (y_test)
    X_test = test_data.drop('exited', axis=1)
    y_test = test_data['exited']

    predictions = model.predict(X_test)

    # Get a score for the new predictions
     # Calculate F1 score
    f1_new = f1_score(y_test, predictions)
    #new_score = score_model(predictions, new_data)

    # Check for model drift
    model_drift = f1_new > latest_score

    print(f"latest_score: {latest_score}")
    print(f"f1_new: {f1_new}")

    return model_drift, latest_score, f1_new


def detect_model_drift():
    # Check and read new data
    new_files = check_and_read_new_data()

    # Deciding whether to proceed (first time)
    if not new_files:
        print("No new files found. Ending the process.")
        return

    # If new data is found, proceed to check for model drift
    print(f"Detecting model drift for new data files:{', '.join(new_files)}")

    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    model_drift, latest_score, f1_new = check_for_model_drift(new_files, config)




    if model_drift == True:
        print(f"Model drift detected. New score: {f1_new:.4f}")
        print('Continue to train model with new data')
        training.train_model()
        print('Re-deploying newly trained model')
        deployment.store_model_into_pickle()
        # Define the command to run the external script
        command = ["python", "apicalls.py"]

        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Check the return code (0 indicates success)
        if result.returncode == 0:
            print("The external script ran successfully.")
            print("Output:", result.stdout)
        else:
            print("The external script encountered an error.")
            print("Error:", result.stderr)

        reporting.score_model()

    else:
        print(f"No model drift detected. Old score: {latest_score:.4f}")
        return


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

if __name__ == "__main__":
    new_files = check_and_read_new_data()

    if new_files:
        detect_model_drift()
    else:
        sys.exit()







