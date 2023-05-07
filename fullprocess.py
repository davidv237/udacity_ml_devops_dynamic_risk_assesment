

import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

import json
import os

import pickle
from scoring import score_model
import pandas as pd


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
        return

    # If new data is found, proceed to the next step (checking for model drift)
    print(f"New files found: {', '.join(new_files)}")

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
    predictions = model.predict(new_data)

    # Get a score for the new predictions
    new_score = score_model(predictions, new_data)

    # Check for model drift
    model_drift = new_score < latest_score

    return model_drift, new_score


def detect_model_drift():
    # Check and read new data
    new_files = check_and_read_new_data()

    # Deciding whether to proceed (first time)
    if not new_files:
        print("No new files found. Ending the process.")
        return

    # If new data is found, proceed to check for model drift
    print(f"New files found: {', '.join(new_files)}")

    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    model_drift, new_score = check_for_model_drift(new_files, config)

    if model_drift:
        print(f"Model drift detected. New score: {new_score:.4f}")
        # Proceed with the rest of the deployment process
    else:
        print(f"No model drift detected. New score: {new_score:.4f}")
        # No need to proceed with the rest of the deployment process


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

if __name__ == "__main__":
    new_files = check_and_read_new_data()
    if new_files:
        print(f"New files ingested: {', '.join(new_files)}")
    else:
        print("No new files found.")
    detect_model_drift()






