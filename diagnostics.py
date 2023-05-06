import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
from sklearn.metrics import f1_score
import subprocess
import sys
from sklearn.preprocessing import LabelEncoder

import requests
import pkg_resources
import re


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')

##################Function to get model predictions
def model_predictions():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    test_data = pd.read_csv(test_data_path)

    # Preprocess categorical features using LabelEncoder
    label_encoders = {}
    categorical_columns = test_data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        test_data[column] = label_encoders[column].fit_transform(test_data[column])

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    predictions = model.predict(X_test)

    return list(predictions)

##################Function to get summary statistics
def dataframe_summary():
    data = pd.read_csv(dataset_csv_path)
    summary_statistics = data.describe().to_dict()

    return summary_statistics

def missing_values_percentage():
    data = pd.read_csv(dataset_csv_path)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    missing_values_percent = data[numeric_columns].isna().mean() * 100

    return dict(missing_values_percent)

##################Function to get timings
def execution_time():
    ingestion_time = timeit.timeit('os.system("python ingestion.py")', setup='import os', number=1)
    training_time = timeit.timeit('os.system("python training.py")', setup='import os', number=1)

    return [ingestion_time, training_time]

##################Function to check dependencies
# def outdated_packages_list():
#     output = subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--outdated'])
#     outdated_packages = output.decode('utf-8').split('\n')[2:-1]

#     package_list = []
#     for package in outdated_packages:
#         package_info = package.split()
#         package_list.append(f"{package_info[0]}: {package_info[1]} -> {package_info[3]}")

#     return package_list

# def outdated_packages_list():
#     # Read requirements.txt
#     with open("requirements.txt", "r") as req_file:
#         required_packages = req_file.read().splitlines()

#     # Get currently installed packages
#     installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

#     outdated_packages = []

#     for package in required_packages:
#         package_name = re.sub(r'[<=>].*', '', package)  # Extract package name
#         installed_version = installed_packages.get(package_name, "Not Installed")
#         latest_version = getoutput(f"pip search {package_name}").split(" ")[1]

#         if latest_version != installed_version:
#             outdated_packages.append({
#                 "package_name": package_name,
#                 "installed_version": installed_version,
#                 "latest_version": latest_version
#             })

#     return outdated_packages

def get_latest_version(package_name):
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    if response.status_code == 200:
        return response.json()["info"]["version"]
    else:
        return None

def outdated_packages_list():
    # Read requirements.txt
    with open("requirements.txt", "r") as req_file:
        required_packages = req_file.read().splitlines()

    # Get currently installed packages
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    outdated_packages = []

    for package in required_packages:
        package_name = re.sub(r'[<=>].*', '', package)  # Extract package name
        installed_version = installed_packages.get(package_name, "Not Installed")
        latest_version = get_latest_version(package_name)

        if latest_version and latest_version != installed_version:
            outdated_packages.append({
                "package_name": package_name,
                "installed_version": installed_version,
                "latest_version": latest_version
            })

    return outdated_packages

if __name__ == '__main__':
    print("Model Predictions:")
    print(model_predictions())
    print("\nDataframe Summary:")
    print(dataframe_summary())
    print("\nExecution Time (Ingestion, Training):")
    print(execution_time())
    print("\nMissing Values in percent:")
    print(missing_values_percentage())
    print("\nOutdated Packages:")
    print(outdated_packages_list())





    
