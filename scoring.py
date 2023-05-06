from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
model_path = os.path.join(config['output_model_path'])

# Function for scoring the model
def score_model():
    # Load test data
    test_data = pd.read_csv(test_data_path)

    # Preprocess the data by encoding categorical features
    le = LabelEncoder()
    test_data['corporation'] = le.fit_transform(test_data['corporation'])

    # Define the features (X_test) and target (y_test)
    X_test = test_data.drop('exited', axis=1)
    y_test = test_data['exited']

    # Load the trained model
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred)

    # Write F1 score to the latestscore.txt file
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as f:
        f.write(str(f1))

    return f1

# Call the function to score the model
score = score_model()
print(f"F1 score: {score}")
