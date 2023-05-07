from flask import Flask, session, jsonify, request, flash
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list
from scoring import score_model
from flask import Flask, jsonify, request
import logging
from logging.handlers import RotatingFileHandler
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['test_data_path'], 'testdata.csv')


prediction_model = None

# @app.route("/prediction", methods=['POST', 'OPTIONS'])
# def predict():
#     if request.method == 'POST':
#         # Check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return jsonify({"error": "No file part in request"}), 400

#         file = request.files['file']

#         # If the user does not select a file, the browser may submit an empty part without a file name.
#         if file.filename == '':
#             flash('No file selected')
#             return jsonify({"error": "No file selected"}), 400

#         # Save the file temporarily and get the file path
#         temp_file_path = os.path.join("temp", file.filename)
#         file.save(temp_file_path)

#         predictions = model_predictions(temp_file_path)

#         # Remove the temporary file after processing
#         os.remove(temp_file_path)

#         return jsonify(predictions=predictions)

#     return jsonify(message="Provide a file to process"), 200

@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():


    # Load the trained model
    model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)


    if request.method == 'POST':
        data = request.get_json()
        #print(data)
        file_path = data.get("file_path", "")
        #print(file_path)

        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 400

        test_data = pd.read_csv(file_path)
        # Preprocess the data by encoding categorical features
        le = LabelEncoder()
        test_data['corporation'] = le.fit_transform(test_data['corporation'])

        # Define the features (X_test) and target (y_test)
        X_test = test_data.drop('exited', axis=1)
        y_test = test_data['exited']

        predictions = model.predict(X_test)

        return jsonify(predictions=predictions.tolist())

    return jsonify(message="Provide file path"), 200




@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    f1_score = score_model()
    return jsonify(f1_score=f1_score), 200

@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    summary_stats = dataframe_summary()
    return jsonify(summary_stats=summary_stats), 200

@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    timings = execution_time()
    missing_data = dataframe_summary()  # Assuming you have implemented missing data check in dataframe_summary()
    dependency_check = outdated_packages_list()

    return jsonify(timings=timings, missing_data=missing_data, dependency_check=dependency_check), 200


if __name__ == "__main__":
    print('dataset_csv_path')
    print(dataset_csv_path)
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)




