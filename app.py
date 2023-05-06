from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list
from scoring import score_model

app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None

@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'POST':
        file_location = request.form['file_location']
        predictions = model_predictions(file_location)
        return jsonify(predictions=predictions)
    return jsonify(message="Provide file location"), 200

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
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
