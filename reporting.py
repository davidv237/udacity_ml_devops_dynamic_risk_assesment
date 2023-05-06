import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 



def score_model():
    # Load test data
    test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
    test_data = pd.read_csv(test_data_path)


    # Preprocess test data
    X_test = test_data.drop(columns=['exited'])
    y_test = test_data['exited']
    label_encoder = LabelEncoder()
    X_test['corporation'] = label_encoder.fit_transform(X_test['corporation'])

    # Load the trained model
    model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save the confusion matrix plot to the workspace
    cm_plot_path = os.path.join(config['output_folder_path'], 'confusionmatrix.png')
    plt.savefig(cm_plot_path)

if __name__ == '__main__':
    score_model()
