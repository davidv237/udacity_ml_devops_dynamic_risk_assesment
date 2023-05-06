import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe(data_folder=input_folder_path):
    #check for datasets, compile them together, and write to an output file
    all_files = os.listdir(data_folder)
    df_list = []

# Read all files and append data to df_list
    for file in all_files:
        # Check if the file has a .csv extension
        if file.endswith('.csv'):
            file_path = os.path.join(data_folder, file)
            df_list.append(pd.read_csv(file_path))

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # Remove duplicates
    deduped_df = combined_df.drop_duplicates()

    # Save the compiled DataFrame to a CSV file
    deduped_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)


    # Record the ingested files in a txt file
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        for file in all_files:
            if file.endswith('.csv'):
                f.write(f"{file}\n")


if __name__ == '__main__':
    merge_multiple_dataframe()
