

import training
import scoring
import deployment
import diagnostics
import reporting

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

    # Check for new files in the input_folder_path
    input_folder_path = config['input_folder_path']
    all_files = os.listdir(input_folder_path)

    new_files = [file for file in all_files if file not in ingested_files]

    # Ingest new data if there are any new files
    if new_files:
        ingestion(config)

    return new_files


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


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






