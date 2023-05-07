import requests
import json
import os

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

output_model_path = config['output_model_path']


# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Call each API endpoint and store the responses
response1 = requests.post(f"{URL}/prediction", json={"file_path": "testdata/testdata.csv"})
response2 = requests.get(f"{URL}/scoring")
response3 = requests.get(f"{URL}/summarystats")
response4 = requests.get(f"{URL}/diagnostics")

# Check and print the status code and content of the responses
for i, response in enumerate([response1, response2, response3, response4], start=1):
    print(f"Response {i}: Status Code: {response.status_code}")
    print(f"Response {i}: Content: {response.content}")

# Combine all API responses
responses = {
    "predictions": response1.json(),
    "accuracy_score": response2.json(),
    "summary_statistics": response3.json(),
    "diagnostics": response4.json(),
}

# Write the responses to your workspace
with open(os.path.join(output_model_path, "apireturns.txt"), "w") as f:
    f.write(json.dumps(responses, indent=4))
