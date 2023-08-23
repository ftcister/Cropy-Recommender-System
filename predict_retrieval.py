import requests
import json
import argparse

parser = argparse.ArgumentParser(description="predict Recommender Model.")
parser.add_argument(
    "-u",
    "--user_id",
    required=True,
    metavar="user_id",
    type=str,
    help="the id of the user to predict for",
)
parser.add_argument(
    "-k",
    "--topK",
    required=True,
    metavar="topK",
    type=str,
    help="number of recommendations to return",
)

args = parser.parse_args()
user_id = args.user_id
topK = args.topK

url = "http://localhost:8501/v1/models/retrieval:predict"
data = {"instances": [str(user_id)]}

data_json = json.dumps(data)

headers = {"content-type": "application/json"}

response = requests.post(url, data=data_json, headers=headers)

print("Recomendaciones para", user_id + ":")
for idx, preds in enumerate(response.json()["predictions"][0]["output_2"]):
    if idx < int(topK):
        print(idx + 1, preds)
