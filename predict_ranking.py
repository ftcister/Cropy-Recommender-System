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
    "-f",
    "--features",
    required=True,
    metavar="features",
    type=str,
    help="features to use for ranking",
)

args = parser.parse_args()
user_id = args.user_id
raw_features = args.products

features = raw_features.split(",")

url = "http://localhost:8501/v1/models/ranking:predict"

data = {
    "instances": [
        {
            "user_id": str(user_id),
            "product": str(features[0]),
            "PRECIO": int(features[1]),
            "sin_weekday": float(features[2]),
            "cos_weekday": float(features[3]),
            "sin_monthday": float(features[4]),
            "cos_monthday": float(features[5]),
            "sin_month": float(features[6]),
            "cos_month": float(features[7]),
            "sin_hour": float(features[8]),
            "cos_hour": float(features[9]),
        }
    ]
}

data_json = json.dumps(data)

headers = {"content-type": "application/json"}

response = requests.post(url, data=data_json, headers=headers)

test_rating = response.json()["predictions"][0]["output_2"][0]

print("Ranking:", test_rating)
