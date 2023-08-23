import pandas as pd
import numpy as np
import copy
import tempfile
import argparse


def prep_time(values, period):
    return np.sin(2 * np.pi * values / period), np.cos(2 * np.pi * values / period)


parser = argparse.ArgumentParser(description="Preprocess the dataset")
parser.add_argument(
    "-d",
    "--dataset",
    required=True,
    metavar="dataset_name",
    type=str,
    help="the name of the dataset",
)

args = parser.parse_args()
df_name = args.dataset

df = pd.read_csv(df_name, engine="pyarrow")

user_dict = {}
user_count = {}
for user, product in zip(df["user_id"], df["product"]):
    if user not in user_count:
        user_count[user] = {}
    if product in user_count[user]:
        user_count[user][product] += 1
    else:
        user_count[user][product] = 1
user_dict = copy.deepcopy(user_count)
for user in user_dict:
    total = sum(user_dict[user].values())
    for product in user_dict[user]:
        user_dict[user][product] /= total

aux_df = (
    df.groupby("store", as_index=False)[["user_id"]]
    .agg(["count", "nunique"])
    .reset_index()
)
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
    aux_df.to_csv(temp_file.name, index=False)
    store_df = pd.read_csv(temp_file.name, index_col=0)
store_dict = store_df.to_dict("index")
item_total_dict = {}
item_unique_dict = {}
item_helper = {}

for user, product, local in zip(df["user_id"], df["product"], df["store"]):
    if product not in item_helper:
        item_total_dict[product] = 0
        item_unique_dict[product] = 0
        item_helper[product]["user_list"] = []
    if user not in item_helper[product]["user_list"]:
        item_helper[product]["user_list"].append(user)
        item_unique_dict[product] += 1 / store_dict[local]["nunique"]
    item_total_dict[product] += 1 / store_dict[local]["count"]

df["hour"] = pd.to_datetime(df["Fecha"]).dt.hour
df["weekday"] = pd.to_datetime(df["Fecha"]).dt.weekday
df["monthday"] = pd.to_datetime(df["Fecha"]).dt.day
df["month"] = pd.to_datetime(df["Fecha"]).dt.month

df["sin_hour"], df["cos_hour"] = prep_time(df["hour"], 24)
df["sin_weekday"], df["cos_weekday"] = prep_time(df["weekday"], 7)
df["sin_monthday"], df["cos_monthday"] = prep_time(df["monthday"], 30)
df["sin_month"], df["cos_month"] = prep_time(df["month"], 12)

df["ranking"] = df.apply(
    lambda x: user_dict[x["user_id"]][x["product"]]
    + 0.1
    * (1 - 1 / len(user_dict[x["user_id"]]))
    * np.random.uniform(
        item_total_dict[x["product"]], 1 - item_unique_dict[x["product"]]
    ),
    axis=1,
)

df_prep_retrieval = df[["user_id", "product"]]
df_prep_ranking = df[
    [
        "user_id",
        "product",
        "PRECIO",
        "sin_hour",
        "cos_hour",
        "sin_weekday",
        "cos_weekday",
        "sin_monthday",
        "cos_monthday",
        "sin_month",
        "cos_month",
        "ranking",
    ]
]

df_prep_retrieval.to_parquet("datasets/retrieval/data_prep.parquet", index=False)
df_prep_ranking.to_parquet("datasets/ranking/data_prep.parquet", index=False)
