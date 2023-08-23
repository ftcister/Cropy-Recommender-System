import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_recommenders as tfrs
from tqdm.keras import TqdmCallback
from datetime import datetime
from sklearn.model_selection import train_test_split
from typing import List
from tensorflow.python.ops.math_ops import _bucketize as bucketize
import shutil
import argparse

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
num_cores = len(gpus)
selected_device = "/GPU:0"


class Bucketizer(tf.keras.layers.Layer):
    """Embedding layer based on bucketing a continuous variable."""

    def __init__(self, buckets: List[float], **kwargs) -> None:
        """Initializes the embedding layer.
        Args:
        buckets: Bucket boundaries.
        **kwargs: Extra args passed to the Keras Layer base class.
        """

        super().__init__(**kwargs)

        self.buckets = buckets

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return bucketize(x, boundaries=self.buckets)


class DCN(tfrs.Model):
    def __init__(
        self,
        use_cross_layer,
        deep_layer_sizes,
        embedding_dim,
        str_features,
        int_features,
        float_features,
        vocabularies,
        projection_dim,
    ):
        super().__init__()

        self.embedding_dimension = embedding_dim
        self.vocabularies = vocabularies
        self.str_features = str_features
        self.int_features = int_features
        self.float_features = float_features
        self._all_features = self.str_features + self.int_features + self.float_features
        self._embeddings = {}

        # Compute embeddings for string features.
        for feature_name in str_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [
                    tf.keras.layers.StringLookup(vocabulary=vocabulary),
                    tf.keras.layers.Embedding(
                        len(vocabulary) + 1, self.embedding_dimension
                    ),
                ]
            )

        # Compute embeddings for int features.
        for feature_name in int_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [
                    tf.keras.layers.IntegerLookup(
                        vocabulary=vocabulary, mask_value=None
                    ),
                    tf.keras.layers.Embedding(
                        len(vocabulary) + 1, self.embedding_dimension
                    ),
                ]
            )

        # Compute embeddings for float features.
        for feature_name in float_features:
            vocabulary = vocabularies[feature_name]
            bucket_array = np.arange(-1, 1, 2 / len(vocabulary)).tolist()
            # timestamp_buckets = np.linspace(-1, 1, num=2/len(vocabulary))
            self._embeddings[feature_name] = tf.keras.Sequential(
                [
                    # tf.keras.layers.IntegerLookup(vocabulary=vocabulary, mask_value=None),
                    Bucketizer(buckets=bucket_array),
                    # tf.keras.layers.Discretization(timestamp_buckets.tolist()),
                    # tf.keras.layers.Embedding(len(timestamp_buckets) + 2, 32)
                    tf.keras.layers.Embedding(
                        len(bucket_array) + 2, self.embedding_dimension
                    ),
                ]
            )

        self.linear = tf.keras.layers.Dense(embedding_dim)

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim, kernel_initializer="glorot_uniform"
            )
        else:
            self._cross_layer = None

        self._deep_layers = [
            tf.keras.layers.Dense(
                layer_size,
                activation="gelu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )
            for layer_size in deep_layer_sizes
        ]

        self._logit_layer = tf.keras.layers.Dense(1)

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.MeanSquaredError("MSE"),
                tf.keras.metrics.RootMeanSquaredError("RMSE"),
                tf.keras.metrics.MeanAbsoluteError("MAE"),
            ],
        )

    def call(self, features):
        # Concatenate embeddings
        embeddings = []
        for feature_name in self._all_features:
            embedding_fn = self._embeddings[feature_name]
            embeddings.append(embedding_fn(features[feature_name]))

        x = tf.concat(embeddings, axis=1)

        # Build Cross Network
        if self._cross_layer is not None:
            x = self._cross_layer(x)

        # Build Deep Network
        for deep_layer in self._deep_layers:
            x = tf.keras.layers.Dropout(0.1)(x)
            x = deep_layer(x)

        return self._logit_layer(x)

    def compute_loss(self, features, training=False):
        labels = features.pop("ranking")
        scores = self(features)
        return self.task(
            labels=labels,
            predictions=scores,
        )


SEED = 42
tf.keras.utils.set_random_seed(SEED)

parser = argparse.ArgumentParser(description="Train Recommender Model.")
parser.add_argument(
    "-d",
    "--dataset",
    required=True,
    metavar="dataset_name",
    type=str,
    help="the name of the dataset",
)
parser.add_argument(
    "-e",
    "--epochs",
    required=True,
    metavar="epochs",
    type=int,
    help="the number of epochs",
)
parser.add_argument(
    "-r",
    "--remove",
    default=False,
    metavar="remove_logs",
    action=argparse.BooleanOptionalAction,
    help="clear the logs folder",
)

args = parser.parse_args()
df_name = args.dataset

if args.remove:
    shutil.rmtree("logs/retrieval", ignore_errors=True)

df_final_retrieval = pd.read_parquet(df_name, engine="pyarrow")
df_final_retrieval = df_final_retrieval.groupby("product").filter(
    lambda x: (x["product"].count() >= 2).any()
)

tf_dict_df = tf.data.Dataset.from_tensor_slices(dict(df_final_retrieval))

# map rows to a dictionary
beer_ratings = tf_dict_df.map(
    lambda x: {
        "user_id": x["user_id"],
        "product": x["product"],
        "PRECIO": x["PRECIO"],
        "sin_weekday": x["sin_weekday"],
        "cos_weekday": x["cos_weekday"],
        "sin_monthday": x["sin_monthday"],
        "cos_monthday": x["cos_monthday"],
        "sin_month": x["sin_month"],
        "cos_month": x["cos_month"],
        "sin_hour": x["sin_hour"],
        "cos_hour": x["cos_hour"],
        "ranking": x["ranking"],
    }
)

products_dataset = beer_ratings.map(lambda x: x["product"])
usernames = beer_ratings.map(lambda x: x["user_id"])

unique_products = np.unique(np.concatenate(list(products_dataset.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(usernames.batch(1000))))

feature_names = [
    "user_id",
    "product",
    "PRECIO",
    "sin_weekday",
    "cos_weekday",
    "sin_monthday",
    "cos_monthday",
    "sin_month",
    "cos_month",
    "sin_hour",
    "cos_hour",
]


def get_vocab(feature_name, dataset):
    return np.unique(
        np.concatenate(list(dataset.map(lambda x: x[feature_name]).batch(1000000)))
    )


with tf.device(selected_device):
    vocabularies = {}
    for feature_name in feature_names:
        vocabularies[feature_name] = get_vocab(feature_name, beer_ratings)

str_features = ["user_id", "product"]
int_features = ["PRECIO"]
float_features = [
    "sin_weekday",
    "cos_weekday",
    "sin_monthday",
    "cos_monthday",
    "sin_month",
    "cos_month",
    "sin_hour",
    "cos_hour",
]

df_train, df_val = train_test_split(
    df_final_retrieval,
    test_size=0.1,
    random_state=SEED,
    stratify=df_final_retrieval["product"],
)

tf_interactions_train = tf.data.Dataset.from_tensor_slices(dict(df_train))
tf_interactions_val = tf.data.Dataset.from_tensor_slices(dict(df_val))

interaction_dataset_train = tf_interactions_train.map(
    lambda x: {
        "user_id": x["user_id"],
        "product": x["product"],
        "PRECIO": x["PRECIO"],
        "sin_weekday": x["sin_weekday"],
        "cos_weekday": x["cos_weekday"],
        "sin_monthday": x["sin_monthday"],
        "cos_monthday": x["cos_monthday"],
        "sin_month": x["sin_month"],
        "cos_month": x["cos_month"],
        "sin_hour": x["sin_hour"],
        "cos_hour": x["cos_hour"],
        "ranking": x["ranking"],
    }
)

interaction_dataset_val = tf_interactions_val.map(
    lambda x: {
        "user_id": x["user_id"],
        "product": x["product"],
        "PRECIO": x["PRECIO"],
        "sin_weekday": x["sin_weekday"],
        "cos_weekday": x["cos_weekday"],
        "sin_monthday": x["sin_monthday"],
        "cos_monthday": x["cos_monthday"],
        "sin_month": x["sin_month"],
        "cos_month": x["cos_month"],
        "sin_hour": x["sin_hour"],
        "cos_hour": x["cos_hour"],
        "ranking": x["ranking"],
    }
)

train_dataset = interaction_dataset_train.shuffle(
    len(df_train), seed=SEED, reshuffle_each_iteration=False
)
val_dataset = interaction_dataset_val.shuffle(
    len(df_val), seed=SEED, reshuffle_each_iteration=False
)

train_size = len(df_train)
val_size = len(df_val)


def optimal_n_params(dimension, n_features, train_size, hidden_dim):
    input_tensor = dimension * n_features
    dcn = (input_tensor * input_tensor) + input_tensor
    output_tensor = 1
    max_params = train_size / 10
    first_layer_params = input_tensor * hidden_dim + hidden_dim
    hidden_layer_params = hidden_dim * hidden_dim + hidden_dim
    final_layer_params = hidden_dim * output_tensor + output_tensor
    free_params = max_params - final_layer_params - first_layer_params - dcn
    n_layers = round(free_params / hidden_layer_params)
    return [hidden_dim] * n_layers


def num_params(dimension, n_features, layer_size, layer_num):
    dim = dimension * n_features
    dcn = (dim * dim) + dim
    first_hidden = dim * layer_size + layer_size
    deep = layer_size * layer_size + layer_size
    outuput = layer_size + 1
    return dcn + first_hidden + deep * (layer_num - 1) + outuput


d_model = int(max(len(unique_user_ids), len(unique_products)) ** 0.25)
layer_sizes = optimal_n_params(d_model, len(feature_names), train_size, d_model * 2)
deep_layer_sizes = layer_sizes
use_cross_layer = True
projection_dim = None

batch_size = train_size
freq = 1
epochs = args.epochs
patience = 50

lr = 1e-3
opt = tf.keras.optimizers.AdamW(amsgrad=True, learning_rate=lr)

model = DCN(
    use_cross_layer,
    deep_layer_sizes,
    d_model,
    str_features,
    int_features,
    float_features,
    vocabularies,
    projection_dim,
)

model.compile(optimizer=opt)

time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/ranking/scalars/" + time_now
checkpoint_filepath = (
    "models/ranking/checkpoints/" + df_name.split("/")[1].split(".")[0] + "/1/"
)

my_callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1, profile_batch="500,520"
    ),
    TqdmCallback(verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_total_loss",
        save_best_only=True,
        mode="min",
        verbose=0,
        save_weights_only=True,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_total_loss",
        restore_best_weights=True,
        min_delta=0,
        patience=patience,
        mode="min",
        start_from_epoch=0,
    ),
]

print("\n----------------------------------------")
print(f"The name of the dataset is: {df_name}")
print("dataset lenght:", len(df_final_retrieval))
print("train:", train_size)
print("val:", val_size)
print("----------------------------------------")
print("embedding dimension:", d_model)
print("number of layers:", len(deep_layer_sizes))
print("Neurons per layer:", deep_layer_sizes[0])
print(
    "Trainable parameters:",
    num_params(d_model, len(feature_names), d_model * 2, len(deep_layer_sizes)),
)
print("----------------------------------------")
print("epochs:", epochs)
print("batch size:", batch_size)
print("learning rate:", lr)
print("----------------------------------------")

history = model.fit(
    train_dataset.batch(
        batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    .cache()
    .prefetch(tf.data.AUTOTUNE),
    validation_freq=freq,
    epochs=epochs,
    verbose=0,
    callbacks=my_callbacks,
    validation_data=val_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .prefetch(tf.data.AUTOTUNE),
    use_multiprocessing=True,
    workers=16,
)

path_ranking = "models/ranking/ranking_" + df_name.split("/")[1].split(".")[0] + "/1"

tf.saved_model.save(
    model,
    path_ranking,
    options=tf.saved_model.SaveOptions(namespace_whitelist=["Ranking"]),
)
