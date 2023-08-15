import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tqdm.keras import TqdmCallback
from datetime import datetime
#import shutil
import argparse
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, diff):
        super().__init__()
        self.linear_1 = tf.keras.layers.Dense(d_model)
        self.dense_act = tf.keras.layers.Dense(diff, activation='gelu', kernel_initializer=tf.keras.initializers.HeNormal())
        self.linear_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        x = self.linear_1(x)
        x = self.dense_act(x)
        x = self.linear_2(x)
        return x

class EmbeddingModel(tf.keras.Model):
    def __init__(self, embedding_dimension, vocab):
        super().__init__()
        self.embedings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None),
            tf.keras.layers.Embedding(input_dim=len(vocab) + 1, output_dim=embedding_dimension)
        ])

    def call(self, inputs):
        return self.embedings(inputs)

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.scale = tf.math.sqrt(tf.cast(dim, tf.float32))
        self.softmax = tf.keras.layers.Softmax()
        self.q_linear = tf.keras.layers.Dense(dim, use_bias=False)
        self.k_linear = tf.keras.layers.Dense(dim, use_bias=False)
        self.v_linear = tf.keras.layers.Dense(dim, use_bias=False)

    def call(self, x):
        attn = tf.matmul(self.q_linear(x), self.k_linear(x), transpose_b=True)
        attn_scaled = attn / self.scale
        attn = self.softmax(attn_scaled)
        attn_out = tf.matmul(attn, self.v_linear(x))
        return attn_out
'''
class AttentionBlock_PostLN(tf.keras.layers.Layer):
    def __init__(self, d_model, diff, num_heads):
        super().__init__()
        self.attention_stack = [AttentionLayer(d_model//num_heads) for _ in range(num_heads)]
        self.ffn = FeedForward(d_model, diff)
        self.LN_1 = tf.keras.layers.LayerNormalization()
        self.LN_2 = tf.keras.layers.LayerNormalization()
        self.ADD_1 = tf.keras.layers.Add()
        self.ADD_2 = tf.keras.layers.Add()
        self.dropout_1 = tf.keras.layers.Dropout(.1)
        self.dropout_2 = tf.keras.layers.Dropout(.1)
        self.linear = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x_prime = tf.concat([self.attention_stack[i](x) for i in range(len(self.attention_stack))], axis=-1)
        x_prime = self.linear(x_prime)
        x_add_1 = self.ADD_1([x, self.dropout_1(x_prime)])
        x_LN1 = self.LN_1(x_add_1)
        x_ffn = self.ffn(x_LN1)
        x_add_2 = self.ADD_2([x_LN1, self.dropout_2(x_ffn)])
        x_LN2 = self.LN_2(x_add_2)
        return x_LN2
'''   
class AttentionBlock_PreLN(tf.keras.layers.Layer):
    def __init__(self, d_model, diff, num_heads):
        super().__init__()
        self.attention_stack = [AttentionLayer(d_model//num_heads) for _ in range(num_heads)]
        self.ffn = FeedForward(d_model, diff)
        self.LN_1 = tf.keras.layers.LayerNormalization()
        self.LN_2 = tf.keras.layers.LayerNormalization()
        self.ADD_1 = tf.keras.layers.Add()
        self.ADD_2 = tf.keras.layers.Add()
        self.dropout_1 = tf.keras.layers.Dropout(.1)
        self.dropout_2 = tf.keras.layers.Dropout(.1)
        self.linear = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x_prime = self.LN_1(x)
        x_prime = tf.concat([self.attention_stack[i](x_prime) for i in range(len(self.attention_stack))], axis=-1)
        x_prime = self.linear(x_prime)
        x_add_1 = self.ADD_1([x, self.dropout_1(x_prime)])
        x_LN2 = self.LN_2(x_add_1)
        x_ffn = self.ffn(x_LN2)
        x_add_2 = self.ADD_2([x_add_1, self.dropout_2(x_ffn)])

        return x_add_2

class EncodingModel(tf.keras.Model):
    def __init__(self, layer_sizes, d_model, vocab, diff, num_heads):
        super().__init__()
        self.embedding_model = EmbeddingModel(d_model, vocab)
        self.attention_layers = [AttentionBlock_PreLN(d_model, diff, num_heads) for _ in range(layer_sizes)]
        self.final_linear = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        x = self.embedding_model(inputs)
        for layer in range(len(self.attention_layers)):
            x = self.attention_layers[layer](x)
        x = self.final_linear(x)
        return x

class RecommenderModel(tfrs.Model):
    def __init__(self, layer_sizes, d_model, user_vocab, products_vocab, task_dataset, diff, num_heads):
        super().__init__()
        self.query_model: tf.keras.Model = EncodingModel(layer_sizes,d_model,user_vocab,diff,num_heads)

        self.candidate_model: tf.keras.Model = EncodingModel(layer_sizes,d_model,products_vocab,diff,num_heads)

        self.task = tfrs.tasks.Retrieval(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=tfrs.metrics.FactorizedTopK(candidates=task_dataset.batch(128).cache().map(self.candidate_model)),
        )

    def train_step(self, features) -> tf.Tensor:
        with tf.GradientTape() as tape:

            context = self.query_model(features['user_id'])
            x = self.candidate_model(features['product'])
            loss = self.task(context, x, compute_metrics=False)

            try:
                del loss._keras_mask
            except AttributeError:
                pass

            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, features) -> tf.Tensor:

        context = self.query_model(features['user_id'])
        x = self.candidate_model(features['product'])
        loss = self.task(context, x, compute_metrics=True)

        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics
'''   
class deep_model(tf.keras.Model):
  

  def __init__(self, layer_sizes, d_model,user_vocab):
    super().__init__()

    self.embedding_model = EmbeddingModel(d_model,user_vocab)

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for _ in range(layer_sizes):
      self.dense_layers.add(tf.keras.layers.Dense(d_model, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()))
      self.dense_layers.add(tf.keras.layers.Dropout(.1))

    self.dense_layers.add(tf.keras.layers.Dense(1))

  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)
        
class RecommenderModel_simple_deep(tfrs.Model):
    def __init__(self, layer_sizes, d_model, user_vocab, products_vocab, task_dataset, diff, num_heads):
        super().__init__()
        self.query_model: tf.keras.Model = deep_model(layer_sizes, d_model, user_vocab)
        self.candidate_model: tf.keras.Model = deep_model(layer_sizes, d_model, products_vocab)


        self.task = tfrs.tasks.Retrieval(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=tfrs.metrics.FactorizedTopK(candidates=task_dataset.batch(128).cache().map(self.candidate_model)),
        )

    def train_step(self, features) -> tf.Tensor:
        with tf.GradientTape() as tape:

            context = self.query_model(features['user_id'])
            x = self.candidate_model(features['product'])
            loss = self.task(context, x, compute_metrics=False)

            try:
                del loss._keras_mask
            except AttributeError:
                pass

            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, features) -> tf.Tensor:

        context = self.query_model(features['user_id'])
        x = self.candidate_model(features['product'])
        loss = self.task(context, x, compute_metrics=True)

        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics
    
class RecommenderModel_base(tfrs.Model):
    def __init__(self, layer_sizes, d_model, user_vocab, products_vocab, task_dataset, diff, num_heads):
        super().__init__()
        self.query_model: tf.keras.Model = EncodingModel(d_model, user_vocab)
        self.candidate_model: tf.keras.Model = EncodingModel(d_model, products_vocab)


        self.task = tfrs.tasks.Retrieval(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=tfrs.metrics.FactorizedTopK(candidates=task_dataset.batch(128).cache().map(self.candidate_model)),
        )

    def train_step(self, features) -> tf.Tensor:
        with tf.GradientTape() as tape:

            context = self.query_model(features['user_id'])
            x = self.candidate_model(features['product'])
            loss = self.task(context, x, compute_metrics=False)

            try:
                del loss._keras_mask
            except AttributeError:
                pass

            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, features) -> tf.Tensor:

        context = self.query_model(features['user_id'])
        x = self.candidate_model(features['product'])
        loss = self.task(context, x, compute_metrics=True)

        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics
   
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
'''  
def optimal_n_params(dimension, train_size):
	params_per_layer = (12*(dimension*dimension)) + (9*dimension)
	max_params = train_size/10
	n_layers = round(max_params/params_per_layer)
	return n_layers
	
def num_params(dimension, num_layers):
    attn = 3*(dimension*dimension) #3x2
    linear = (dimension*dimension)+dimension # x2 + x 
    ffn = dimension*(dimension*4)+(4*dimension) + ((dimension*4)*dimension) + dimension #4x2 + 4x + 4x2 + x => 8x2 + 5x 
    return (attn + linear + ffn)*num_layers #12x²+ 6x

SEED = 42
tf.random.set_seed(SEED)

#shutil.rmtree('logs/retrieval', ignore_errors=True)

parser = argparse.ArgumentParser(description='Train Recommender Model.')
parser.add_argument('-d', '--dataset', required=True,metavar='dataset_name', type=str, help='the name of the dataset')
parser.add_argument('-e', '--epochs', required=True,metavar='epochs', type=int, help='the number of epochs')

args = parser.parse_args()
df_name = args.dataset

df_final_retrieval = pd.read_parquet(df_name, engine="pyarrow")
df_final_retrieval = df_final_retrieval.groupby('product').filter(lambda x : (x['product'].count()>=2).any())

df_products = df_final_retrieval['product'].unique()
df_products = pd.DataFrame(df_products, columns=['product'])

interactions = df_final_retrieval[['user_id', 'product']]
df_interaction = pd.DataFrame(interactions, columns=['user_id', 'product'])

tf_products = tf.data.Dataset.from_tensor_slices(dict(df_products))
tf_interactions = tf.data.Dataset.from_tensor_slices(dict(df_interaction))

interaction_dataset = tf_interactions.map(lambda x: {
    "product": x["product"],
    "user_id": x["user_id"]
})

products_dataset = tf_products.map(lambda x: x['product'])
usernames = interaction_dataset.map(lambda x: x['user_id'])

unique_products = np.unique(np.concatenate(list(products_dataset.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(usernames.batch(1000))))

df_train, df_val = train_test_split(df_final_retrieval, test_size=0.1, random_state=SEED, stratify=df_final_retrieval['product'])

tf_interactions_train = tf.data.Dataset.from_tensor_slices(dict(df_train))
tf_interactions_val = tf.data.Dataset.from_tensor_slices(dict(df_val))

interaction_dataset_train = tf_interactions_train.map(lambda x: {
    "product": x["product"],
    "user_id": x["user_id"]
})

interaction_dataset_val = tf_interactions_val.map(lambda x: {
    "product": x["product"],
    "user_id": x["user_id"]
})

train_dataset = interaction_dataset_train.shuffle(len(df_train), seed=SEED, reshuffle_each_iteration=False)
val_dataset = interaction_dataset_val.shuffle(len(df_val), seed=SEED, reshuffle_each_iteration=False)

train_size = len(df_train)
val_size = len(df_val)

d_model = int(max(len(unique_user_ids),len(unique_products))**.25)
layer_sizes = optimal_n_params(d_model, train_size)
diff = d_model*4
num_heads = 4

freq = 1
epochs = args.epochs
patience = 50

#batch_list = [32,64,128,256,512,1024,2048,4096]
batch_size = 1024*3

#warm_up_epochs = 50
#warm_up_steps = int((train_size)/ batch_size) * warm_up_epochs
#warm_up_steps = 4000

#lr_list = [1e-2,5e-2,1e-3,5e-3,1e-4,5e-4,1e-5,5e-5]
lr = 1e-4
#lr = CustomSchedule(d_model, warm_up_steps)

#opt = tf.keras.optimizers.Adam(learning_rate=lr) #variar
opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr) #legacy for m1 mac

model = RecommenderModel(layer_sizes,d_model,unique_user_ids,unique_products,products_dataset,diff,num_heads)

model.compile(optimizer=opt)

time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/retrieval/scalars/" + time_now
#checkpoint_filepath = 'models/retrieval/checkpoints/' + df_name.split('/')[1].split('.')[0]+'/1/'

my_callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch="500,520"),
    TqdmCallback(verbose=1),
    #tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,monitor="val_total_loss",save_best_only=True,mode="min",verbose=0,save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_total_loss",restore_best_weights=True,min_delta=0,patience=patience,mode="min",start_from_epoch=0),
]

print("\n----------------------------------------")
print(f"The name of the dataset is: {df_name}")
print("dataset lenght:", len(df_final_retrieval))
print("train:", train_size)
print("val:", val_size)
print("unique_users:", len(unique_user_ids))
print("unique_products:", len(unique_products))
#print("test:", int(len(df_final_retrieval)*0.1))
print("----------------------------------------")
print("embedding dimension:", d_model)
print("number of heads: ", num_heads)
print("number of layers:", layer_sizes)
print("Trainable parameters:", num_params(d_model, layer_sizes))
print("----------------------------------------")
print("epochs:", epochs)
print('batch size:', batch_size)
print('learning rate:', lr)
#print("warm_up_epochs:", warm_up_epochs)
print("----------------------------------------")

history = model.fit(
    train_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE,deterministic=True).cache().prefetch(tf.data.AUTOTUNE),
    validation_freq=freq,
    epochs=epochs,
    verbose=0,
    callbacks=my_callbacks,
    validation_data=val_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE),
    use_multiprocessing=True,
    workers=16,
)
# SCANN

''' 
scann = tfrs.layers.factorized_top_k.ScaNN(model.query_model, num_reordering_candidates=1000)

products_embeddings = products_dataset.batch(4096).map(model.candidate_model)
scann.index_from_dataset(tf.data.Dataset.zip((products_dataset.batch(4096), products_embeddings)))
# Need to call it to set the shapes.
_ = scann(np.array(["42"]))

path_scann = 'models/scann/scann_'+df_name.split('/')[1].split('.')[0]+'/1'
tf.saved_model.save(
    scann,
    path_scann,
    options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]))

'''
