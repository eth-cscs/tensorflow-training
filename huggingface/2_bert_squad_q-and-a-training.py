import json
import dataset_utils as du
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFBertModel


hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
train_path = keras.utils.get_file("train.json", train_data_url, cache_dir="./")
eval_path = keras.utils.get_file("eval.json", eval_data_url, cache_dir="./")


with open(train_path) as f:
    raw_train_data = json.load(f)

with open(eval_path) as f:
    raw_eval_data = json.load(f)


max_len = 384

train_squad_examples = du.create_squad_examples(raw_train_data, max_len)
x_train, y_train = du.create_inputs_targets(train_squad_examples)
print(f"{len(train_squad_examples)} training points created.")

#eval_squad_examples = du.create_squad_examples(raw_eval_data, max_len)
#x_eval, y_eval = du.create_inputs_targets(eval_squad_examples)
#print(f"{mlen(eval_squad_examples)} evaluation points created.")

encoder = TFBertModel.from_pretrained("bert-base-uncased", cache_dir='/scratch/snx3000/sarafael/bert_model')

input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
embedding = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
start_logits = layers.Flatten()(start_logits)
start_probs = layers.Activation(keras.activations.softmax)(start_logits)

end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
end_logits = layers.Flatten()(end_logits)
end_probs = layers.Activation(keras.activations.softmax)(end_logits)

model = keras.Model(inputs=[input_ids, token_type_ids, attention_mask],
                    outputs=[start_probs, end_probs])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

optimizer = keras.optimizers.Adam(lr=5e-5)
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(optimizer=optimizer, loss=[loss, loss])


def dataset_generator(x=x_train, y=y_train):
    for i in range(x[0].shape[0]):
        yield ((x[0][i], x[1][i], x[2][i]),
               (y[0][i], y[1][i]))


dataset = tf.data.Dataset.from_generator(dataset_generator,
                                         output_types=((tf.int32, tf.int32, tf.int32),(tf.int32, tf.int32)),
                                         output_shapes=(((max_len,), (max_len,), (max_len,)),((), ()))
                                        )
dataset = dataset.batch(16)  # originally 64
dataset = dataset.shard(hvd.size(), hvd.rank())


fit = model.fit(dataset.take(20),
                epochs=1,
                # verbose=2,
                callbacks=[hvd.callbacks.BroadcastGlobalVariablesCallback(0)])
