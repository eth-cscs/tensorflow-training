# https://github.com/logicalclocks/hops-examples/blob/master/tensorflow/notebooks/Distributed_Training/collective_allreduce_strategy/keras.ipynb
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distribute#multi-worker-training

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


def input_fn():
    x = np.random.random((1024, 10))
    y = np.random.randint(2, size=(1024, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(32)
    return dataset


# Define a Keras Model.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model.
optimizer = tf.train.GradientDescentOptimizer(0.2)
model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.summary()
tf.keras.backend.set_learning_phase(True)

# Define DistributionStrategies and convert the Keras Model to an
# Estimator that utilizes these DistributionStrateges.
# Evaluator is a single worker, so using MirroredStrategy.
run_config = tf.estimator.RunConfig(
    experimental_distribute=tf.contrib.distribute.DistributeConfig(
        train_distribute=tf.contrib.distribute.CollectiveAllReduceStrategy(
            num_gpus_per_worker=1),
        eval_distribute=tf.contrib.distribute.MirroredStrategy(
            num_gpus_per_worker=1)))
keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                        config=run_config,
                                                        model_dir='./logs')

tf.estimator.train_and_evaluate(keras_estimator,
                                train_spec=tf.estimator.TrainSpec(
                                    input_fn=input_fn),
                                eval_spec=tf.estimator.EvalSpec(
                                    input_fn=input_fn))
