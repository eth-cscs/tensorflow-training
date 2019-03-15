import os
import errno
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow import keras


print('TensorFlow-%s' % tf.VERSION)
print('tf.keras-%s' % keras.__version__)
tf.logging.set_verbosity(tf.logging.INFO)


class Parameters:
    def __init__(self):
        self.batch_size = 64
        self.num_epochs = 2
        self.hidden1 = 128
        self.hidden2 = 32
        self.train_dir = '.'
        self.learning_rate = 0.01


params = Parameters()


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                           logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.001 * hvd.size(), momentum=0.9)

        optimizer = hvd.DistributedOptimizer(optimizer)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    label = tf.cast(features['label'], tf.int32)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    return image, label


def normalize(image, label):
    """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label


def train_input_fn():
    """Input function for training"""
    batch_size = params.batch_size
    filename = os.path.join('.', 'train.tfrecords')
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)

    features, label = dataset.make_one_shot_iterator().get_next()

    return features, label

# the train_input_fn could be reused for evaluation by giving arguments
# to and then simply using a lambda function, but for the purpose of
# ilustration of the API we use here two different functions
# without arguments.
def evalualte_input_fn():
    """Input function for training"""
    batch_size = params.batch_size
    filename = os.path.join('.', 'test.tfrecords')
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        # dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)

    features, label = dataset.make_one_shot_iterator().get_next()

    return features, label


hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())

# Horovod: save checkpoints only on worker 0 to prevent other workers from
# corrupting them.
# This doesn't prevent the other workers from saving checkpoints. They save
# their check points on `/tmp`.
# model_dir = './mnist_convnet_model' if hvd.rank() == 0 else None
model_dir = './chkpt_tfestimator_mnist_%s' % hvd.rank()

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=model_dir,
    config=tf.estimator.RunConfig(session_config=config))

# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=500)

# Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
# rank 0 to all other processes. This is necessary to ensure consistent
# initialization of all workers when training is started with random weights or
# restored from a checkpoint.
bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

# Horovod: adjust number of steps based on number of GPUs.
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=200 // hvd.size(),
    hooks=[logging_hook, bcast_hook])

# # Evaluate the model and print results
eval_results = mnist_classifier.evaluate(input_fn=evalualte_input_fn)
print(eval_results)
