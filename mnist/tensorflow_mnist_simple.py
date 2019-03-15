import os
import errno
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np

from tensorflow import keras


tf.logging.set_verbosity(tf.logging.INFO)


def conv_model(feature, target, mode):
    """2-layer convolution model."""

    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = tf.layers.conv2d(feature, 32, kernel_size=[5, 5],
                                   activation=tf.nn.relu, padding="SAME")
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope('conv_layer2'):
        h_conv2 = tf.layers.conv2d(h_pool1, 64, kernel_size=[5, 5],
                                   activation=tf.nn.relu, padding="SAME")
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely connected layer with 1024 neurons.
    h_fc1 = tf.layers.dropout(
        tf.layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu),
        rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Compute logits (1 per class) and compute loss.
    logits = tf.layers.dense(h_fc1, 10, activation=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    return tf.argmax(logits, 1), loss


def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
        })
    width = tf.cast(features['width'], tf.int64)
    height = tf.cast(features['height'], tf.int64)
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, depth=10)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, (height, width, 1))
    return image, label


def normalize(image, label):
    """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label


def train_input_fn():
    """Input function for training"""
    dataset = tf.data.TFRecordDataset('train.tfrecords')
    dataset = dataset.shard(hvd.size(), hvd.rank())
    dataset = dataset.map(decode)
    dataset = dataset.map(normalize)
    # dataset = dataset.shuffle(1000 + 3 * batch_size)
    dataset = dataset.batch(64)
    dataset = dataset.repeat(1)
    features, label = dataset.make_one_shot_iterator().get_next()
    return features, label


# Horovod: initialize Horovod.
hvd.init()

features, label = train_input_fn()
predict, loss = conv_model(features, label, tf.estimator.ModeKeys.TRAIN)

# Horovod: adjust learning rate based on number of GPUs.
opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

train_op = opt.minimize(loss)

hooks = [
    hvd.BroadcastGlobalVariablesHook(0),
    tf.train.LoggingTensorHook(tensors={'loss': loss},
                               every_n_iter=10),
]

with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
    while not mon_sess.should_stop():
        mon_sess.run(train_op)
