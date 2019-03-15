import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from models.resnet_model import create_resnet101_model


tf.logging.set_verbosity(tf.logging.INFO)

resnet101 = create_resnet101_model(None)

def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.resize_images(image, (224, 224))
    label = tf.cast(features['image/class/label'], tf.int64)

    return image, label


def train_input_fn():
    data_dir = '/scratch/snx3000/stud71/imagenet/'
    list_of_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                     if int(f.split('-')[-3]) < 1000]
    dataset = tf.data.Dataset.list_files(list_of_files)
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 cycle_length=157,
                                 block_length=1,
                                 num_parallel_calls=12)
    dataset = dataset.map(decode, num_parallel_calls=12)
    dataset = dataset.batch(64)
    dataset = dataset.apply(
        tf.data.experimental.prefetch_to_device('/gpu:0', buffer_size=4))
    # dataset = dataset.repeat(NUM_EPOCHS)
    images, label = dataset.make_one_shot_iterator().get_next()
    return images, label


def test_input_fn():
    data_dir = '/scratch/snx3000/stud71/imagenet/'
    list_of_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                     if int(f.split('-')[-3]) > 1000]
    dataset = tf.data.Dataset.list_files(list_of_files)
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 cycle_length=3,
                                 block_length=1,
                                 num_parallel_calls=3)
    dataset = dataset.map(decode, num_parallel_calls=12)
    dataset = dataset.batch(64)
    dataset = dataset.apply(
        tf.data.experimental.prefetch_to_device('/gpu:0', buffer_size=4))
    # dataset = dataset.repeat(NUM_EPOCHS)
    images, label = dataset.make_one_shot_iterator().get_next()
    return images, label


def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability.
    Input:
    * features :: This is batch_features from input_fn
    * labels   :: This is batch_labels from input_fn
    * mode     :: An instance of tf.estimator.ModeKeys, see below
    * params   :: Additional configuration parameters like leraning rate,
                  number of hiden units, etc.
    """
    # Use a model from https://github.com/tensorflow/benchmarks
    logits = resnet101.build_network((features, labels),
                                 phase_train=True,
                                 nclass=1001).logits
    predicted_classes = tf.argmax(logits, 1)

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     predictions = {
    #         'class_ids': predicted_classes[:, tf.newaxis],
    #         'probabilities': tf.nn.softmax(logits),
    #         'logits': logits,
    #     }
    #     return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}            # during evaluation
    tf.summary.scalar('accuracy', accuracy[1])  # during training

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)

    train_op = optimizer.minimize(loss,
                                  global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    


classifier = tf.estimator.Estimator(
    model_fn=my_model,
    model_dir='./checkpoints_resnet101',
    # params={
    #     'feature_columns': my_feature_columns,
    #     'hidden_units': [10, 10],  # Two hidden layers of 10 nodes each.
    #     'n_classes': 3,  # The model must choose between 3 classes.
    # }
)

train = classifier.train(
    input_fn=train_input_fn,
    steps=100)

eval_result = classifier.evaluate(
    input_fn=test_input_fn,
    steps=10)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
