import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


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
    label = tf.one_hot(label, 1000)
    return image, label


def train_input_fn():
    data_dir = '/scratch/snx3000/stud71/imagenet/'
    list_of_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    dataset = tf.data.Dataset.list_files(list_of_files)
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 cycle_length=120,
                                 block_length=1,
                                 num_parallel_calls=12)
    dataset = dataset.map(decode, num_parallel_calls=12)
    dataset = dataset.batch(64)
    dataset = dataset.apply(
        tf.data.experimental.prefetch_to_device('/gpu:0', buffer_size=64))
    # dataset = dataset.repeat(2)
    images, label = dataset.make_one_shot_iterator().get_next()
    return images, label


tf.logging.set_verbosity(tf.logging.INFO)

keras_model = keras.applications.InceptionV3(weights=None,
                                             input_shape=(224, 224, 3),
                                             classes=1000)

keras_model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                    loss='categorical_crossentropy',
                    metric='accuracy')

classifier = keras.estimator.model_to_estimator(
    keras_model=keras_model,
    model_dir='./checkpoints/')

profile_hook = tf.train.ProfilerHook(save_steps=100,
                                     show_dataflow=False,
                                     show_memory=False,
                                     output_dir='./profile/')

classifier.train(
    input_fn=train_input_fn,
    hooks=[profile_hook])
