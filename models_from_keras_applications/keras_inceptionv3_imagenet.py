import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


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
    label = tf.one_hot(label, 1001)
    return image, label


def train_input():
    data_dir = '/scratch/snx3000/stud71/imagenet/'
    list_of_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    dataset = tf.data.Dataset.list_files(list_of_files)
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 cycle_length=120,
                                 block_length=1,
                                 num_parallel_calls=12)
    dataset = dataset.map(decode, num_parallel_calls=12)
    dataset = dataset.batch(64)
    # dataset = dataset.apply(
    #    tf.data.experimental.prefetch_to_device('/gpu:0', buffer_size=64))
    dataset = dataset.repeat(2)
    return dataset


model = keras.applications.InceptionV3(weights=None,
                                       input_shape=(224, 224, 3),
                                       classes=1001)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

optimizer = keras.optimizers.SGD(lr=0.0001, momentum=0.9)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fit = model.fit(train_input(),
                epochs=2,
                steps_per_epoch=100
                )
