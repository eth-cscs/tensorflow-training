import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from datetime import datetime


hvd.init()


def decode(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.resize(image, (224, 224))
    label = tf.cast(features['image/class/label'], tf.int64)
    label = tf.one_hot(label, 1001)
    return image, label


data_dir = '/scratch/snx3000/stud50/imagenet/'
list_of_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

dataset = tf.data.Dataset.list_files(list_of_files)
dataset = dataset.interleave(tf.data.TFRecordDataset,
                             cycle_length=120,
                             block_length=1)
dataset = dataset.map(decode)
dataset = dataset.batch(128)
dataset = dataset.shard(hvd.size(), hvd.rank())

model = tf.keras.applications.InceptionV3(weights=None,
                                          input_shape=(224, 224, 3),
                                          classes=1001)

optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hvd_callback = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

fit = model.fit(dataset,
                epochs=1,
                callbacks=[hvd_callback])
