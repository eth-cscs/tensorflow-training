import os
import glob
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from datetime import datetime


hvd.init()

image_shape = (224, 224)
batch_size = 128

def decode(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.resize(image, image_shape, method='bicubic')
    label = tf.cast(features['image/class/label'], tf.int64)
    return image, label-1


list_of_files = glob.glob('/scratch/snx3000/stud50/imagenet/train*')

AUTO = tf.data.experimental.AUTOTUNE
dataset = (tf.data.TFRecordDataset(list_of_files, num_parallel_reads=AUTO)
           .shard(hvd.size(), hvd.rank())
           .map(decode, num_parallel_calls=AUTO)
           .shuffle(2048)
           .batch(batch_size)
           .prefetch(AUTO)
          )

model = tf.keras.applications.InceptionV3(weights=None,
                                          input_shape=(*image_shape, 3),
                                          classes=1000)

optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

hvd_callback = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('inceptionv3_logs',
                                                                  datetime.now().strftime("%d-%H%M")),
                                             histogram_freq=1,
                                             profile_batch='80,100')

fit = model.fit(dataset,
                epochs=1,
                callbacks=[hvd_callback, tb_callback])
