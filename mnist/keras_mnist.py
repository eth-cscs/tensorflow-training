import os
import math
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow import keras
from tensorflow.keras import backend as K


# If using `import horovod.keras as hvd` instead
# of `import horovod.tensorflow.keras as hvd` makes
# Horovod's internals to use the standalone Keras
# intead of TensorFlow's Keras.
# https://github.com/uber/horovod/issues/511


# Horovod: initialize Horovod.
hvd.init()

# Horovod: adjust number of epochs based on number of GPUs.
BATCH_SIZE = 64
NUM_EPOCS = 4
NUM_EPOCS = int(math.ceil(NUM_EPOCS / hvd.size()))

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))


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


def get_dataset(filename):
    """Reads input data num_epochs times."""
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.shard(hvd.size(), hvd.rank())
    dataset = dataset.map(decode)
    dataset = dataset.map(normalize)
    dataset = dataset.shuffle(6000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)

    return dataset


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(784,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# The input shape in keras doesn't need to be a vector (1D),
# it can be a matrix and keras takes care of it. Here
# it would be also possible to use
# keras.layers.Flatten(input_shape=(28, 28)
# but I would prefer to let the input pipeline take care of
# that so it can be reused by tensorflow.

# Horovod: adjust learning rate based on number of GPUs.
opt = keras.optimizers.Adam(0.001 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all
    # other processes. This is necessary to ensure consistent initialization
    # of all workers when training is started with random weights or restored
    # from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers
# from corrupting them.
if hvd.rank() == 0:
    callbacks.append(
        keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Here it is necessary to specify both the batch size on the tensorflow input
# pipeline (tf.dataset) and `steps_per_epoch` for keras.
# This seemscontradictory but I think that may be because the total number of
# sample maybe is unknown if tf.dataset is used, which would make impossible
# to determine the batch size.
# `steps_per_epoch` is needed if using dataset iterators.
fit = model.fit(get_dataset('train.tfrecords'),
                epochs=NUM_EPOCS,
                steps_per_epoch=math.ceil(60000 / BATCH_SIZE),
                callbacks=callbacks)

test_loss, test_acc = model.evaluate(get_dataset('test.tfrecords'), steps=10)
print('Predict: loss = %f, accuracy = %f' % (test_loss, test_acc))
# print(fit.params)
