# # Convert MNIST data to TFRecords file format with Example protos
#
# This is based on the examples convert_to_records.py and fully_connected_reader.py on
# https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/examples/how_tos/reading_data/
# at the TensorFlow repository.
#
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist


print(f'TensorFlow-{tf.version.VERSION}')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, filename, directory):
    """Converts a dataset to tfrecords."""
    num_examples = images.shape[0]

    # This needs to be written on the TFRecord file to construct back
    # the image when reading the file. On the reading example this is not
    # used though, they use `mnist.IMAGE_PIXELS` instead.
    rows = images.shape[1]   # 28
    cols = images.shape[2]   # 28

    filename = os.path.join(directory, filename + '.tfrecords')
    print('Writing', filename)

    with tf.io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(rows),
                        'width': _int64_feature(cols),
                        'label': _int64_feature(int(labels[index])),
                        'image_raw': _bytes_feature(image_raw)
                    }))
            writer.write(example.SerializeToString())


(x_train, y_train), (x_test, y_test) = mnist.load_data()

convert_to(x_train, y_train, 'train', '.')
convert_to(x_test, y_test, 'test', '.')
