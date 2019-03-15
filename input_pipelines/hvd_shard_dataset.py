# This is the horovod version of `getting_started_with_tensorflows_dataset_api_4.ipynb`
#
# wget https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv
# echo "sepal_length,sepal_width,petal_length,petal_width,species" > iris_setosa.csv
# grep setosa iris.csv >> iris_setosa.csv
# echo "sepal_length,sepal_width,petal_length,petal_width,species" > iris_versic.csv
# grep versicolor iris.csv >> iris_versic.csv
# echo "sepal_length,sepal_width,petal_length,petal_width,species" > iris_virgin.csv
# grep virginica iris.csv >> iris_virgin.csv

import tensorflow as tf
import horovod.tensorflow as hvd


hvd.init()

classes = ['setosa', 'virginica', 'versicolor']


def parse_columns(*row, classes):
    """Convert the string classes to one-hot encoded:
    setosa     -> [1, 0, 0]
    virginica  -> [0, 1, 0]
    versicolor -> [0, 0, 1]
    """
    features = tf.convert_to_tensor(row[:4])
    label_int = tf.where(tf.equal(classes, row[4]))
    label = tf.one_hot(label_int, 3)
    return features, label


def get_csv(filename):
    return tf.data.experimental.CsvDataset(filename, header=True,
                                           record_defaults=[tf.float32,
                                                            tf.float32,
                                                            tf.float32,
                                                            tf.float32,
                                                            tf.string])


dataset = tf.data.Dataset.list_files(['iris_setosa.csv',
                                      'iris_versic.csv'],
                                     shuffle=False)
dataset = dataset.shard(hvd.size(), hvd.rank())
dataset = dataset.interleave(get_csv,
                             cycle_length=2,
                             block_length=1,
                             num_parallel_calls=2)
dataset = dataset.map(lambda *row: parse_columns(*row, classes=classes))
iterator = dataset.make_one_shot_iterator()
next_item = iterator.get_next()

with tf.Session() as sess:
    try:
        for i in range(10):
            features, label = sess.run(next_item)
            print('rank %d: features: %s  |  label: %s' %
                  (hvd.rank(), features, label))
    except tf.errors.OutOfRangeError:
        print('The dataset ran out of entries!')
