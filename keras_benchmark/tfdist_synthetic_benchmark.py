# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import numpy as np
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.keras import applications

# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-batches-per-iter', type=int, default=16,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda
device = 'GPU' if args.cuda else 'CPU'

size = int(os.environ['SLURM_NNODES'])
rank = int(os.environ['SLURM_NODEID'])

if rank == 0:
    print('Model: %s' % args.model)
    print('Batch size: %d' % args.batch_size)
    print('Number of %ss: %d' % (device, size))

if args.cuda:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set up standard model.
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    cluster_resolver=tf.distribute.cluster_resolver.SlurmClusterResolver(),
#     communication=tf.distribute.experimental.CollectiveCommunication.NCCL,
)

with strategy.scope():
    model = getattr(applications, args.model)(weights=None)
    opt = tf.optimizers.SGD(0.05, momentum=0.9, nesterov=True)
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=opt)

# Synthetic dataset
tf.random.set_seed(42)
data = tf.random.uniform([args.batch_size * args.num_batches_per_iter, 224, 224, 3])
target = tf.random.uniform([args.batch_size * args.num_batches_per_iter, 1], minval=0, maxval=999, dtype=tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((data, target)).repeat().batch(args.batch_size * size)

# Sharding by data
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)

steps_per_epoch = np.ceil(args.num_batches_per_iter / size)

callbacks = []

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.img_secs = []

    def on_train_end(self, logs=None):
        img_sec_mean = np.mean(self.img_secs)
        img_sec_conf = 1.96 * np.std(self.img_secs)
        print('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
        print('Total img/sec on %d %s(s): %.1f +-%.1f' %
             (size, device, size * img_sec_mean, size * img_sec_conf))

    # def on_train_batch_end(self, batch, logs=None):
    #     print(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs=None):
        time = timer() - self.starttime
        img_sec = args.batch_size * steps_per_epoch / time
        # print(logs, args.batch_size, steps_per_epoch, time)
        print('Iter #%d: %.1f img/sec per %s' % (epoch, img_sec, device))
        # skip warm up epoch
        if epoch > 0:
            self.img_secs.append(img_sec)

# write logs on worker 0.
if rank == 0:
    timing = TimingCallback()
    callbacks.append(timing)

# Train the model.
model.fit(
    dataset,
    steps_per_epoch=steps_per_epoch,
    # validation_data=dataset,
    # validation_steps=steps_per_epoch,
    callbacks=callbacks,
    epochs=args.num_iters,
    verbose=0,
)

# print(model.evaluate(
#     dataset,
#     batch_size=args.batch_size,
#     steps=args.num_batches_per_iter,
# ))
