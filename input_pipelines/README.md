# Getting started with TensorFlow's input pipelines and TFRecord files

In this tutorial we are going to learn how to use `tf.data` to build input pipelines.

We will go first over a few simple examples where we build a series of input pipelines. Then we will just loop over them, printing the items in order to understand the effect of the transformation we include. Here we will see how to create pipelines from arrays in memory, from python generators and files. We will also see how to interleave data that's stored on multiple files and how to distribute it over multiple compute nodes.

For this part, go over the following notebooks:
 1. [Building input pipelines from data in memory](1_getting_started_with_tensorflows_dataset_api.ipynb) 
 2. [Building input pipelines from data stored in single or multiple files](2_getting_started_with_tensorflows_dataset_api.ipynb)
 3. [Building input pipelines from user-defined generators](3_getting_started_with_tensorflows_dataset_api.ipynb)
 4. [Building input pipelines for distributed training (Horovod)](4_getting_started_with_tensorflows_dataset_api.ipynb)
 5. [Building input pipelines for distributed training (Horovod)](5_getting_started_with_tensorflows_dataset_api.ipynb)
 6. [Building input pipelines for distributed training (`tf.distribute`)](5_getting_started_with_tensorflows_dataset_api.ipynb)

> Remember to restart the kernel after moving from a notebook to the other in order to free the GPU.

`tf.data` provides a number of optimizations that can be applied to the input pipeline to avoid having the GPUs idle waiting from data. Many of such of optimization involve starting asynchronous threads that populate a buffer with data ready to be sent to the GPU before it's needed. In the next notebook we simulate a data reader that will allow us to visualize in a timeline diagram the effect of some of such optimizations:
 7. [Understanding input pipeline optimizations](pipeline-timeline.ipynb)


Next we are going to learn to write and read TFRecord files. TFRecord is the recommended format for storing data to be used with TensorFlow:
  8. [Reading and writing TFRecord files](tfrecords/read_and_write_TFRecord_files.ipynb)
  9. [Converting MNIST dataset to TFRecord file](tfrecords/convert-MNIST-dataset-to-tfrecords.py)
  10. [Decoding a TFRecord file from ImageNet](tfrecords/decoding-imagenet.ipynb)
 
A full example including building an input pipeline and training a model is shown in the last notebook:
 11. [MNIST example](mnist)

