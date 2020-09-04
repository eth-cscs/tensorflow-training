# CNNs with `keras-applications`
Here we consider the distributed training of a CNN model from `keras-applications` for image classification.

These are simple examples intended only to show how to add the code for distributed training from data stored as a set of TFRecord files. The dataset is a randomly drawn subset of ImageNet of about 120000 images.

 * [Exercise: Writing a decode for an Imagenet TFRecord file](00_decoding-imagenet.ipynb)
 * [Training Inception on Imagenet (single-node)](01_inceptionv3_tfr.ipynb)
 * [Distributed training Inception on Imagenet with Horovod (1)](02_inceptionv3-hvd_tfr.ipynb)
 * [Distributed training Inception on Imagenet (`tf.distributed)`](04_inceptionv3-tf.dist.ipynb)
 * [Distributed training Inception on Imagenet with Horovod (`dataset.shard`/interleave)](05_inceptionv3-hvd_tfr.ipynb)
 * [Distributed training Inception on Imagenet with Horovod (sharding files by hand/interleave)](06_inceptionv3-hvd_tfr.ipynb)
