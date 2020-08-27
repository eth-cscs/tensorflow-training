# Examples with `keras-applications`
Here we consider the distributed training of a CNN model from `keras-applications` for image classification.

This is a simple example intended only to show how to add the code for distributed training from data stored as a set of TFRecord files. The dataset is a randomly drawn subset of ImageNet of about 120000 images.

Before running the python scripts, it's necessary do the following exports:
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1
```
