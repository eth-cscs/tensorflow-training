# Multi-GPU training with TensorFlow on Piz Daint

 * [Slides](slides)
 * [Visualizing SGD and distributed SGD](SGD)
 * [Getting started with TensorFlow's input pipelines and TFRecord files](input_pipelines)
 * [Distributed training: Image classification with ImageNet](imagenet)
 * [Distributed training: Fine-tuning BERT for text extraction](nlp_squad)

# Notes

Please, clone this repo on your `$SCRATCH`.
Since Jupyter's file explorer can not access directories above your `$HOME`, to access the notebooks from https://jupyter.cscs.ch it's necessary to create a symlink from `$HOME` to `$SCRATCH`
```bash
cd
ln -s $SCRATCH scratch
```

# Jupyter kernel

We have provided a TensorFlow installation to run the Jupyter notebooks of this repository.
It can be accessed in Jupyter through the following kernels.
Please, create the file `$HOME/.local/share/jupyter/kernels/tf-multigpu/launcher` containing the following
```bash
#!/usr/bin/env bash

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1

export TENSORBOARD_PROXY_URL=https://$USER.jupyter.cscs.ch/hub/user-redirect/proxy/%PORT%/

export HOROVOD_TIMELINE=$HOME/timeline.json

export PYTHONPATH=''
if [ "$SOURCE_JUPYTERHUBENV" == true ]; then
    source $HOME/.jupyterhub.env
fi

source /apps/daint/UES/6.0.UP04/sandboxes/sarafael/miniconda-tf2.3/bin/activate
/apps/daint/UES/6.0.UP04/sandboxes/sarafael/miniconda-tf2.3/bin/python -m ipykernel_launcher $@
```
and make it executable
```bash
chmod a+x $HOME/.local/share/jupyter/kernels/tf-multigpu/launcher
```

Create file `$HOME/.local/share/jupyter/kernels/tf-multigpu/kernel.json` with containing the following (make sure of replacing `<username>` byt you user name)
```json
{
 "display_name": "tf-multigpu",
 "language": "python",
 "argv": [
 "/users/<username>/.local/share/jupyter/kernels/tf-multigpu/launcher",
 "-f",
 "{connection_file}"
 ]
}
```
Now both your Jupyter launcher and kernel dropdown menu will be showing the kernel `tf-multigpu`. Make sure it is selected before running a notebook.

# Datasets

For the CNN examples we use the Imagenet dataset. We provide a subset of it that can be used to run the example. You can copy it on your `$SCRATCH` from `/project/csstaff/sarafael/imagenet`. Please, note that this is only a small subset of imagenet and it's not enough for real applications.

For the rest of the examples the datasets are downloaded on the fly.
