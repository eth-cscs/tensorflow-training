# Examples with Custom Estimators

Training Inception-V3, ResNet-101 and VGG16 on ImageNet and logging performance. The scripts starting with `hvd` provide the Horovod distributed implementation.

The script [get_models_from_tfbenchmarks.sh](get_models_from_tfbenchmarks.sh) downloads the necessary files to reuse the models Inception-V3, ResNet-101 and VGG16 defined on [Tensorsflow's benchmark repository](https://github.com/tensorflow/benchmarks).


### Running the examples

The modules with the definition of the models can be downloaded with:
```
bash get_models_from_tfbenchmarks.sh
```

This will create the folder `models_from_benchmark`. The scripts need to be ran from inside the folder.
```
cd models_from_benchmark
cp ../estimator_inceptionv3_imagenet.py .
```

The examples can be run with the following script
```
#!/bin/bash -l

#SBATCH --job-name=inceptionv3_estimator
#SBATCH --time=00:15:00
#SBATCH --nodes=<num-nodes>
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu

module load daint-gpu
module load Horovod/0.16.0-CrayGNU-18.08-tf-1.12.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Environment variables needed by the NCCL backend
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1

srun python estimator_inceptionv3_imagenet
```