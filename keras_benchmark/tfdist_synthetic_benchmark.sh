#!/bin/bash -l
 
#SBATCH --job-name=tfTfDistTest
#SBATCH --time=00:15:00
#SBATCH --switches=1
#SBATCH --nodes=4
#SBATCH --output=tfTfDistTest004.out
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --partition=normal

conda activate /apps/daint/UES/6.0.UP04/sandboxes/sarafael/miniconda-tf2.3
srun which python

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python -u tfdist_synthetic_benchmark.py --model ResNet101 --batch-size 64
