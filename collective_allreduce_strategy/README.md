# Running TensorFlow's Collective Allreduce Strategy on Piz Daint

The Collective Allreduce Strategy is TensorFlow's own implementation of the ring-allreduce for training.
As for TensorFlow-1.12.0, the Collective Allreduce Strategy is still under development and it's documentation
points out that the API may change. The script [setup.sh](setup.sh) takes care of the necessary setup to run
TensorFlow calculations using the Collective Allreduce Strategy.

This example uses fake data generated with Numpy to ilustrate the use of the Collective Allreduce Strategy.

### Running the example
```
#!/bin/bash -l

#SBATCH --job-name=test_tf
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --constraint=gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load daint-gpu
module load TensorFlow/1.12.0-CrayGNU-18.08-cuda-9.1-python3

bash setup.sh
srun -u --multi-prog config_tf
```