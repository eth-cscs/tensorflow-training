# Currently (TensorFlow-1.12.0), to use the collective allreduce stratedy with TensorFlow, it is necessary
# to define the environment variable `TF_CONF` for each of the workers
# TF_CONFIG='{
#     "cluster": {
#         "worker": ["host1:port", "host2:port", "host3:port"],
#         "ps": ["host4:port", "host5:port"]
#     },
#    "task": {"type": "worker", "index": worker_index}
# }'
# Since `TF_CONF` contains the worker_index (0, 1, 2, ...), it is necessary to pass to the srun command
# a different value of the variable for each of the workes. With slurm this is possible only with
# the --multi-prog command line option of srun. This requires a bit of setup before submitting the job
# 
# This scripts takes care of the setup necessary for a --multi-prog run.
# Running `./setup.sh` creates the file `run_tf.sh` which for two nodes looks like this
#
# TF_CONFIG="{\"cluster\": {\"worker\": [\"IP_node_1:8000\",\"IP_node_2:8000\"]},
#             \"task\": {\"type\": \"worker\", \"index\": $1}}" python $2
#
# and the file `tf_config` which for two nodes looks like this
#
# 0-1 bash run_tf.sh %t collective_allreduce_strategy_keras.py
#
# The `%t` is a template value which will take the value of the index of the worker automatically, so
# `TF_CONF` can be defined properly for each worker.
#
# After running `./setup.sh`, the job can be submited with
# srun -u --multi-prog config_tf
#
# A Slurm script should look like this
#
# !/bin/bash -l
# #SBATCH --job-name=collective_allreduce_strategy
# #SBATCH --time=00:05:00
# #SBATCH --nodes=2
# #SBATCH --ntasks-per-core=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=12
# #SBATCH --partition=normal
# #SBATCH --constraint=gpu
# 
# module load daint-gpu
# module load Horovod/0.15.2-CrayGNU-18.08-tf-1.12.0
# 
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# 
# ./setup.sh
# srun -u --multi-prog config_tf


# it gets something like this
# IPs="148.187.48.213 148.187.48.214"
IPs=`srun hostname -i; wait`
workers=`for ip in $IPs ; do
	echo -n '\"'${ip}:8000'\" '  
done`

workers=`echo $workers | sed 's/ /,/g'`

# echo "{\"cluster\": {\"worker\": [$workers]}, \"task\": {\"type\": \"worker\", \"index\": ARG}}" | sed 's/ARG/$1/g'

# The following should echo something like this:
# TF_CONFIG="{\"cluster\": {\"worker\": [\"148.187.48.213:8000\", \"148.187.48.214:8000\"]},
#            \"task\": {\"type\": \"worker\", \"index\": $1}}" python collective_allreduce_strategy_keras.py

echo    "TF_CONFIG=\"{\\\"cluster\\\": {\\\"worker\\\": [$workers]}," > run_tf.sh
echo -n "            \\\"task\\\": {\\\"type\\\": \\\"worker\\\", \\\"index\\\": \$1}}\" " >> run_tf.sh

echo "python collective_allreduce_strategy_keras.py" >> run_tf.sh


echo "0-$(($SLURM_NNODES - 1)) bash run_tf.sh %t" > config_tf
