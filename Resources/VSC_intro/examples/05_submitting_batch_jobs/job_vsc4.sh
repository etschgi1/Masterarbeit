#!/bin/bash

#SBATCH -J testN
#SBATCH -N 1
#SBATCH --qos=skylake_0096        # use a qos
#SBATCH --partition=skylake_0096  # use partition that fits to the qos
#SBATCH --tasks-per-node=48       # SLURM_NTASKS_PER_NODE  [1 mpi/core: 48/96]


module purge                      # recommended to be done in all jobs !!!!!
# module load <modules>           # load only modules actually needed by job

echo 
echo 'Hello from node: '$HOSTNAME
echo 'Number of nodes: '$SLURM_JOB_NUM_NODES
echo 'Tasks per node:  '$SLURM_TASKS_PER_NODE
echo 'Partition used:  '$SLURM_JOB_PARTITION
echo 'QOS used:        '$SLURM_JOB_QOS
echo 'Using the nodes: '$SLURM_JOB_NODELIST
echo 
sleep 30 # <do_my_work>
