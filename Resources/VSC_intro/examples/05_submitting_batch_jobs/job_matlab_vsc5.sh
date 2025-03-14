#!/bin/bash

#SBATCH -J matlab
#SBATCH -N 1
#SBATCH --qos=zen3_0512           # use a qos
#SBATCH --partition=zen3_0512     # use partition that fits to the qos
#SBATCH --tasks-per-node=128      # SLURM_NTASKS_PER_NODE  [1 mpi/core]

module purge                      # recommended to be done in all jobs !!!!!
# module load <modules>           # load only modules actually needed by job
#module load Matlab/v9.13_REL2022b
module load Matlab/R2024b

echo 
echo 'Hello from node: '$HOSTNAME
echo 'Number of nodes: '$SLURM_JOB_NUM_NODES
echo 'Tasks per node:  '$SLURM_TASKS_PER_NODE
echo 'Partition used:  '$SLURM_JOB_PARTITION
echo 'QOS used:        '$SLURM_JOB_QOS
echo 'Using the nodes: '$SLURM_JOB_NODELIST
echo 
# <do_my_work>
echo 'Using matlab to calculate "2+2"'
echo 
echo "2+2" | matlab
