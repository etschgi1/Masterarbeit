#!/bin/bash

#SBATCH -J devel
#SBATCH -N 1
#SBATCH --qos=zen3_0512_devel     # use the development qos  
#SBATCH --partition=zen3_0512     # use partition that fits to the qos
#SBATCH --tasks-per-node=128      # SLURM_NTASKS_PER_NODE  [1 mpi/core]


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
# <do_my_work>
hostname
free
