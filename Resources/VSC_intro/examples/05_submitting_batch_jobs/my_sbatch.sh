#!/bin/bash

#SBATCH -J testN
#SBATCH -N 1
#SBATCH --qos=zen3_0512_devel     # use a qos
#SBATCH --partition=zen3_0512     # use partition that fits to the qos
#SBATCH --tasks-per-node=128      # SLURM_NTASKS_PER_NODE  [1 mpi/core]
#SBATCH --time=1:00

module purge                      # recommended to be done in all jobs !!!!!
# module load <modules>           # load only modules actually needed by job

echo
echo 'Hello from node: '$HOSTNAME
hostname 
free