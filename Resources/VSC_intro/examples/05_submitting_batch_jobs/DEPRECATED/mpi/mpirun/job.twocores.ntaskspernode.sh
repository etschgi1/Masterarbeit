#!/bin/bash
#SBATCH -J twocores-ntaskspernode
#SBATCH -N 2
#SBATCH -p asperitas
#SBATCH -o slurm-twocores-ntaskspernode-%j.out
#SBATCH --tasks-per-node=2

module purge
module load intel/16 intel-mpi/5

export I_MPI_PIN_PROCESSOR_LIST=0,1
mpirun ./helloWorld |sort -k 2 -n 
