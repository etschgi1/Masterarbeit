#!/bin/bash
#SBATCH -J default
#SBATCH -N 2
#SBATCH -p asperitas
#SBATCH -o slurm-default-%j.out
##SBATCH --tasks-per-node=4
##SBATCH --threads-per-core=2

module purge
module load intel/16 intel-mpi/5

mpirun -np $SLURM_NTASKS ./helloWorld |sort -k 2 -n 
