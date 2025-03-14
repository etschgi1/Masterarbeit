#!/bin/bash
#SBATCH -J default
#SBATCH -N 2
#SBATCH -o slurm-default-%j.out
#SBATCH -p asperitas

module purge
module load intel/16 intel-mpi/5

srun ./helloWorld |sort -k 2 -n 
