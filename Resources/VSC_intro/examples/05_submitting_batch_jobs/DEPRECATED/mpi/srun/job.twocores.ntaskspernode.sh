#!/bin/bash
#SBATCH -J twocoresntaskpernode
#SBATCH -N 2
#SBATCH -o slurm-twocores-ntaskpernode-%j.out
#SBATCH -p asperitas
#SBATCH --tasks-per-node=2

module purge
module load intel/16 intel-mpi/5

srun --cpu_bind=map_cpu:0,1 ./helloWorld |sort -k 2 -n 
