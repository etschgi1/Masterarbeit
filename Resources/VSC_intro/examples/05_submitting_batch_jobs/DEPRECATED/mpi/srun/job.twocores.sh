#!/bin/bash
#SBATCH -J twocores
#SBATCH -N 2
#SBATCH -o slurm-twocores-%j.out
#SBATCH -p asperitas
##SBATCH --tasks-per-node=4
##SBATCH --threads-per-core=2

module purge
module load intel/16 intel-mpi/5

srun --cpu_bind=map_cpu:0,1 ./helloWorld |sort -k 2 -n 
