#!/bin/bash
#SBATCH -J twocores
#SBATCH -N 2
#SBATCH -p asperitas
#SBATCH -o slurm-twocores-%j.out
##SBATCH --tasks-per-node=4
##SBATCH --threads-per-core=2

module purge
module load intel/16 intel-mpi/5

export I_MPI_PIN_PROCESSOR_LIST=0,1
mpirun -np $SLURM_NTASKS ./helloWorld |sort -k 2 -n 
