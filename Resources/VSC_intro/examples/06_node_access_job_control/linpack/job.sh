#!/bin/bash
#SBATCH -J hpl
#SBATCH -N 1
#SBATCH --ntasks-per-node=16

module purge
module load intel/17
module load intel-mpi/2017
module load intel-mkl/2017

mpirun -np 16 ./xhpl
