#!/bin/bash
#SBATCH -J h5test
#SBATCH -N 1
#SBATCH --tasks-per-node=48         # SLURM_NTASKS_PER_NODE  [1 mpi/core: 48/96]

module purge
module load hdf5/1.10.5-gcc-9.1.0-qvix74n
module load intel-mpi/2019.7
module load gcc/9.1.0-gcc-4.8.5-mj7s6dg

VSC_HDF5_ROOT=/opt/sw/spack-0.12.1/opt/spack/linux-centos7-x86_64/gcc-9.1.0/hdf5-1.10.5-qvix74ns2mkwk7juot6oak6aout4h5wh
cp $VSC_HDF5_ROOT/share/hdf5_examples/c/ph5example.c .

mpicc -lhdf5 ph5example.c -o ph5example

mpirun -np 8  ./ph5example -c -v
