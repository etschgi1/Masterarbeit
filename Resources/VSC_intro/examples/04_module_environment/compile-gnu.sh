module purge
module load gcc/5.3 intel-mpi/5 hdf5/1.8.18-MPI

cp $VSC_HDF5_ROOT/share/hdf5_examples/c/ph5example.c .
mpicc -lhdf5 ph5example.c -o ph5example
