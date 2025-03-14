module purge
module load intel/16 intel-mpi/5 hdf5/1.8.18-MPI

cp $VSC_HDF5_ROOT/share/hdf5_examples/c/ph5example.c .
mpiicc -lhdf5 ph5example.c -o ph5example
