module purge
module load intel/16 intel-mpi/5

mpiicc helloWorld.c -o helloWorld
