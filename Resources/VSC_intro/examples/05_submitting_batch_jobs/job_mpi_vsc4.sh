#!/bin/bash

#SBATCH -J mpi                       # SLURM_JOB_NAME
#SBATCH -N 2                         # SLURM_JOB_NUM_NODES
#SBATCH --qos=skylake_0096           # use a qos
#SBATCH --partition=skylake_0096     # use partition that fits to the qos
#SBATCH --tasks-per-node=48          # SLURM_NTASKS_PER_NODE  [1 mpi/core: 48/96]

module purge                         # recommended to be done in all jobs !!!!!
# module load <modules>              # load only modules actually needed by job
module load intel intel-mpi          ### load the Intel compiler and Intel-MPI
export I_MPI_PIN_RESPECT_CPUSET=0    ### now needed on vsc4 with latest Intel-MPI
export I_MPI_PIN_PROCESSOR_LIST=0-47 ### take care of pinning for pure MPI explicitly

echo 
echo 'Hello from node: '$HOSTNAME
echo 'Number of nodes: '$SLURM_JOB_NUM_NODES
echo 'Tasks per node:  '$SLURM_TASKS_PER_NODE
echo 'Partition used:  '$SLURM_JOB_PARTITION
echo 'QOS used:        '$SLURM_JOB_QOS
echo 'Using the nodes: '$SLURM_JOB_NODELIST
echo 
# <do_my_work>
#
# using the example from Compiling:
#
# mpiicc --version
# cp ../15_compiling/hello-mpi.c .
# mpiicc -O3 -xHost hello-mpi.c -o hello-mpi_c
# mpirun -np 96 ./hello-mpi_c | sort -k2
#
# using an example from the MPI+X course:
#

which mpiicc
which mpirun
ulimit -l

mpiicc -O3 -xHost -o he-mpi he-mpi.c
mpirun -np 96 ./he-mpi | sort -n | cut -c 1-54
