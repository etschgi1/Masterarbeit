
#define _GNU_SOURCE
#include <sched.h>                                  // sched_getcpu()
#include <mpi.h>                                    // MPI header
#include <stdio.h>

int main(int argc, char *argv[])
{

int rank = 0,      size = 1;                        // MPI - initialized
int core_id;                                        // ... core_id
int namelen;                                        // ... MPI processor_name
char name[MPI_MAX_PROCESSOR_NAME];                  // ... MPI processor_name

MPI_Init(&argc, &argv);                             // MPI initialization

MPI_Comm_rank(MPI_COMM_WORLD, &rank);               // MPI rank
MPI_Comm_size(MPI_COMM_WORLD, &size);               // MPI size

MPI_Get_processor_name(name, &namelen);             // ... MPI processor_name
core_id = sched_getcpu();                           // ... core_id

if (rank == 0)
{
printf ("he-mpi = MPI program prints core_id & node_name (cb)\n");
printf ("Hello world! -Running with %4i MPI processes\n", size);
}

printf ("MPI process %4i / %4i ON core %4i of node %s\n", rank, size, core_id, name);

MPI_Finalize();                                     // MPI finalization

}
