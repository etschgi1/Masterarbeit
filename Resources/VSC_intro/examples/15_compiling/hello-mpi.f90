PROGRAM HelloWorld
INCLUDE 'mpif.h'
INTEGER my_rank, p
INTEGER source, dest, tag
INTEGER ierr, status(MPI_STATUS_SIZE)
CALL MPI_Init(ierr)
CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr)
CALL MPI_Comm_size(MPI_COMM_WORLD, p, ierr)
WRITE(*,FMT="(A,I4)") "Hello world from process ", my_rank
CALL MPI_Finalize(ierr)
END PROGRAM HelloWorld
