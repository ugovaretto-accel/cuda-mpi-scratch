//Basic init/shutdown

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main( int argc, char** argv ) {
    int rank, size, len;
    char nid[MPI_MAX_PROCESSOR_NAME];

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Get_processor_name( nid, &len );
    printf( "Hello world from process %d of %d -- Node ID = %s\n", rank, size, nid );
    MPI_Finalize();
    return 0;
}
