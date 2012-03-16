// "sync" invocation 2

#include <mpi.h>
#include <iostream>
#include <vector>
#include "mpierr.h"
#include <unistd.h> 

int main( int argc, char** argv ) {

    int numtasks = 0;
    int task     = 0;
    MPI_( MPI_Init( &argc, &argv ) );
    MPI_( MPI_Errhandler_set( MPI_COMM_WORLD, MPI_ERRORS_RETURN ) );
    MPI_( MPI_Comm_size( MPI_COMM_WORLD, &numtasks ) );
    MPI_( MPI_Comm_rank( MPI_COMM_WORLD, &task  ) );
    int dest   = 0;
    int source = 0;
    const int tag0to1 = 0x01;
    const int tag1to0 = 0x10;
    int k = 0;
    const int kmax = 10;
    if( task == 0 ) std::cout << "\nRank 0\tRank 1\n" << std::endl; 
    while( k != kmax ) {
        if( task == 0 ) {
            dest   = 1;
            source = 1;
            ++k;
            MPI_Status status;
            std::cout << '\r' << k; std::cout.flush();
            sleep( 1 );
            MPI_( MPI_Send( &k, 1, MPI_INT, dest,   tag0to1, MPI_COMM_WORLD ) );
            MPI_( MPI_Recv( &k, 1, MPI_INT, source, tag1to0, MPI_COMM_WORLD, &status ) );
         } else if( task == 1 ) {
            dest   = 0;
            source = 0;
            MPI_Status status;
            MPI_( MPI_Recv( &k, 1, MPI_INT, source, tag0to1, MPI_COMM_WORLD, &status ) );
            ++k;
            std::cout << "\r\t" << k; std::cout.flush();
            sleep( 1 );
            MPI_( MPI_Send( &k, 1, MPI_INT, dest, tag1to0, MPI_COMM_WORLD ) );
         }
    }
  
    if( task == 0 ) {
         std::cout << "\n\nTotal: " << k << std::endl;
    }  
  
    MPI_( MPI_Finalize() );

    return 0;
}
