// "sync" invocation 

#include <mpi.h>
#include <iostream>
#include <vector>
#include "mpierr.h"


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
    if( task == 0 ) {
        const char buf[] = "Hello from rank 0";
        std::vector< char > outmsg0to1( buf, buf + sizeof( buf ) );
        dest   = 1;
        source = 1;
        MPI_( MPI_Send( &outmsg0to1[ 0 ], outmsg0to1.size(), MPI_CHAR, dest, tag0to1, MPI_COMM_WORLD ) );
        MPI_Status status;
        MPI_( MPI_Probe( source, tag1to0, MPI_COMM_WORLD, &status ) );
        int count = 0;
        MPI_( MPI_Get_count( &status, MPI_CHAR, &count ) );
        std::vector< char > inbuffer( count + 1, '\0' ); 
        MPI_( MPI_Recv( &inbuffer[ 0 ], count, MPI_CHAR, source, tag1to0, MPI_COMM_WORLD, &status ) );
        std::cout << "Task 0: " << " received message \"" << &inbuffer[ 0 ] << '"' << std::endl;           
    } else if( task == 1 ) {
        const char buf[] = "Hello from rank 1";
        std::vector< char > outmsg1to0( buf, buf + sizeof( buf ) );
        dest   = 0;
        source = 0;
        MPI_Status status;
        MPI_( MPI_Probe( source, tag0to1, MPI_COMM_WORLD, &status ) );
        int count = 0;
        MPI_( MPI_Get_count( &status, MPI_CHAR, &count ) );
        std::vector< char > inbuffer( count + 1, '\0' ); 
        MPI_( MPI_Recv( &inbuffer[ 0 ], count, MPI_CHAR, source, tag0to1, MPI_COMM_WORLD, &status ) );
        std::cout << "Task 1: " << " received message \"" << &inbuffer[ 0 ] << '"' << std::endl; 
        MPI_( MPI_Send( &outmsg1to0[ 0 ], outmsg1to0.size(), MPI_CHAR, dest, tag1to0, MPI_COMM_WORLD ) );
    }
    
    MPI_( MPI_Finalize() );

    return 0;
}
