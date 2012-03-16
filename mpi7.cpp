// "derived types" 

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "mpierr.h"
#include <unistd.h> 

int main( int argc, char** argv ) {

    int numtasks = 0; 
    int task     = 0;
    const int NELEMENTS = 6;
    MPI_( MPI_Init( &argc, &argv ) );
    MPI_( MPI_Errhandler_set( MPI_COMM_WORLD, MPI_ERRORS_RETURN ) );
    MPI_( MPI_Comm_size( MPI_COMM_WORLD, &numtasks ) );
    MPI_( MPI_Comm_rank( MPI_COMM_WORLD, &task  ) );
    std::vector< char > nodeid( MPI_MAX_PROCESSOR_NAME, '\0' );
    int len = 0;
    MPI_( MPI_Get_processor_name( &nodeid[ 0 ], &len ) );
    
    MPI_Status stat;
    MPI_Datatype indextype;    

    float a[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 11, 12, 13, 14, 15 };
    std::vector< float > b( NELEMENTS );

    std::vector< int > blocklengths( 2 );
    std::vector< int > displacements( 2 );

    blocklengths[ 0 ]  = 4;
    blocklengths[ 1 ]  = 2;
    displacements[ 0 ] = 5;
    displacements[ 1 ] = 12;

    MPI_( MPI_Type_indexed( 2, &blocklengths[ 0 ], &displacements[ 0 ], MPI_FLOAT, &indextype ) );
    MPI_( MPI_Type_commit( &indextype ) );
       
    const int tag = 1;
    if( task == 0 ) {
        // need to use async send if we want rank 0 to receive data
        // without Isend rank 0 waits indefinitely for data to be delivered (through a matching Recv)
        // to rank 0 which will never happen since the Recv is reached only after Send returns
        MPI_Request r;
        for( int i = 0; i != numtasks; ++i ) {
            MPI_( MPI_Isend( &a[ 0 ], 1, indextype, i, tag, MPI_COMM_WORLD, &r ) );
        }
    } 
    const int source = 0;
    MPI_( MPI_Recv( &b[ 0 ], NELEMENTS, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &stat ) );
   
    // reason for using a string stream: with concurrent output it is better to have
    // each MPI task output a single string to avoid possible interleaves 
    std::ostringstream os;
    os << &nodeid[ 0 ] << " - rank " << task << ":\t";
    std::copy( b.begin(), b.end(), std::ostream_iterator< float >( os, "," ) );
    os << '\n';
    std::cout << os.str(); os.flush();
        
    MPI_( MPI_Type_free( &indextype ) );
    MPI_( MPI_Finalize() );
    
    return 0;
}


