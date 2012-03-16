// "cartesian grid" 

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "mpierr.h"
#include <unistd.h> 
#include <cmath>


int main( int argc, char** argv ) {

    int numtasks = 0; 
 
    MPI_( MPI_Init( &argc, &argv ) );
    MPI_( MPI_Errhandler_set( MPI_COMM_WORLD, MPI_ERRORS_RETURN ) );
    MPI_( MPI_Comm_size( MPI_COMM_WORLD, &numtasks ) );
    
    const int DIM = int( std::sqrt( double( numtasks ) ) );
    std::vector< int > dims( 2, DIM );
    std::vector< int > periods( 2, 0 ); //periodic - false -> non-periodic
    const int reorder = 0; //false - no reorder
    MPI_Comm cartcomm;
    MPI_( MPI_Cart_create( MPI_COMM_WORLD, 2, &dims[ 0 ], &periods[ 0 ], reorder, &cartcomm ) ); 
    int task = -1;
    MPI_( MPI_Comm_rank( cartcomm, &task ) );
    std::vector< int > coords( 2, -1 );
    MPI_( MPI_Cart_coords( cartcomm, task, 2, &coords[ 0 ] ) );
     
    std::vector< int > neighbors( 4, -1 );
    enum { UP = 0, DOWN, LEFT, RIGHT };
    // compute the shifted source and destination ranks, given a shift direction and amount
    //MPI_Cart_shift is uses to find two "nearby" neighbors of the calling process
    //along a specified direction of an N-dimensional grid
    //The direction and offset are specified as a signed integer
    //If the sign of the displacement is positive the "source" rank is lower
    //than the destination rank; if it's negative the opposite is true 
    MPI_( MPI_Cart_shift( cartcomm, 0, 1, &neighbors[ UP ],   &neighbors[ DOWN ] ) );
    MPI_( MPI_Cart_shift( cartcomm, 1, 1, &neighbors[ LEFT ], &neighbors[ RIGHT ] ) );
    int sendbuf = task;
    const int tag = 0x01;
    std::vector< int > recvbuf( 4, MPI_PROC_NULL ); 
    std::vector< MPI_Request > reqs( 2 * 4 );
    for( int i = 0; i != 4; ++i ) {
        int dest = neighbors[ i ];
        int src  = neighbors[ i ];
        MPI_( MPI_Isend( &sendbuf, 1, MPI_INT, dest, tag, MPI_COMM_WORLD, &reqs[ i ] ) );
        MPI_( MPI_Irecv( &recvbuf[ i ], 1, MPI_INT, src, tag, MPI_COMM_WORLD, &reqs[ i + 4 ] ) );
    }
    std::vector< MPI_Status  > status( 2 * 4 ); 
    MPI_( MPI_Waitall( 8, &reqs[ 0 ], &status[ 0 ] ) );

    std::ostringstream os;
    os << "rank= " << task << " coords= " << coords[ 0 ] << ',' << coords[ 1 ]
       << " neighbors= " << neighbors[ UP ] << ',' << neighbors[ DOWN ] << ','
       << neighbors[ LEFT ] << ',' << neighbors[ RIGHT ] << '\n';
    std::cout << os.str(); os.flush();
 
    MPI_( MPI_Finalize() );
    
    return 0;
}







