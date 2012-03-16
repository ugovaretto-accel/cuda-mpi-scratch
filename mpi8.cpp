// "derived types" 

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "mpierr.h"
#include <unistd.h> 


struct Particle {
    float x, y, z;
    float velocity;
    int id, type;
}; 

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
        
    typedef std::vector< Particle > Particles;

    Particles particles( numtasks );

    // define type 
    MPI_Status stat;
    MPI_Datatype particletype, oldtypes[ 2 ];
    MPI_Aint offsets[ 2 ], extent; // to make it compatible with MPI_Type_extent()
    int blockcount[ 2 ];

    offsets[ 0 ]  = 0; // offset in bytes from start of struct
    oldtypes[ 0 ] = MPI_FLOAT; // 4 floats
    blockcount[ 0 ] = 4;   
    
    //compute offset of first int in struct
    MPI_( MPI_Type_extent( MPI_FLOAT, &extent ) );
    if( task == 0 ) std::cout << "\nMPI_FLOAT extent: " << extent << std::endl;
    offsets[ 1 ] = 4 * extent; 
    oldtypes[ 1 ] = MPI_INT;
    blockcount[ 1 ] = 2;

    MPI_( MPI_Type_create_struct( 2, blockcount, offsets, oldtypes, &particletype ) );
    MPI_( MPI_Type_commit( &particletype ) );

    // send
    const int root = 0;
    const int tag  = 1;
    if( task == root ) {
        for( int i = 0; i != particles.size(); ++i ) {
            particles[ i ].x =  i;
            particles[ i ].y = -i;
            particles[ i ].z =  i;
            particles[ i ].velocity = 0.5;
            particles[ i ].id = i;
            particles[ i ].type = i % 2;
            MPI_Request r;
            MPI_( MPI_Isend( &particles[ i ], 1, particletype, i, tag, MPI_COMM_WORLD, &r ) );
        }
    } 

    // receive    
    Particle particle;
    MPI_( MPI_Recv( &particle, 1, particletype, root, tag, MPI_COMM_WORLD, &stat ) );    

    // reason for using a string stream: with concurrent output it is better to have
    // each MPI task output a single string to avoid possible interleaves 
    std::ostringstream os;
    os << &nodeid[ 0 ] << " - rank " << task << ":\t" << "particle id: " << particle.id;
    os << '\n';
    std::cout << os.str(); os.flush();

    // free        
    MPI_( MPI_Type_free( &particletype ) );
    MPI_( MPI_Finalize() );
    
    return 0;
}


