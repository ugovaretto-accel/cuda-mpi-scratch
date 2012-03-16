// "async" invocation

#include <mpi.h>
#include <iostream>
#include <sstream>
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
    std::vector< char > nodeid( MPI_MAX_PROCESSOR_NAME, '\0' );
    int len = 0;
    MPI_( MPI_Get_processor_name( &nodeid[ 0 ], &len ) );
    
    const int NUM_REQS = 4;
    std::vector< MPI_Request >  req( NUM_REQS );
    std::vector< MPI_Status  > stat( NUM_REQS );

    const int prev_task = task == 0            ?  numtasks - 1 : task - 1;
    const int next_task = task == numtasks - 1 ?  0             : task + 1;
    const int tag_prev = -1;
    const int tag_next = +1;

    // send tasks id to adjacent tasks i.e. task 4 sends its id to task 3 and 5
    MPI_( MPI_Isend( &task, 1, MPI_INT, prev_task, tag_prev, MPI_COMM_WORLD, &req[ 0 ] ) );
    MPI_( MPI_Isend( &task, 1, MPI_INT, next_task, tag_next, MPI_COMM_WORLD, &req[ 1 ] ) );
 
    // receive results from tasks with adjacent id i.e. taks 4 receives ids from 3 and 5
    int prev_id = -1;
    int next_id = -1;
    // previous task id uses 'tag_next' tag to send data to the current (previous' next) task 
    MPI_( MPI_Irecv( &prev_id, 1, MPI_INT, prev_task, tag_next, MPI_COMM_WORLD, &req[ 2 ] ) );
    // next task id uses 'tag_prev' tag to send data to the current (next's previous) task
    MPI_( MPI_Irecv( &next_id, 1, MPI_INT, next_task, tag_prev, MPI_COMM_WORLD, &req[ 3 ] ) );

    // wait for all requests to complete   
    MPI_( MPI_Waitall( req.size(), &req[ 0 ], &stat[ 0 ] ) );

#ifdef SERIALIZE_OUTPUT
    sleep( task );
#endif

    std::ostringstream os;
    os << task << '/' << numtasks - 1 << ":\t" << '(' << prev_id << ',' << next_id << ")\t- " << &nodeid[ 0 ]  << '\n';
    std::cout << os.str(); os.flush();

    MPI_( MPI_Finalize() );

    return 0;
}       
