// "async" invocation
// each task sends its id to its neighbours

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
    
    std::vector< MPI_Request >  req;
    std::vector< MPI_Status  > stat;

    const int prev_task = task - 1;
    const int next_task = task + 1;
    const int send_right_tag = 0x01;
    const int send_left_tag  = 0x10;
    // send tasks id to adjacent tasks i.e. task 4 sends its id to task 3 and 5
    // only add request and status to array for 
    if(prev_task >= 0) {
        MPI_Request r;
        MPI_Status s;
        MPI_(MPI_Isend(&task, 1, MPI_INT, prev_task, send_left_tag, 
             MPI_COMM_WORLD, &r));
        req.push_back(r);
        stat.push_back(s);
    }
    if(next_task < numtasks) {
        MPI_Request r;
        MPI_Status s;
        MPI_(MPI_Isend(&task, 1, MPI_INT, next_task, send_right_tag, 
             MPI_COMM_WORLD, &r));
        req.push_back(r);
        stat.push_back(s);
    }
    // receive results from tasks with adjacent id i.e. 
    // task 4 receives ids from 3 and 5, tag is the one used when sending
    // task 4 uses tag send_left when receiving from 5 since task 5 used 
    // tag send_left when sending data to the left (4), similarly task 4 uses
    // tag send_right when receiving from 3 since 3 used send_right tag when
    // sending data to the right (4)
    int prev_id = -1;
    int next_id = -1;
    if(prev_task >= 0) {
        MPI_Request r;
        MPI_Status s;
        // left task used send_right tag when data sent
        MPI_(MPI_Irecv(&prev_id, 1, MPI_INT, prev_task, send_right_tag, 
             MPI_COMM_WORLD, &r ) );
        req.push_back(r);
        stat.push_back(s);
    }
    if(next_task < numtasks) {
        MPI_Request r;
        MPI_Status s;
        // right task used send_left tag when data sent
        MPI_(MPI_Irecv(&next_id, 1, MPI_INT, next_task, send_left_tag,
             MPI_COMM_WORLD, &r));
        req.push_back(r);
        stat.push_back(s);
    }
   
    // wait for all requests to complete   
    MPI_(MPI_Waitall( req.size(), &req[ 0 ], &stat[ 0 ]));

    std::ostringstream os;
    os << task << '/' << numtasks - 1 << ":\t" << '(' << prev_id << ", " 
       << task << ", " << next_id << ")\t- " << &nodeid[ 0 ]  << '\n';
    std::cout << os.str(); os.flush();

    MPI_( MPI_Finalize() );

    return 0;
}       
