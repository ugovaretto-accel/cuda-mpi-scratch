// "collective" 
// same as mpi5.cpp but at the end each task sends its id to the root task
// through a gather operation 

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
        MPI_(MPI_Isend( &task, 1, MPI_INT, prev_task, send_left_tag, 
             MPI_COMM_WORLD, &r));
        req.push_back(r);
        stat.push_back(s);
    }
    if(next_task < numtasks) {
        MPI_Request r;
        MPI_Status s;
        MPI_(MPI_Isend( &task, 1, MPI_INT, next_task, send_right_tag, 
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
    std::vector< int > neighbor( 3, task ); //initialize all element to current
                                            //task id, second and third
                                            //elements will be overwritten
                                            //with task ids of neighbors                                         
    if(prev_task >= 0) {
        MPI_Request r;
        MPI_Status s;
        // left task used send_right tag when data sent
        MPI_(MPI_Irecv(&neighbor[ 1 ], 1, MPI_INT, prev_task, send_right_tag, 
             MPI_COMM_WORLD, &r ) );
        req.push_back(r);
        stat.push_back(s);
    }
    if(next_task < numtasks) {
        MPI_Request r;
        MPI_Status s;
        // right task used send_left tag when data sent
        MPI_(MPI_Irecv(&neighbor[ 2 ], 1, MPI_INT, next_task, send_left_tag,
             MPI_COMM_WORLD, &r));
        req.push_back(r);
        stat.push_back(s);
    }
   
    // wait for all requests to complete   
    MPI_(MPI_Waitall( req.size(), &req[ 0 ], &stat[ 0 ]));

    const int root = 0;
    const int ELEMENTS_SENT_PER_TASK     = 3;
    const int ELEMENTS_RECEIVED_PER_TASK = 3;
    const int TOTAL_ELEMENTS_TO_GATHER   = numtasks * 
                                           ELEMENTS_RECEIVED_PER_TASK;
    // show how to deal with const pointers
    const std::vector< int > cneighbor( neighbor); 
    if(task == root) {
        std::vector< int > neighbors( TOTAL_ELEMENTS_TO_GATHER, -2);
        MPI_(MPI_Gather(const_cast< int* >(&cneighbor[ 0 ]), 
             ELEMENTS_SENT_PER_TASK, MPI_INT, &neighbors[ 0 ],
             ELEMENTS_RECEIVED_PER_TASK, MPI_INT, root, MPI_COMM_WORLD));
        for(std::vector< int >::const_iterator i = neighbors.begin();
             i < neighbors.end(); i += ELEMENTS_RECEIVED_PER_TASK ) {
            std::cout << '(' << *( i + 1 ) << '<' << *( i ) << '>' 
                      << *( i + 2 ) << ") ";
        }
        std::cout << std::endl;
    } else {
        MPI_(MPI_Gather( const_cast< int* >(&cneighbor[ 0 ]), 
             ELEMENTS_SENT_PER_TASK, MPI_INT, 0, 0, 0, root, MPI_COMM_WORLD));
    } 

    MPI_(MPI_Finalize());

    return 0;
}       
