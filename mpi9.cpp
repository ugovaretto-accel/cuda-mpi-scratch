// "groups" 

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "mpierr.h"
#include <unistd.h> 


// only available in origianl SGI STL implementation
template < typename ForwardIterator, typename T >
void Iota( ForwardIterator begin, ForwardIterator end, T value ) {
    for( ; begin != end; ++begin, ++value ) *begin = value;    

}  


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
        
    // extract the original group handle
    MPI_Group orig_group, new_group;
    MPI_( MPI_Comm_group( MPI_COMM_WORLD, &orig_group ) );
    std::vector< int > first_group_ranks( numtasks / 2, -1 );
    std::vector< int > second_group_ranks( numtasks / 2, -1 );
    Iota( first_group_ranks.begin(),  first_group_ranks.end(),  0 );
    Iota( second_group_ranks.begin(), second_group_ranks.end(), numtasks / 2 );
    // divide tasks into two groups: each call to MPI_Group_incl
    // generates a new group with a new set of rank ids
    if( task < numtasks / 2 ) {
        // put all tasks with rank < numtasks/2 in the same group
        MPI_( MPI_Group_incl( orig_group, numtasks / 2, &first_group_ranks[ 0 ], &new_group ) );
    } else {
        // put all tasks with rank < numtasks/2 in the same group
        MPI_( MPI_Group_incl( orig_group, numtasks / 2, &second_group_ranks[ 0 ], &new_group ) );
    }

    // create a new communicator for exchanging messages within each sub-group
    MPI_Comm new_comm;
    MPI_( MPI_Comm_create( MPI_COMM_WORLD, new_group, &new_comm ) );      
    int new_rank = -1;
    MPI_( MPI_Group_rank( new_group, &new_rank ) );

    int sendbuf = task;
    int recvbuf = -1;
    // perform all-reduce separately on the two groups...
    MPI_( MPI_Allreduce( &sendbuf, &recvbuf, 1, MPI_INT, MPI_SUM, new_comm ) );
    // ...and on the entire task pool
    int recvbuf_total = -1;
    MPI_( MPI_Allreduce( &sendbuf, &recvbuf_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ) );
    
    const int group_id = task < numtasks / 2 ? 0 : 1;
    // reason for using a string stream: with concurrent output it is better to have
    // each MPI task output a single string to avoid possible interleaves 
    std::ostringstream os;
    os << &nodeid[ 0 ] << " - group: " << group_id << " - rank: " << task << "\tnew rank: " << new_rank << "\treceived: " << recvbuf;
    os << '\n';
    std::cout << os.str(); os.flush(); 

    if( task == 0 ) {
        std::ostringstream os;
        os << "\nAllreduce total: " << recvbuf_total;
        os << '\n';
        std::cout << os.str(); os.flush();     
    }

    MPI_( MPI_Comm_free( &new_comm ) );
    MPI_( MPI_Group_free( &new_group ) );
    MPI_( MPI_Group_free( &orig_group ) );

    MPI_( MPI_Finalize() );
    
    return 0;
}


