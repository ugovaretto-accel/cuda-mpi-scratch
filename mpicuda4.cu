// "mpi + full reduction on gpu + timing" 

#ifdef GPU
#include <cuda.h>
#endif
#include <mpi.h>
#include <iostream>
#include <vector>
#include "mpierr.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <string>
#include <set>
#include <numeric>
#include <ctime>

// switches:
// #GPU : enable GPU computation
// #NO_LOG: do not printout log messages
// #REDUCE_CPU: perform final per-task reduction step on the CPU
// #DOUBLE_: double precision
// #MPI_RROBIN_: assume a round robin layout i.e process 0 -> node 0, process 1 -> node 1 ...
// #NO_GPU_MALLOC_TIME: do not take into account malloc time; usually this is part of an initialization step


// compilation with mvapich2:
// nvcc -L/apps/eiger/mvapich2/1.6/mvapich2-gnu/lib -I/apps/eiger/mvapich2/1.6/mvapich2-gnu/include \
// -libumad -lmpich -lpthread -lrdmacm -libverbs -arch=sm_20 -DGPU \
// ~/projects/gpu-training/trunk/cuda_exercises_ugo/resources/mpiscratch/mpicuda2.cu


// run:
// 1) w/o scheduler: mpiexec -np ... -hosts ... ./a.out
// 2) w/ scheduler: see mpi_cuda_pbs_ref.sh script

// note: when using mvapich2/1.6 and *not* going through the pbs scheduler it seems
//       the default behavior is rrobin, using the pbs launch script the default
//       behavior is "bunch" (as defined by the mvapich2 documentation) 

// note: using single precision floats because that's the only supported type
//       for atomics on CUDA 4

// note: experiment with different number of MPI tasks per GPU/node; using
//       256 Mi floats, 16 MPI tasks on two nodes (8 per node, 4 per GPUs)
//       CUDA fails to allocate memory exaclty for one task on each node;
//       Everything works fine with the same data with 8 tasks (4 per node, 2 per GPU ).

// note: it is possible to implement a discovery step to find the current MPI layout
//       by checking if MPI rank 0 and 1 are on the same processor ("bunch" layout) or
//       not ("scatter" layout)
//
// note: with CUDA 4.0 and architecture 2.0 atomics are available for single precision only!!!



#ifndef DOUBLE_
typedef float real_t;
#define MPI_REAL_T_ MPI_FLOAT
#else
typedef double real_t;
#define MPI_REAL_T_ MPI_DOUBLE
#endif

#ifdef GPU
const int BLOCK_SIZE = 512;

//------------------------------------------------------------------------------
// partial dot product: each thread block generates on value stored into the out array
// final reduction step must be performed on the CPU
__global__ void partial_dot_product_kernel( const real_t* v1, const real_t* v2, int N, real_t* out ) {
    __shared__ real_t cache[ BLOCK_SIZE ];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= N ) return;
    cache[ threadIdx.x ] = 0.f;
    while( i < N ) {
        cache[ threadIdx.x ] += v1[ i ] * v2[ i ];
        i += gridDim.x * blockDim.x;
    }
    __syncthreads();    
    i = BLOCK_SIZE / 2;
    while( i > 0 ) {
        if( threadIdx.x < i ) cache[ threadIdx.x ] += cache[ threadIdx.x + i ];
        __syncthreads();
        i /= 2; //not sure bitwise operations are actually faster
    }
    if( threadIdx.x == 0 ) out[ blockIdx.x ] = cache[ 0 ];
}


//------------------------------------------------------------------------------

//Full on-gpu reduction

// each block atomically increments this variable when done
// performing the first reduction step
__device__ unsigned int count = 0;
// shared memory used by partial_dot and sum functions
// for temporary partial reductions; declare as global variable
// because used in more than one function
__shared__ real_t cache[ BLOCK_SIZE ];

// partial dot product: each thread block produces a single value
__device__ real_t partial_dot( const real_t* v1, const real_t* v2, int N, real_t* out  ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= N ) return real_t( 0 );
    cache[ threadIdx.x ] = 0.f;
    // the threads in the thread block iterate over the entire domain; iteration happens
    // whenever the total number of threads is lower than the domain size
    while( i < N ) {
        cache[ threadIdx.x ] += v1[ i ] * v2[ i ];
        i += gridDim.x * blockDim.x;
    }
    __syncthreads(); // required because later on the current thread is accessing
                     // data written by another thread    
    i = BLOCK_SIZE / 2;
    while( i > 0 ) {
        if( threadIdx.x < i ) cache[ threadIdx.x ] += cache[ threadIdx.x + i ];
        __syncthreads();
        i /= 2; 
    }
    return cache[ 0 ];
}

// sum all elements in array; array size assumed to be equal to number of blocks
__device__ real_t sum( const real_t* v ) {
    cache[ threadIdx.x ] = 0.f;
    int i = threadIdx.x;
    // the threads in the thread block iterate oevr the entire domain
    // of size == gridDim.x == total number of blocks; iteration happens
    // whenever the number of threads in a thread block is lower than
    // the total number of thread blocks
    while( i < gridDim.x ) {
        cache[ threadIdx.x ] += v[ i ];
        i += blockDim.x;
    }
    __syncthreads(); // required because later on the current thread is accessing
                     // data written by another thread        
    i = BLOCK_SIZE / 2;
    while( i > 0 ) {
        if( threadIdx.x < i ) cache[ threadIdx.x ] += cache[ threadIdx.x + i ];
        __syncthreads();
        i /= 2; 
    }
    return cache[ 0 ];
}

// perform parallel dot product in two steps:
// 1) each block computes a single value and stores it into an array of size == number of blocks
// 2) the last block to finish step (1) performs a reduction on the array produced in the above step
// parameters:
// v1 first input vector
// v2 second input vector
// N  size of input vector
// out output vector: size MUST be equal to the number of GPU blocks since it us used
//     for partial reduction; result is at position 0
__global__ void dot_product_full_kernel( const real_t* v1, const real_t* v2, int N, real_t* out ) {
    // true if last block to compute value
    __shared__ bool lastBlock;
    // each block computes a value
    real_t r = partial_dot( v1, v2, N, out );
    if( threadIdx.x == 0 ) {
        // value is stored into output array by first thread of each block
        out[ blockIdx.x ] = r;
        // wait for value to be available to all the threads on the device
        __threadfence();
        // increment atomic counter and retrieve value
        const unsigned int v = atomicInc( &count, gridDim.x );
        // check if last block to perform computation
        lastBlock = ( v == gridDim.x - 1 );
    }
    // the code below is executed by *all* threads in the block:
    // make sure all the threads in the block access the correct value
    // of the variable 'lastBlock'
    __syncthreads();
    
    // last block performs a the final reduction steps which produces one single value
    if( lastBlock ) {
        r = sum( out );
        if( threadIdx.x == 0 ) {
            out[ 0 ] =  r;
            count = 0;
        }
    }   
}
#endif

//------------------------------------------------------------------------------
int main( int argc, char** argv ) {

    int numtasks = 0;
    int task     = 0;
    // INIT ENV
    MPI_( MPI_Init( &argc, &argv ) );
    MPI_( MPI_Errhandler_set( MPI_COMM_WORLD, MPI_ERRORS_RETURN ) );
    MPI_( MPI_Comm_size( MPI_COMM_WORLD, &numtasks ) );
    MPI_( MPI_Comm_rank( MPI_COMM_WORLD, &task  ) );
    std::vector< char > nodeid( MPI_MAX_PROCESSOR_NAME, '\0' );
    int len = 0;
    MPI_( MPI_Get_processor_name( &nodeid[ 0 ], &len ) );

#ifdef MPI_RROBIN_     
    // RETRIEVE TOTAL NUMBER OF NODES USED, is there an easier way ?
    // required to have each GPU assigned to the same number of processes
    // on each node
    const int SEND_NODE_TAG = 0x01;
    //const int SEND_NUM_NODES = 0x10;
    MPI_Request req;
    MPI_( MPI_Isend( &nodeid[ 0 ], MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, SEND_NODE_TAG,
                     MPI_COMM_WORLD, &req ) );     
    int node_count = -1;
    if( task == 0 ) {
        typedef std::set< std::string > NodeCount;
        NodeCount ncount;
        std::vector< char > n( MPI_MAX_PROCESSOR_NAME, '\0' );
        MPI_Status s;
        for( int r = 0; r != numtasks; ++r ) {
            MPI_( MPI_Recv( &n[ 0 ], MPI_MAX_PROCESSOR_NAME, MPI_CHAR, r, SEND_NODE_TAG,
                            MPI_COMM_WORLD, &s ) );   
            ncount.insert( &n[ 0 ] );    
        }
        node_count = int( ncount.size() );
#ifndef NO_LOG
        std::cout << "Number of nodes: " << node_count << std::endl;
#endif 
    }
  
    // SEND INFORMATION USED FOR GPU <-> RANK MAPPING TO EACH PROCESS
    // Option 1: use scatter, useful only to send per-process specific information like e.g
    //           the GPU to use. It is in general a more robust method to have the root process
    //           compute the rank -> gpu map
    //std::vector< int > sendbuf( numtasks, node_count );
    // MPI Scatter parameters: address of send buffer,
    //                         per-receiving process receive buffer size,...
    // send buffer size = num tasks x per-receiving-process buffer size
    //MPI_( MPI_Scatter( &sendbuf[ 0 ],  1, MPI_INT, &node_count, 1, MPI_INT, 0, MPI_COMM_WORLD ) ); 
    // Option 2: simply broadcast the number of nodes
    MPI_( MPI_Bcast( &node_count, 1, MPI_INT, 0, MPI_COMM_WORLD ) );
#endif
    // PER TASK DATA INIT - in the real world this is the place where data are read from file
    // through the MPI_File_ functions or, less likely received from the root process
    const int ARRAY_SIZE = 1024 * 1024 * 256;// * 1024 * 256; // 256 Mi floats x 2 == 2 GiB total storage
    // @WARNING: ARRAY_SIZE must be evenly divisible by the number of MPI processes
    const int PER_MPI_TASK_ARRAY_SIZE = ARRAY_SIZE / numtasks;
    if( ARRAY_SIZE % numtasks != 0  && task == 0 ) {
        std::cerr << ARRAY_SIZE << " must be evenly divisible by the number of mpi processes" << std::endl;
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
    
    std::vector< real_t > v1( ARRAY_SIZE / numtasks, 0. );
    std::vector< real_t > v2( ARRAY_SIZE / numtasks, 0. );
    for( int i = 0; i != PER_MPI_TASK_ARRAY_SIZE; ++i ) {
        v1[ i ] = 1;
        v2[ i ] = 1;  
    }

    std::vector< double > begins( numtasks );
    std::vector< double > ends( numtasks );
    double begin = clock();
    MPI_( MPI_Gather( &begin, 1, MPI_DOUBLE, &begins[ 0 ], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD ) ); 

    // PARALLEL DOT PRODUCT COMPUTATION
    real_t partial_dot = 0.f;
#ifndef GPU
    int t = 0;
    for( t = 0; t != PER_MPI_TASK_ARRAY_SIZE; ++t ) {
        partial_dot += v1[ t ] * v2[ t ];
    }
    //partial_dot = real_t( p );
#ifndef NO_LOG    
    std::ostringstream os;
    os << &nodeid[ 0 ] << " - rank: " << task << " size: " << PER_MPI_TASK_ARRAY_SIZE 
       << ' ' << t << "  partial dot: " << partial_dot << '\n' ;
    std::cout << os.str(); os.flush();
#endif     
#else
    // SELECT GPU = task % <num gpus on node>, note that with this
    // approach it is possible to support nodes with different numbers of GPUs
    int device_count = 0;
    if( cudaGetDeviceCount( &device_count ) != cudaSuccess ) {
        std::cerr << task << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " cudaGetDeviceCount FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
#ifdef MPI_RROBIN_
    const int device = ( task / node_count ) % device_count;
#else
    const int device = task % device_count;
#endif
#ifndef NO_LOG
    {    
        std::ostringstream os;
        os << &nodeid[ 0 ] << " - rank: " << task << "\tGPU: " << device << '\n';
        std::cout << os.str(); os.flush();
    }
#endif     
    if( cudaSetDevice( device ) != cudaSuccess ) {
        std::cerr << task << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " cudaGetSetDevice FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
#ifdef NO_GPU_MALLOC_TIME
    double malloc_begin = clock();
#endif
    real_t* dev_v1   = 0;
    real_t* dev_v2   = 0;
    real_t* dev_dout = 0;
    if( cudaMalloc( &dev_v1,   sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE ) != cudaSuccess ) {
        std::cerr << task << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " cudaMalloc FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
    if( cudaMalloc( &dev_v2,   sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE ) != cudaSuccess ) {
        std::cerr << task << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " cudaMalloc FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
#ifdef NO_GPU_MALLOC_TIME
    double malloc_end = clock();
    begin += malloc_end - malloc_begin;
#endif
    // MOVE DATA TO GPU
    if( cudaMemcpy( dev_v1, &v1[ 0 ], sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE,
                    cudaMemcpyHostToDevice ) != cudaSuccess ) {
        std::cerr << task << ' ' << __LINE__ << ' ' <<  cudaGetErrorString( cudaGetLastError() ) <<  " cudaMemcpy FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;    
    }
    if( cudaMemcpy( dev_v2, &v2[ 0 ], sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE,
                    cudaMemcpyHostToDevice ) != cudaSuccess ) {
        std::cerr << task << ' ' << __LINE__ << ' ' <<  cudaGetErrorString( cudaGetLastError() ) <<  " cudaMemcpy FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }     
    // INVOKE KERNEL
    const int NUM_THREADS_PER_BLOCK = BLOCK_SIZE; // must match size of buffer used for reduction
    const int NUM_BLOCKS = std::min( PER_MPI_TASK_ARRAY_SIZE / NUM_THREADS_PER_BLOCK,
                                     0xffff ); // max number of blocks is 64k 
    
    const int PARTIAL_REDUCE_SIZE = NUM_BLOCKS; 
    if( cudaMalloc( &dev_dout, sizeof( real_t ) * PARTIAL_REDUCE_SIZE ) != cudaSuccess ) {
        std::cerr << task << ' ' << __LINE__ << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " cudaMalloc FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
#ifndef REDUCE_GPU
    partial_dot_product_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>( dev_v1, dev_v2, PER_MPI_TASK_ARRAY_SIZE, dev_dout );  
    std::vector< real_t > rdot( PARTIAL_REDUCE_SIZE );
    cudaMemcpy( &rdot[ 0 ], dev_dout, sizeof( real_t ) * PARTIAL_REDUCE_SIZE, cudaMemcpyDeviceToHost );
    partial_dot = std::accumulate( rdot.begin(), rdot.end(), 0.f );
#else
    dot_product_full_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>( dev_v1, dev_v2, PER_MPI_TASK_ARRAY_SIZE, dev_dout );
    cudaMemcpy( &partial_dot, dev_dout, sizeof( real_t ) * 1, cudaMemcpyDeviceToHost ); 
#endif

#ifndef NO_LOG
    {    
        std::ostringstream os;
        os << &nodeid[ 0 ] << " - rank: " << task << " partial dot: " << partial_dot << '\n' ;
        std::cout << os.str(); os.flush();
    }
#endif
#endif

    // REDUCE (SUM) ALL ranks -> rank 0
    real_t result = 0.;
    MPI_( MPI_Reduce( &partial_dot, &result, 1, MPI_REAL_T_, MPI_SUM, 0, MPI_COMM_WORLD ) );

    double end = clock();
    MPI_( MPI_Gather( &end, 1, MPI_DOUBLE, &ends[ 0 ], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD ) ); 

    const std::pair< double, double > minmax( *std::min_element( begins.begin(), begins.end() ),
                                              *std::max_element( ends.begin(), ends.end() ) );  


    // IF RANK == 0 -> PRINT RESULT
    if( task == 0 ) {
        std::cout << "dot product result: " << result << std::endl;
        std::cout << "time: " << ( minmax.second - minmax.first ) / CLOCKS_PER_SEC << 's' << std::endl;
    } 
  
#ifdef GPU
    // RELEASE GPU RESOURCES
    cudaFree( dev_v1 );
    cudaFree( dev_v2 );
    cudaFree( dev_dout );
    cudaDeviceReset(); 
#endif

    // RELEASE MPI RESOURCES   
    MPI_( MPI_Finalize() );

    return 0;
}
