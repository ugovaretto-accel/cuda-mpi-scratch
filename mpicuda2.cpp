// "mpi + cuda reduction" 

#ifdef GPU
#include <cuda.h>
#endif
#include <mpi.h>
#include <iostream>
#include <vector>
#include "mpierr.h"
#include <cmath>

typedef double real_t;

//static const real_t PI = 3.14159265358979323846;

//------------------------------------------------------------------------------
#ifdef GPU
const size_t BLOCK_SIZE = 16;

__global__ void full_dot( const real_t* v1, const real_t* v2, int N, real_t* out ) {
    __shared__ real_t cache[ BLOCK_SIZE ];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cache[ threadIdx.x ] = 0.f;
    while( i < N ) {
        cache[ threadIdx.x ] += v1[ i ] * v2[ i ];
        i += gridDim.x * blockDim.x;
    }    
    i = BLOCK_SIZE / 2;
    while( i > 0 ) {
        if( threadIdx.x < i ) cache[ threadIdx.x ] += cache[ threadIdx.x + i ];
        __syncthreads();
        i /= 2; //not sure bitwise operations are actually faster
    }
    if( threadIdx.x == 0 ) atomicAdd( out, cache[ 0 ] );   
}
#endif


//------------------------------------------------------------------------------
int main( int argc, char** argv ) {

    int numtasks = 0;
    int task     = 0;
    // INIT ENV
    MPI_( MPI_Errhandler_set( MPI_COMM_WORLD, MPI_ERRORS_RETURN ) );
    MPI_( MPI_Init( &argc, &argv ) );
    MPI_( MPI_Comm_size( MPI_COMM_WORLD, &numtasks ) );
    MPI_( MPI_Comm_rank( MPI_COMM_WORLD, &task  ) );
    
    const int ARRAY_SIZE = 1024 * 1024 * 1024;
    // @WARNING: ARRAY_SIZE must be evenly divisable by the number of MPI processes
    const int PER_MPI_TASK_ARRAY_SIZE = ARRAY_SIZE / numtasks;
    if( ARRAY_SIZE % numtasks != 0  && task == 0 ) {
        std::cerr << ARRAY_SIZE << " must be evenly divisable by the number of mpi processes" << std::endl;
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
    // PER TASK DATA INIT - in the real world this is the plae where data are read from file
    // through the MPI_File_ functions or, less likely received from the root process
    std::vector< real_t > v1( ARRAY_SIZE / numtasks, 0. );
    std::vector< real_t > v2( ARRAY_SIZE / numtasks, 0. );
    for( int i = 0; i != PER_MPI_TASK_ARRAY_SIZE; ++i ) {
        //const real_t arg = i * 2 * PI / PER_MPI_TASK_ARRAY_SIZE + task * 2 * PI / numtasks;
        //v1[ i ] = std::sin( arg );
        //v2[ i ] = std::cos( arg );
        v1[ i ] = 1;
        v2[ i ] = 1;  
    }

    real_t partial_scalar = 0.;
#ifndef CUDA
    for( int i = 0; i != PER_MPI_TASK_ARRAY_SIZE; ++i ) partial_scalar += v1[ i ] * v2[ i ];
#else
    // SELECT GPU = task % <num gpus on node>, note that with this
    // approach it is possible to support nodes with different numbers of GPUs
    int device_count = 0;
    if( cudaGetDeviceCount( &device_count ) != cudaSuccess ) {
        std::cerr << task << " cudaGetDeviceCount FAILED\n" 
        MPI_( MPI_Abort( MPI_COMM_WORLD ) );
        return 1;
    }
    const int device = task % device_count;
    if( cudaSetDevice( device ) != cudaSuccess ) {
        std::cerr << task << " cudaGetDeviceCount FAILED\n" 
        MPI_( MPI_Abort( MPI_COMM_WORLD ) );
        return 1;
    }
    real_t* dev_v1   = 0;
    real_t* dev_v2   = 0;
    real_t* dev_dout = 0;
    cudaMalloc( &dev_v1,   sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE );
    cudaMalloc( &dev_v2,   sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE );
    cudaMalloc( &dev_dout, sizeof( real_t ) * 1 );
    // MOVE DATA TO GPU
    cudaMemcpy( dev_v1, &v1[ 0 ], sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE , cudaMemcpyDeviceToHost );
    cudaMemcpy( dev_v2, &v2[ 0 ], sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE , cudaMemcpyDeviceToHost );
    // INVOKE KERNEL
    const int NUM_THREADS_PER_BLOCK = 16;
    const int NUM_BLOCKS = ( PER_MPI_TASK_ARRAY_SIZE + NUM_THREADS_PER_BLOCK - 1 ) / NUM_THREADS_PER_BLOCK;  
    dot_product_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>( dev_v1, dev_v2, PER_MPI_TASK_ARRAY_SIZE, dev_dout );
    
    // MOVE DATA TO CPU
    cudaMemcpy( &partial_scalar, sizeof( real_t ) * 1, cudaMemcpyDeviceToHost );
#endif

    // REDUCE (SUM) ALL ranks -> rank 0
    real_t result = 0.;
    MPI_( MPI_Reduce( &partial_scalar, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD ) );

    // IF RANK == 0 -> PRINT RESULT
    if( task == 0 ) {
        std::cout << "dot product result: " << result << std::endl;
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
