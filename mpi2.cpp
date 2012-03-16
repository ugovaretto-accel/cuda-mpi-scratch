//Error handling

//FYI: output of ldd when compiled with mvapich2's mpicxx:
//  linux-vdso.so.1 =>  (0x00007fff1f3b8000)
//  libmpichcxx.so.1.2 => /apps/eiger/mvapich2/1.5.1p1/mvapich2-gnu/lib/libmpichcxx.so.1.2 (0x00007f52e3e51000)
//  ibmpich.so.1.2 => /apps/eiger/mvapich2/1.5.1p1/mvapich2-gnu/lib/libmpich.so.1.2 (0x00007f52e3a7a000)
//  liblimic2.so.0 => /usr/local/lib/liblimic2.so.0 (0x00007f52e3878000)
//  libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f52e365b000)
//  librdmacm.so.1 => /usr/lib64/librdmacm.so.1 (0x00007f52e3452000)
//  libibverbs.so.1 => /usr/lib64/libibverbs.so.1 (0x00007f52e3243000)
//  libibumad.so.3 => /usr/lib64/libibumad.so.3 (0x00007f52e303c000)
//  libdl.so.2 => /lib64/libdl.so.2 (0x00007f52e2e38000)
//  librt.so.1 => /lib64/librt.so.1 (0x00007f52e2c2f000)
//  libstdc++.so.6 => /usr/lib64/libstdc++.so.6 (0x00007f52e2924000)
//  libm.so.6 => /lib64/libm.so.6 (0x00007f52e26ce000)
//  libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f52e24b7000)
//  libc.so.6 => /lib64/libc.so.6 (0x00007f52e2159000)
//  /lib64/ld-linux-x86-64.so.2 (0x00007f52e4075000)

#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <mpi.h>
#include "mpierr.h"

           
int main( int argc, char** argv ) {
    int rank, size, len;
    char nid[MPI_MAX_PROCESSOR_NAME];
    MPI_( MPI_Init( &argc, &argv ) );
    MPI_( MPI_Errhandler_set( MPI_COMM_WORLD, MPI_ERRORS_RETURN ) );
    MPI_( MPI_Comm_rank( MPI_COMM_WORLD, &rank ) );
    MPI_( MPI_Comm_size( MPI_COMM_WORLD, &size ) );
    MPI_( MPI_Get_processor_name( nid, &len ) );
    std::cout << "Hello world from process " << rank << " of " << size << " -- " << nid << std::endl;
    MPI_( MPI_Finalize() );
    return 0;
}
