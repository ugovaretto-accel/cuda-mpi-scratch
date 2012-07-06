#include "mpi.h"
#include <stdio.h>
#include <iostream>

int main( int argc, char *argv[] ) {

    int rank, size, i;

    int blocklen[ 3 ] = { 1, 1, 1 };
    MPI_Aint displacement[ 3 ];
    MPI_Status status;

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    if( size < 2 ) {
        printf("Please run with 2 processes.\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if( rank == 0 ) {
    /*
     * 0 sends the bracketed subregion
     *
     * B1 [0  1  2  [3  4  5]  6  7]
     * B2 [0  2  4  [6  8 10] 12 14]
     * B3 [1  3  5  [7  9 11] 13 15]
     */

        int *B1 = new int[ 1500 ], *B2 = new int[ 8 ], *B3 = new int[ 28 ];
        int sz = 8, ssz = 3, starts = 3;
        MPI_Datatype blocktype1, finaltype;
        MPI_Type_create_subarray( 1, &sz, &ssz, &starts, MPI_ORDER_C, MPI_INT, &blocktype1 );
        MPI_Type_commit( &blocktype1 );

        displacement[ 0 ] = 0;
        displacement[ 1 ] = reinterpret_cast< char* >( B2 ) - reinterpret_cast< char* >( B1 );
        displacement[ 2 ] = reinterpret_cast< char* >( B3 ) - reinterpret_cast< char* >( B1 );

        std::cout << "(1) : "
                  << B2 << " - " << B1 << " = "
                  << displacement[1] << " ; "
                  << B3 << " - " << B1 << " = "
                  << displacement[2]
                  << std::endl;

        MPI_Type_create_hindexed(3, blocklen, displacement, blocktype1, &finaltype);
        MPI_Type_commit(&finaltype);

        {
            for( i = 0; i < 8; i++ ) {
                B1[ i ] = i;
                B2[ i ] = i * 2;
                B3[ i ] = i * 2 + 1;
            }
            MPI_Send(B1, 1, finaltype, 1, 123, MPI_COMM_WORLD);
        }

    }

    if( rank == 1 ) {
    /*
     * 1 receives the bracketed subregion
     *
     * B1 [[3  4  5] -1 -1 -1 -1 -1]
     * B2 [[6  8 10] -1 -1 -1 -1 -1]
     * B3 [[7  9 11] -1 -1 -1 -1 -1]
     */
        int B1[ 58 ], B2[ 8 ], *B3 = new int[ 28 ];
        int sz = 8, ssz = 3, starts = 0;
        MPI_Datatype blocktype1, finaltype;
        MPI_Type_create_subarray( 1, &sz, &ssz, &starts, MPI_ORDER_C, MPI_INT, &blocktype1 );
        MPI_Type_commit( &blocktype1 );

        displacement[ 0 ] = 0;
        displacement[ 1 ] = reinterpret_cast< char* >( B2 ) - reinterpret_cast< char* >( B1 );
        displacement[ 2 ] = reinterpret_cast< char* >( B3 ) - reinterpret_cast< char* >( B1 );

        std::cout << "(1) : "
                  << B2 << " - " << B1 << " = "
                  << displacement[1]<< " ; "
                  << B3 << " - " << B1 << " = "
                  << displacement[2]
                  << std::endl;

        MPI_Type_create_hindexed(3, blocklen, displacement, blocktype1, &finaltype);
        MPI_Type_commit(&finaltype);

        for( i = 0; i < 8; i++ ) {
            B1[ i ] = -1;
            B2[ i ] = -1;
            B3[ i ] = -1;
        }

        MPI_Recv( B1, 1, finaltype, 0, 123, MPI_COMM_WORLD, &status );
        for( i = 0; i < 8; i++ )
            printf("B1[%d] = %d\n", i, B1[ i ] );
        for ( i = 0; i < 8; i++ )
            printf( "B2[%d] = %d\n", i, B2[ i ] );
        for( i = 0; i < 8; i++ )
            printf( "B3[%d] = %d\n", i, B3[ i ] );
        fflush( stdout );
    }

    MPI_Finalize();
    return 0;
}


}
 */
}
}
}
 */
}
}
}
