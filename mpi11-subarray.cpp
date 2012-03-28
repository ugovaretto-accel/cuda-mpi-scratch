// "cartesian grid" 

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unistd.h> 
#include <cmath>
#include <cassert>

typedef double REAL;

//------------------------------------------------------------------------------
struct Grid2D {
    int width;
    int height;
    int xOffset;
    int yOffset;
    int rowStride;
    REAL*  data;
    Grid2D( REAL* d, int w, int h, int rs, int xoff = 0, int yoff = 0 ) :
        data( d ), width( w ), height( h ), xOffset( xoff ), yOffset( yoff ),
        rowStride( rs )
    {}
    Grid2D() : width( 0 ), height( 0 ), xOffset( 0 ), yOffset( 0 ), data( 0 ),
               rowStride( 0 )
    {}
    REAL operator()( int x, int y ) const { 
        return *( data + ( yOffset + y ) * rowStride + xOffset + x ); 
    }
    REAL& operator()( int x, int y ) {
        return *( data + ( yOffset + y ) * rowStride + xOffset + x ); 
    }

};

//------------------------------------------------------------------------------
enum RegionID { TOP_LEFT,    TOP_CENTER,    TOP_RIGHT,
                CENTER_LEFT, CENTER,        CENTER_RIGHT,
                BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT };

//------------------------------------------------------------------------------
//Possible to use templated function specialized with RegionID
Grid2D SubGridRegion( const Grid2D& g, 
                      int stencilWidth, 
                      int stencilHeight,
                      RegionID rid ) {
    int width = 0;
    int height = 0;
    int xoff = 0;
    int yoff = 0;
    const int stride = g.width;

    switch( rid ) {
        case TOP_LEFT:
            width = stencilWidth;
            height = stencilHeight;
            xoff = g.xOffset;
            yoff = g.yOffset;
            break;
        case TOP_CENTER:
            width = g.width - 2 * stencilWidth;
            height = stencilHeight;
            xoff = g.xOffset + stencilWidth;
            yoff = g.yOffset;
            break;
        case TOP_RIGHT:
            width = stencilWidth;
            height = stencilHeight;
            xoff = g.xOffset + g.width - stencilWidth;
            yoff = g.yOffset;
            break;
        case CENTER_LEFT:
            width = stencilWidth;
            height = g.height - 2 * stencilHeight;
            xoff = g.xOffset; 
            yoff = g.yOffset + stencilHeight;
            break;
        case CENTER: //core space
            width = g.width - 2 * stencilWidth;
            height = g.height - 2 * stencilHeight;
            xoff = g.xOffset + stencilWidth;
            yoff = g.yOffset + stencilHeight;
            break;
        case CENTER_RIGHT:
            width = stencilWidth;
            height = g.height - 2 * stencilHeight;
            xoff = g.xOffset + g.width - stencilWidth;
            yoff = g.yOffset + stencilHeight;
            break;
        case BOTTOM_LEFT:
            width = stencilWidth;
            height = stencilHeight;
            xoff = g.xOffset;
            yoff = g.yOffset + g.height - stencilHeight;
            break;
        case BOTTOM_CENTER:
            width = g.width - 2 * stencilWidth;
            height = stencilHeight;
            xoff = g.xOffset + stencilWidth;
            yoff = g.yOffset + g.height - stencilHeight;
            break;
        case BOTTOM_RIGHT:
            width = stencilWidth;
            height = stencilHeight;
            xoff = g.xOffset + g.width - stencilWidth;
            yoff = g.yOffset + g.height - stencilHeight;
            break;
        default:
            break; 
    }   
    return Grid2D( g.data, width, height, stride, xoff, yoff );
}

//------------------------------------------------------------------------------
template < typename T > MPI_Datatype CreateArrayElementType();

template <> MPI_Datatype CreateArrayElementType< REAL >() { return MPI_DOUBLE_PRECISION; }

//------------------------------------------------------------------------------
MPI_Datatype CreateMPISubArrayType( const Grid2D& g, const Grid2D& subgrid ) {
    int dimensions = 2;
    int sizes[] = { g.width, g.height };
    int subsizes[] = { subgrid.width, subgrid.height };
    int offsets[] = { subgrid.xOffset, subgrid.yOffset };
    int order = MPI_ORDER_C;
    MPI_Datatype arrayElementType = CreateArrayElementType< REAL >();// array element type
    MPI_Datatype newtype;
    MPI_Type_create_subarray( dimensions,
                              sizes,
                              subsizes,
                              offsets,
                              order,
                              arrayElementType,
                              &newtype );
    MPI_Type_commit( &newtype );
    return newtype;
}

//------------------------------------------------------------------------------
int OffsetTaskId( MPI_Comm comm, int xOffset, int yOffset ) {
    int thisRank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &thisRank );
    int coord[] = { -1, -1 }; 
    MPI_Cart_coords( comm, thisRank, 2, coord );
    coord[ 0 ] + xOffset;
    coord[ 1 ] + yOffset;
    int rank = -1;
    MPI_Cart_rank( comm, coord, &rank );
    return rank; 
}


//------------------------------------------------------------------------------
struct Offset {
    int x;
    int y;
    Offset( int ox, int oy ) : x( ox ), y( oy ) {}
};

Offset OffsetRegion( RegionID rid ) {
    int xoff = 0;
    int yoff = 0;
    switch( rid ) {
    case TOP_LEFT:
        xoff = -1;
        yoff =  1;
        break;
    case TOP_CENTER:
        xoff = 0;
        yoff = 1;
        break;
    case TOP_RIGHT:
        xoff = 1;
        yoff = 1;
        break;
    case CENTER_LEFT:
        xoff = -1;
        yoff =  0;
        break;
    case CENTER_RIGHT:
        xoff = 1;
        yoff = 0;
        break;
    case BOTTOM_LEFT:
        xoff = -1;
        yoff = -1;
        break;
    case BOTTOM_CENTER:
        xoff =  0;
        yoff = -1;
        break;
    case BOTTOM_RIGHT:
        xoff =  1;
        yoff = -1;
        break;
    default:
        break;
    }
    return Offset( xoff, yoff );
}

//------------------------------------------------------------------------------
struct TransferInfo {
    int srcTaskId;
    int destTaskId;
    int tag;
    void* data;
    MPI_Request request;
    MPI_Datatype type;
    MPI_Comm comm;
};

//------------------------------------------------------------------------------
TransferInfo ReceiveInfo( MPI_Comm cartcomm, int rank, RegionID target, Grid2D& g,
                          int stencilWidth, int stencilHeight ) {
    TransferInfo ti;
    ti.comm = cartcomm;
    ti.data = g.data;
    ti.destTaskId = rank;
    ti.tag = target;
    ti.type = CreateMPISubArrayType( g, 
                                     SubGridRegion( g, stencilWidth, stencilHeight, target ) );
    Offset offset = OffsetRegion( target ); 
    ti.srcTaskId = OffsetTaskId( cartcomm, offset.x, offset.y );  
    return ti;     
}
 
//------------------------------------------------------------------------------
TransferInfo SendInfo( MPI_Comm cartcomm, int rank, RegionID source, Grid2D& g,
                       int stencilWidth, int stencilHeight ) {
    TransferInfo ti;
    ti.comm = cartcomm;
    ti.data = g.data;
    ti.srcTaskId = rank;
    ti.tag = source;
    Grid2D core = SubGridRegion( g, stencilWidth, stencilHeight, CENTER );
    ti.type = CreateMPISubArrayType( g, 
                                     SubGridRegion( core, stencilWidth, stencilHeight, source ) );
    Offset offset = OffsetRegion( source ); 
    ti.destTaskId = OffsetTaskId( cartcomm, offset.x, offset.y );  
    return ti;     
}

//------------------------------------------------------------------------------
void ExchangeData( std::vector< TransferInfo >& recvArray,
                   std::vector< TransferInfo >& sendArray ) {

    std::vector< int > requests( recvArray.size() );
    for( int i = 0; i != recvArray.size(); ++i ) {
        TransferInfo& t = recvArray[ i ];
        MPI_Irecv( t.data, 1, t.type, t.srcTaskId, t.tag, t.comm, &( requests[ i ] ) );  
    }
    for( std::vector< TransferInfo >::iterator i = sendArray.begin();
         i != sendArray.end(); ++i ) {
        MPI_Isend( i->data, 1, i->type, i->destTaskId, i->tag, i->comm, &( i->request ) );  
    }
    std::vector< MPI_Status > status( sendArray.size() );
    MPI_Waitall( requests.size(), &requests[ 0 ], &status[ 0 ] );  
}

//------------------------------------------------------------------------------
std::pair< std::vector< TransferInfo > > 
CreateSendRecvArrays( MPI_Comm cartcomm, int rank, Grid2D& g,
                      int stencilWidth, int stencilHeight ) {
    std::vector< TransferInfo > ra;
    std::vector< TransferInfo > sa;
    RegionID rids[] = { TOP_LEFT,    TOP_CENTER,    TOP_RIGHT,
                        CENTER_LEFT, CENTER,        CENTER_RIGHT,
                        BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT };
    for( RegionID* i = rids; i != rids + sizeof( rids ) / sizeof( RegionID ); ++i ) {
        ra.push_back( ReceiveInfo( cartcomm, rank, *i, g, stencilWidth, stencilHeight ) );   
        sa.push_back( SendInfo( cartcomm, rank, *i, g, stencilWidth, stencilHeight ) );   
    }
    return std::makepair( ra, sa ); 
}

//------------------------------------------------------------------------------
void InitGrid( Grid2D& g ) {

}

//------------------------------------------------------------------------------
void Compute( Grid2D& g ) {}

//------------------------------------------------------------------------------
bool TerminateCondition( const Grid2D& /*g*/ ) { return true; }


//------------------------------------------------------------------------------
int main( int argc, char** argv ) {
    int numtasks = 0; 
    // Init, world size     
    MPI_Init( &argc, &argv );
    MPI_Errhandler_set( MPI_COMM_WORLD, MPI_ERRORS_RETURN );
    MPI_Comm_size( MPI_COMM_WORLD, &numtasks );
    // 2D square MPI cartesian grid 
    const int DIM = int( std::sqrt( double( numtasks ) ) );
    std::vector< int > dims( 2, DIM );
    std::vector< int > periods( 2, 1 ); //periodic in both dimensions
    const int reorder = 0; //false - no reorder, is it actually used ?
    MPI_Comm cartcomm; // communicator for cartesian grid
    MPI_Cart_create( MPI_COMM_WORLD, 2, &dims[ 0 ], &periods[ 0 ], reorder, &cartcomm ); 
    // current mpi task id of this process
    int task = -1;
    MPI_Comm_rank( cartcomm, &task );
    std::vector< int > coords( 2, -1 );
    MPI_Cart_coords( cartcomm, task, 2, &coords[ 0 ] );
    //////////////////////
    // Init data
    int localWidth = 128;
    int localHeight = 128;
    int stencilWidth = 3;
    int stencilHeight = 3;
    int localTotalWidth = localWidth + 2 * stencilWidth;
    int localTotalHeight = localHeight + 2 * stencilHeight;
    std::vector< REAL > buffer( localTotalWidth * localTotalHeight, 0 );  
    Grid2D localGrid( &buffer[ 0 ], localTotalWidth, localTotalHeight, localTotalHeight );
    // Create transfer info arrays
    std::pair< std::vector< TransferInfo > > infoArrays =
        CreateSendRecvArrays( cartcomm, task, grid, stencilWidth, stencilHeight );     
    Grid2D core = SubGridRegion( grid, stencilWidth, stencilHeight, CENTER );
    InitGrid( core );
    ///////////////////////
    do {
        ExchangeData( infoArray.first, infoArray.second );
        Compute( core );
    } while( !TerminateCondition( core ) );
    
    return 0;
}







