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
struct Array2D {
    int width;
    int height;
    int xOffset;
    int yOffset;
    int rowStride;
    Array2D( int w, int h, int rs, int xoff = 0, int yoff = 0 ) :
        width( w ), height( h ), xOffset( xoff ), yOffset( yoff ),
        rowStride( rs )
    {}
    Array2D() : width( 0 ), height( 0 ), xOffset( 0 ), yOffset( 0 ),rowStride( 0 )
    {}
};

//------------------------------------------------------------------------------
template < typename T >
class Array2DAccessor {
public:
    const T& operator()( int x, int y ) const { 
        return *( data_ + 
                  ( layout_.yOffset * layout_.rowStride + layout_.xOffset ) + //constant
                    layout_.rowStride * y + x ); 
    }
    T& operator()( int x, int y ) {
        return *( data_ + 
                  ( layout_.yOffset * layout_.rowStride + layout_.xOffset ) + //constant
                  layout_.rowStride * y + x ); 
    }
    Array2DAccessor() : data_( 0 ) {}
    Array2DAccessor( T* data, const Array2D& layout ) : data_( data ), layout_( layout) {}
    const Array2D& Layout() const { return layout_; }
private:
    Array2D layout_;
    T* data_;
};

//------------------------------------------------------------------------------
enum RegionID { TOP_LEFT,    TOP_CENTER,    TOP_RIGHT,
                CENTER_LEFT, CENTER,        CENTER_RIGHT,
                BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT };

//------------------------------------------------------------------------------
//Possible to use templated function specialized with RegionID
Array2D SubArrayRegion( const Array2D& g, 
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
    return Array2D( width, height, stride, xoff, yoff );
}

//------------------------------------------------------------------------------
template < typename T > MPI_Datatype CreateArrayElementType();

template <> MPI_Datatype CreateArrayElementType< REAL >() { return MPI_DOUBLE_PRECISION; }

//------------------------------------------------------------------------------
MPI_Datatype CreateMPISubArrayType( const Array2D& g, const Array2D& subgrid ) {
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
TransferInfo ReceiveInfo( void* pdata, MPI_Comm cartcomm, int rank, RegionID target, Array2D& g,
                          int stencilWidth, int stencilHeight ) {
    TransferInfo ti;
    ti.comm = cartcomm;
    ti.data = pdata;
    ti.destTaskId = rank;
    ti.tag = target;
    ti.type = CreateMPISubArrayType( g, 
                                     SubArrayRegion( g, stencilWidth, stencilHeight, target ) );
    Offset offset = OffsetRegion( target ); 
    ti.srcTaskId = OffsetTaskId( cartcomm, offset.x, offset.y );  
    return ti;     
}
 
//------------------------------------------------------------------------------
TransferInfo SendInfo( void* pdata, MPI_Comm cartcomm, int rank, RegionID source, Array2D& g,
                       int stencilWidth, int stencilHeight ) {
    TransferInfo ti;
    ti.comm = cartcomm;
    ti.data = pdata;
    ti.srcTaskId = rank;
    ti.tag = source;
    Array2D core = SubArrayRegion( g, stencilWidth, stencilHeight, CENTER );
    ti.type = CreateMPISubArrayType( g, 
                                     SubArrayRegion( core, stencilWidth, stencilHeight, source ) );
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
std::pair< std::vector< TransferInfo >,
           std::vector< TransferInfo > > 
CreateSendRecvArrays( void* pdata, MPI_Comm cartcomm, int rank, Array2D& g,
                      int stencilWidth, int stencilHeight ) {
    std::vector< TransferInfo > ra;
    std::vector< TransferInfo > sa;
    RegionID rids[] = { TOP_LEFT,    TOP_CENTER,    TOP_RIGHT,
                        CENTER_LEFT, CENTER,        CENTER_RIGHT,
                        BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT };
    for( RegionID* i = rids; i != rids + sizeof( rids ) / sizeof( RegionID ); ++i ) {
        ra.push_back( ReceiveInfo( pdata, cartcomm, rank, *i, g, stencilWidth, stencilHeight ) );   
        sa.push_back( SendInfo( pdata, cartcomm, rank, *i, g, stencilWidth, stencilHeight ) );   
    }
    return std::make_pair( ra, sa ); 
}

//------------------------------------------------------------------------------
template < typename T >
void InitArray( T* pdata, const Array2D& g, const T& value ) {
    Array2DAccessor< T > a( pdata, g );
    for( int row = 0; row != a.Layout().height; ++row ) {
        for( int column = 0; column != a.Layout().width; ++column ) {
            a( column, row ) = value;

        }
    }
}

//------------------------------------------------------------------------------
template < typename T >
void Compute( T* pdata, const Array2D& g ) {}

//------------------------------------------------------------------------------
template < typename T >
bool TerminateCondition( T* pdata, const Array2D& g ) { return true; }


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
    std::vector< REAL > dataBuffer( localTotalWidth * localTotalHeight, 0 );  
    Array2D localArray( localTotalWidth, localTotalHeight, localTotalHeight );
    // Create transfer info arrays
    typedef std::vector< TransferInfo > VTI;
    std::pair< VTI, VTI > transferInfoArrays =
        CreateSendRecvArrays( &dataBuffer[ 0 ], cartcomm, task, localArray, stencilWidth, stencilHeight );     
    Array2D core = SubArrayRegion( localArray, stencilWidth, stencilHeight, CENTER );
    InitArray( &dataBuffer[ 0 ], core, REAL( task ) ); //init with this MPI task id
    // Exchange data and compute until condition met
    do {
        ExchangeData( transferInfoArrays.first, transferInfoArrays.second );
        Compute( &dataBuffer[ 0 ], core );
    } while( !TerminateCondition( &dataBuffer[ 0 ], core ) );

    MPI_Finalize();
    
    return 0;
}







