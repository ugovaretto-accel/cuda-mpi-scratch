// mvapich2 gpu to gpu communication 

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unistd.h> 
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>

typedef double REAL;


//------------------------------------------------------------------------------
struct Array2D {
    int width;
    int height;
    int xOffset;
    int yOffset;
    int rowStride;
    __host__ __device__
    Array2D( int w, int h, int rs, int xoff = 0, int yoff = 0 ) :
        width( w ), height( h ), xOffset( xoff ), yOffset( yoff ),
        rowStride( rs )
    {}
    __host__ __device__
    Array2D() : width( 0 ), height( 0 ), xOffset( 0 ), yOffset( 0 ),rowStride( 0 )
    {}
};

std::ostream& operator<<( std::ostream& os, const Array2D& a ) {
    os << "width:  " << a.width << ", " 
       << "height: " << a.height << ", "
       << "x offset: " << a.xOffset << ", "
       << "y offset: " << a.yOffset;
    return os;
}

//------------------------------------------------------------------------------
template < typename T >
class Array2DAccessor {
public:
    __host__ __device__
    const T& operator()( int x, int y ) const { 
        return *( data_ + 
                  ( layout_.yOffset * layout_.rowStride + layout_.xOffset ) + //constant
                    layout_.rowStride * y + x ); 
    }
    __host__ __device__
    T& operator()( int x, int y ) {
        
        return *( data_ + 
                  ( layout_.yOffset * layout_.rowStride + layout_.xOffset ) + //constant
                  layout_.rowStride * y + x ); 
    }
    __host__ __device__
    Array2DAccessor() : data_( 0 ) {}
    __host__ __device__
    Array2DAccessor( T* data, const Array2D& layout ) : data_( data ), layout_( layout) {}
    __host__ __device__
    const Array2D& Layout() const { return layout_; }
private:
    Array2D layout_;
    T* data_;
};

//------------------------------------------------------------------------------
enum RegionID { TOP_LEFT,    TOP_CENTER,    TOP_RIGHT,
                CENTER_LEFT, CENTER,        CENTER_RIGHT,
                BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT,
                TOP, LEFT, BOTTOM, RIGHT };

//------------------------------------------------------------------------------
template < typename T >
void Print( T* pdata, const Array2D& g, std::ostream& os ) {
    Array2DAccessor< T > a( pdata, g );
    for( int row = 0; row != a.Layout().height; ++row ) {
        for( int column = 0; column != a.Layout().width; ++column ) {
            os << a( column, row ) << ' ';

        }
        os << std::endl;
    }
}

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
    const int ghostRegionWidth  = stencilWidth  / 2;
    const int ghostRegionHeight = stencilHeight / 2;
    switch( rid ) {
        case TOP_LEFT:
            width = ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset;
            yoff = g.yOffset;
            break;
        case TOP_CENTER:
            width = g.width - 2 * ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset + ghostRegionWidth;
            yoff = g.yOffset;
            break;
        case TOP_RIGHT:
            width = ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset + g.width - ghostRegionWidth;
            yoff = g.yOffset;
            break;
        case CENTER_LEFT:
            width = ghostRegionWidth;
            height = g.height - 2 * ghostRegionHeight;
            xoff = g.xOffset; 
            yoff = g.yOffset + ghostRegionHeight;
            break;
        case CENTER: //core space
            width = g.width - 2 * ghostRegionWidth;
            height = g.height - 2 * ghostRegionHeight;
            xoff = g.xOffset + ghostRegionWidth;
            yoff = g.yOffset + ghostRegionHeight;
            break;
        case CENTER_RIGHT:
            width = ghostRegionWidth;
            height = g.height - 2 * ghostRegionHeight;
            xoff = g.xOffset + g.width - ghostRegionWidth;
            yoff = g.yOffset + ghostRegionHeight;
            break;
        case BOTTOM_LEFT:
            width = ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset;
            yoff = g.yOffset + g.height - ghostRegionHeight;
            break;
        case BOTTOM_CENTER:
            width = g.width - 2 * ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset + ghostRegionWidth;
            yoff = g.yOffset + g.height - ghostRegionHeight;
            break;
        case BOTTOM_RIGHT:
            width = ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset + g.width - ghostRegionWidth;
            yoff = g.yOffset + g.height - ghostRegionHeight;
            break;
        case TOP:
            width = g.width;
            height = ghostRegionHeight;
            xoff = g.xOffset;
            yoff = g.yOffset;
            break;
        case RIGHT:
            width = ghostRegionWidth;
            height = g.height;
            xoff = g.xOffset + g.width - ghostRegionWidth;
            yoff = g.yOffset;
            break;
        case BOTTOM:
            width = g.width;
            height = ghostRegionHeight;
            xoff = g.xOffset;
            yoff = g.yOffset + g.height - ghostRegionHeight;
            break;
        case LEFT:
            width = ghostRegionWidth;
            height = g.height;
            xoff = g.xOffset;
            yoff = g.yOffset;
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
//    printf( "%d %d\n", coord[ 0 ], coord[ 1 ] );
    coord[ 0 ] += xOffset;
    coord[ 1 ] += yOffset;
    int rank = -1;
    MPI_Cart_rank( comm, coord, &rank );
//    printf( "In rank: %d, offset: %d, %d; out rank: %d\n", thisRank, xOffset, yOffset, rank ); 
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
    case TOP:
        xoff = 0;
        yoff = 1;
        break;
    case RIGHT:
        xoff = 1;
        yoff = 0;
        break;
    case BOTTOM:
        xoff = 0;
        yoff = -1;
        break;
    case LEFT:
        xoff = -1;
        yoff = 0;
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
TransferInfo ReceiveInfo( void* pdata, MPI_Comm cartcomm, int rank, 
                          RegionID source, RegionID target, Array2D& g,
                          int stencilWidth, int stencilHeight, int tag ) {
    TransferInfo ti;
    ti.comm = cartcomm;
    ti.data = pdata;
    ti.destTaskId = rank;
    ti.tag = tag;
    ti.type = CreateMPISubArrayType( g, 
                                     SubArrayRegion( g, stencilWidth, stencilHeight, target ) );
    Offset offset = OffsetRegion( source ); 
    ti.srcTaskId = OffsetTaskId( cartcomm, offset.x, offset.y );

  //  printf( "source %d dest %d\n", ti.srcTaskId, ti.destTaskId ); 
  
    return ti;     
}
 
//------------------------------------------------------------------------------
TransferInfo SendInfo( void* pdata, MPI_Comm cartcomm, int rank, 
                       RegionID source, RegionID target, Array2D& g,
                       int stencilWidth, int stencilHeight, int tag ) {
    TransferInfo ti;
    ti.comm = cartcomm;
    ti.data = pdata;
    ti.srcTaskId = rank;
    ti.tag = tag;
    Array2D core = SubArrayRegion( g, stencilWidth, stencilHeight, CENTER );
    ti.type = CreateMPISubArrayType( g, 
                                     SubArrayRegion( core, stencilWidth, stencilHeight, source ) );
    Offset offset = OffsetRegion( target ); 
    ti.destTaskId = OffsetTaskId( cartcomm, offset.x, offset.y );  
    return ti;     
}

//------------------------------------------------------------------------------
void ExchangeData( std::vector< TransferInfo >& recvArray,
                   std::vector< TransferInfo >& sendArray ) {

    std::vector< int > requests( recvArray.size() + sendArray.size() );
    for( int i = 0; i != recvArray.size(); ++i ) {
        TransferInfo& t = recvArray[ i ];
        MPI_Irecv( t.data, 1, t.type, t.srcTaskId, t.tag, t.comm, &( requests[ i ] ) );  
    }
    for( int i = 0; i != sendArray.size(); ++i ) {
        TransferInfo& t = sendArray[ i ];
        MPI_Isend( t.data, 1, t.type, t.destTaskId, t.tag, t.comm, &( requests[ recvArray.size() + i ] ) );  
    }
    std::vector< MPI_Status > status( recvArray.size() + sendArray.size() );
    MPI_Waitall( requests.size(), &requests[ 0 ], &status[ 0 ] );  
}

//------------------------------------------------------------------------------
std::pair< std::vector< TransferInfo >,
           std::vector< TransferInfo > > 
CreateSendRecvArrays( void* pdata, MPI_Comm cartcomm, int rank, Array2D& g,
                      int stencilWidth, int stencilHeight ) {
    std::vector< TransferInfo > ra;
    std::vector< TransferInfo > sa;
    RegionID sendRegions[] = { TOP_LEFT,    TOP,    TOP_RIGHT,
                               LEFT,                RIGHT,
                               BOTTOM_LEFT, BOTTOM, BOTTOM_RIGHT };
    RegionID recvRegions[] = { BOTTOM_RIGHT,  BOTTOM_CENTER,    BOTTOM_LEFT,
                               CENTER_RIGHT,                CENTER_LEFT,
                               TOP_RIGHT, TOP_CENTER, TOP_LEFT };

    // use send regions as tags
    for( RegionID *s = sendRegions, *r = recvRegions; s !=  sendRegions + sizeof( sendRegions ) / sizeof( RegionID ); ++s, ++r ) {
        ra.push_back( ReceiveInfo( pdata, cartcomm, rank, *s, *r, g, stencilWidth, stencilHeight, *s ) );   
        sa.push_back( SendInfo( pdata, cartcomm, rank, *s, *r, g, stencilWidth, stencilHeight, *s ) );   
    }
    return std::make_pair( ra, sa ); 
}


//------------------------------------------------------------------------------
template < typename T >
__global__ void InitData( T* pdata, Array2D layout, int value ) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column =  blockIdx.x * blockDim.x + threadIdx.x;
    Array2DAccessor< T > a( pdata, layout );
    a( column, row ) = value;
}


//------------------------------------------------------------------------------
template < typename T >
void InitArray( T* pdata, const Array2D& g, const T& value ) {
    InitData<<<dim3( g.width, g.height, 1 ), 1 >>>( pdata, g, value );
}


//------------------------------------------------------------------------------
template < typename T >
void Compute( T* pdata, const Array2D& g ) {}

//------------------------------------------------------------------------------
template < typename T >
bool TerminateCondition( T* pdata, const Array2D& g ) { return true; }


//------------------------------------------------------------------------------
int main( int argc, char** argv ) {
#if 0 
    void TestSubRegionExtraction();
    TestSubRegionExtraction();
#else
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
      
    std::ostringstream ss;
    ss << coords[ 0 ] << '_' << coords[ 1 ];
    std::ofstream os( ss.str().c_str() );
    os << "Rank:  " << task << std::endl
       << "Coord: " << coords[ 0 ] << ", " << coords[ 1 ] << std::endl;
    // Init data
    int localWidth = 16;
    int localHeight = 16;
    int stencilWidth = 5;
    int stencilHeight = 5;
    int localTotalWidth = localWidth + 2 * ( stencilWidth / 2 );
    int localTotalHeight = localHeight + 2 * ( stencilHeight / 2 );
    
    REAL* deviceBuffer = 0;
    cudaMalloc( &deviceBuffer, localTotalWidth * localTotalHeight * sizeof( REAL ) );
    cudaMemset( deviceBuffer, 0, localTotalWidth * localTotalHeight * sizeof( REAL ) );   
    Array2D localArray( localTotalWidth, localTotalHeight, localTotalWidth );
    // Create transfer info arrays
    typedef std::vector< TransferInfo > VTI;
    std::pair< VTI, VTI > transferInfoArrays =
        CreateSendRecvArrays( deviceBuffer, cartcomm, task, localArray, stencilWidth, stencilHeight );     
    Array2D core = SubArrayRegion( localArray, stencilWidth, stencilHeight, CENTER );
    InitArray( deviceBuffer, core, REAL( task + 1 ) ); //init with this MPI task id
    os << "Array" << std::endl;
    std::vector< REAL > hostBuffer( localArray.width * localArray.height );
    cudaMemcpy( &hostBuffer[ 0 ], deviceBuffer, localArray.width * localArray.height * sizeof( REAL ),
                cudaMemcpyDeviceToHost );
    Print( &hostBuffer[ 0 ], localArray, os );
    os << std::endl;
    // Exchange data and compute until condition met
    do {
        ExchangeData( transferInfoArrays.first, transferInfoArrays.second );
        Compute( deviceBuffer, core );
    } while( !TerminateCondition( deviceBuffer, core ) );
    os << "Array after exchange" << std::endl;    
    MPI_Finalize();
    cudaMemcpy( &hostBuffer[ 0 ], deviceBuffer, localArray.width * localArray.height * sizeof( REAL ),
                cudaMemcpyDeviceToHost );
    Print( &hostBuffer[ 0 ], localArray, os );   
 #endif
    return 0;
}
 
//------------------------------------------------------------------------------
void TestSubRegionExtraction() {
    const int w = 32;
    const int h = 32;
    const int stencilWidth = 5;
    const int stencilHeight = 5;
    const int totalWidth = w + stencilWidth / 2;
    const int totalHeight = h + stencilHeight / 2;
    std::vector< int > data( totalWidth * totalHeight, 0 );
    Array2D grid( totalWidth, totalHeight, totalWidth );
    Array2D topleft = SubArrayRegion( grid, stencilWidth, stencilHeight, TOP_LEFT );
    Array2D topcenter = SubArrayRegion( grid, stencilWidth, stencilHeight, TOP_CENTER );
    Array2D topright= SubArrayRegion( grid, stencilWidth, stencilHeight, TOP_RIGHT );
    Array2D centerleft = SubArrayRegion( grid, stencilWidth, stencilHeight, CENTER_LEFT );
    Array2D center = SubArrayRegion( grid, stencilWidth, stencilHeight, CENTER );
    Array2D centerright = SubArrayRegion( grid, stencilWidth, stencilHeight, CENTER_RIGHT );
    Array2D bottomleft = SubArrayRegion( grid, stencilWidth, stencilHeight, BOTTOM_LEFT );
    Array2D bottomcenter = SubArrayRegion( grid, stencilWidth, stencilHeight, BOTTOM_CENTER );
    Array2D bottomright = SubArrayRegion( grid, stencilWidth, stencilHeight, BOTTOM_RIGHT );
  
    std::cout << "\nGRID TEST\n";
    
    std::cout << "Width: " << totalWidth << ", " << "Height: " << totalHeight << std::endl;
    std::cout << "Stencil: " << stencilWidth << ", " << stencilHeight << std::endl;
 
    std::cout << "top left:      " << topleft      << std::endl;
    std::cout << "top center:    " << topcenter    << std::endl;
    std::cout << "top right:     " << topright     << std::endl;
    std::cout << "center left:   " << centerleft   << std::endl;
    std::cout << "center:        " << center       << std::endl;
    std::cout << "center right:  " << centerright  << std::endl;
    std::cout << "bottom left:   " << bottomleft   << std::endl;
    std::cout << "bottom center: " << bottomcenter << std::endl;
    std::cout << "bottom right:  " << bottomright  << std::endl;

    std::cout << "\nSUBGRID TEST\n";

    Array2D core = center;
    topleft = SubArrayRegion( core, stencilWidth, stencilHeight, TOP_LEFT );
    topcenter = SubArrayRegion( core, stencilWidth, stencilHeight, TOP_CENTER );
    topright= SubArrayRegion( core, stencilWidth, stencilHeight, TOP_RIGHT );
    centerleft = SubArrayRegion( core, stencilWidth, stencilHeight, CENTER_LEFT );
    center = SubArrayRegion( core, stencilWidth, stencilHeight, CENTER );
    centerright = SubArrayRegion( core, stencilWidth, stencilHeight, CENTER_RIGHT );
    bottomleft = SubArrayRegion( core, stencilWidth, stencilHeight, BOTTOM_LEFT );
    bottomcenter = SubArrayRegion( core, stencilWidth, stencilHeight, BOTTOM_CENTER );
    bottomright = SubArrayRegion( core, stencilWidth, stencilHeight, BOTTOM_RIGHT );
    Array2D top = SubArrayRegion( core, stencilWidth, stencilHeight, TOP );
    Array2D right = SubArrayRegion( core, stencilWidth, stencilHeight, RIGHT );
    Array2D bottom = SubArrayRegion( core, stencilWidth, stencilHeight, BOTTOM );
    Array2D left = SubArrayRegion( core, stencilWidth, stencilHeight, LEFT );

    std::cout << "Width: " << core.width << ", " << "Height: " << core.height << std::endl;
    std::cout << "Stencil: " << stencilWidth << ", " << stencilHeight << std::endl;
    
    std::cout << "top left:      " << topleft      << std::endl;
    std::cout << "top center:    " << topcenter    << std::endl;
    std::cout << "top right:     " << topright     << std::endl;
    std::cout << "center left:   " << centerleft   << std::endl;
    std::cout << "center:        " << center       << std::endl;
    std::cout << "center right:  " << centerright  << std::endl;
    std::cout << "bottom left:   " << bottomleft   << std::endl;
    std::cout << "bottom center: " << bottomcenter << std::endl;
    std::cout << "bottom right:  " << bottomright  << std::endl;
    std::cout << "top:           " << top          << std::endl;
    std::cout << "right:         " << right        << std::endl;
    std::cout << "bottom:        " << bottom       << std::endl;
    std::cout << "left:          " << left         << std::endl;

}

