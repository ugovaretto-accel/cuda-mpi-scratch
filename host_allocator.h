// author: Ugo Varetto
// std-compliant CUDA page-locked allocator
#pragma once
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>


// CONSTRUCTORS
// non-POD
template < typename T >
void construct( T* p, const T& val ) {
    new ( p ) T( val );  
}
// POD
template <>
void construct<char>( char* p, const char& val ) {
    *p = val;
}
template <>
void construct<unsigned char>( unsigned char* p, const unsigned char& val ) {
    *p = val;
}
template <>
void construct<float>( float* p, const float& val ) {
    *p = val;
}
template <>
void construct<double>( double* p, const double& val ) {
    *p = val;
}
template <>
void construct<int>( int* p, const int& val ) {
    *p = val;
}

// DESTRUCTORS
// non-POD
template < typename T >
void destroy( T* p ) {
    p->~T();
}

// POD
template <>
void destroy<char>( char* ) {}
template <>
void destroy<unsigned char>( unsigned char* ) {}
template <>
void destroy<float>( float* ) {}
template <>
void destroy<double>( double* ) {}
template <>
void destroy<int>( int* ) {}



template <class T > class host_allocator {
public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef T*        pointer;
    typedef const T*  const_pointer;
    typedef T&        reference;
    typedef const T&  const_reference;
    typedef T         value_type;
    template <class U> struct rebind { typedef host_allocator<U> other; };
public:
    ~host_allocator() throw() {}
    pointer address( reference x ) const { return &x; }
    const_pointer address( const_reference x ) const { return &x; };
    pointer allocate( size_type sz, const void* = 0 ) {
        char* buffer = 0;
        cudaError_t status = cudaMallocHost( &buffer, sz * sizeof( value_type ) );
        if( buffer == 0 || status != cudaSuccess ) {
          throw std::bad_alloc();
          return 0; // in case exceptions are not enabled
        }
        return reinterpret_cast< pointer >( buffer );
    }
    void deallocate(pointer p, size_type /*num_elements*/ ) {
       cudaFreeHost( p );
    }
    size_type max_size() const throw() {
        return std::numeric_limits< size_t >::max();
    }
    void construct( pointer p, const T& val ) {
        ::construct( p, val );
    }
    void destroy( pointer p ) {
        ::destroy( p );
    }
};



