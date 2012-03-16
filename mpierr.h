#pragma once
#ifndef MPIERR_H_
#define MPIERR_H_

// #Author: Ugo Varetto

#include <mpi.h>
#include <iostream>
#include <string>
#include <sstream>

// Error handling macros
//------------------------------------------------------------------------------
/* inline added for easy insertion into include file */
inline std::string format_mpi_err_msg( int code ) {
    int length_of_error_string = 0; 
    char error_string[ MPI_MAX_ERROR_STRING ]; 
    MPI_Error_string( code, error_string, &length_of_error_string );
    std::ostringstream err;
    err << "Error " << code << ":\n  error message: ";
    err << error_string;
    err << "\n  error class message: ";
    int error_class = 0;
    MPI_Error_class( code, &error_class );
    MPI_Error_string( error_class, error_string, &length_of_error_string );
    err << error_string;
    return err.str();
}

#define HANDLE_MPI_ERROR_EXCEPTION( code ) \
    { \
        if( code != MPI_SUCCESS ) { \
            throw( std::runtime_error( format_mpi_err_msg( code ) ) ); \
        } \
     } 

#define HANDLE_MPI_ERROR_STDERR( code ) \
    { \
        if( code != MPI_SUCCESS ) { \
            std::cerr << format_mpi_err_msg( code ) << std::endl; \
            MPI_Abort( MPI_COMM_WORLD, code ); \
        } \
    }
//------------------------------------------------------------------------------


//helper switch
#ifdef MPI_ERR_USE_EXCEPTIONS
  #define MPI_ HANDLE_MPI_ERROR_EXCEPTION
#else
  #define MPI_ HANDLE_MPI_ERROR_STDERR
#endif

#endif //MPIERR_H_
