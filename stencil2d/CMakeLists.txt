Cmake_minimum_required( VERSION 2.8 )

project(cudasubarray)

FIND_PACKAGE(MPI REQUIRED)
INCLUDE(CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER(mpicxx "MPI C++ Compiler")

include_directories( ${MPI_INCLUDE_PATH} )
find_package( CUDA )

cuda_add_executable( cuda-2d-stencil-subarray mpi-2d-stencil-subarray-cuda.cu )

