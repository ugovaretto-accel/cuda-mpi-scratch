##Stencil computation with GPU-aware MPI implementation

Sample code performing data exchange between tiles of a 2D grid(array) mapped to a 2D MPI cartesian grid.

Works on both CPU and GPU(with mvapich2 >= 1.8) using MPI subarray type. 

_sample2D.h_ implements a minimal stencil library to generate MPI subarray data types associated
with the regions of the local grids and to exchange data over MPI.

Data and layout are kept separate:

* the data region is referenced by a plain pointer
* layout information is stored in a struct

Data exhange works by first creating an array of data transfer configurations and issuing
Isend/Irecev calls all at once at each exhange requests.

Everything is templated on the 2d array data type, the point of customization for client code
is the function

template < typename T > MPI_Datatype CreateArrayElementType();

which must be implemented for the data type used as the array element. 


###Building

To build the cuda-enabled version:

```
module load cmake
module load cuda (>= 4.1 required)
module load mvapich2 ( >= 1.8.1 required)
module load gcc/4.6.3 (works with anything < 4.7 due to CUDA not supporting gcc 4.7)

cmake

make
```

To build the non-cuda version invoke mpic++ directly on the .cpp file.


###Running

Sample _slurm_ sbatch script:

```
#!/bin/bash
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:2
#SBATCH --mem=24000
#SBATCH --time=00:30:00
#SBATCH --output=slurm_9_1_1_cuda-2d-stencil-subarray.out
. /etc/profile.d/modules.bash
module load mvapich2
module load cuda
module load gcc/4.6.3

mpiexec.hydra -rmk slurm ./cuda-2d-stencil-subarray
```

The sample code retrieves the number of MPI processes N and creates a 2D cartesian grid of size sqrt(N) x sqrt(N);
each process is associated with a corresponding 2D tile initialized with the MPI process id.

After initialization the halo(ghost) regions are exchanged, and each process writes its data to a file named with the cartesian coordinates of each process.

The data written to each is:
* global grid layout
* elements of the 2D tile before data exchange
* elements of the 2D tile after data exchange 

checkout the files in the _sample-output_ folder for examples of generated output data.


