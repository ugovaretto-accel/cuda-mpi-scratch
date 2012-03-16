#!/bin/bash
#PBS -N mpi_cuda
#PBS -l select=4:ncpus=12:mpiprocs=16
#PBS -l place=scatter:excl
#PBS -l walltime=00:00:10
#PBS -l cput=00:00:10
#PBS -q restricted@eiger170
#PBS -r n
#======START====
. /etc/profile.d/modules.bash
module load mvapich2/1.6
module load cuda
echo "On which nodes it executes :"
cat $PBS_NODEFILE
echo "Which MPI Implementation is used :"
which mpiexec.hydra
echo "Now run the MPI tasks..."
mpiexec.hydra -rmk pbs /users/uvaretto/builds/eiger/basel-wshop/a.out
#======END====
