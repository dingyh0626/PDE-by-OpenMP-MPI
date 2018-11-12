# PDE-by-OpenMP-MPI

Solve PDE by OpenMP and MPI.

## Usage

* -np: number of processors
* --grids: size of grids
* --threads: number of threads
* --method: 0 for OpenMP and MPI hybrid, 1 for OpenMP only, default for Sequential method
* --output: output path

`$ mpirun -np 6 <XXX>/pde --grids 100 --threads 3 --method 0 --output <XXX>`
