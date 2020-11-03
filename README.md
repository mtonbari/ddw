# Distributed Dantzig-Wolfe Decomposition
A parallel implementation of distributed Dantzig-Wolfe decomposition, where the master is solved in a distributed fashion using ADMM. The classical Dantzig-Wolfe algorithm where the master is solved centrally is also implemented. Parallelization is done using MPI.

## Quickstart Examples:
    * run_csp.py: script to read and solve cutting-stock test instances.
    * run_random.py: script to create and solve randomly generated instances

An example file containing instances for one-dimensional cutting stock problems with multiple stock lengths can be found under the data folder. This file was obtained from the [CaPaD-group](http://www.math.tu-dresden.de/~capad/), from which more and larger instances can be found.

## Requirements
All of these can be installed using conda:
    * gurobipy: Gurobi for Python
    * numpy
    * mpi4py: MPI for Python package
To solve the algorithms in parallel, an MPI implementation such as MPICH must be installed.

## References
    * Cutting stock instances - http://www.math.tu-dresden.de/~capad/

## Publication
Mohamed El Tonbari, Shabbir Ahmed, [Distributed Dantzig-Wolfe Decomposition](https://arxiv.org/pdf/1905.03309.pdf)
