# Distributed Dantzig-Wolfe Decomposition
A distributed Dantzig-Wolfe decomposition parallel implementation, where the master is solved using ADMM. The classical Dantzig-Wolfe algorithm where the master is solved centrally is also implemented. Parallel computations are done using MPI.

# Examples
Included are a script to create and solve randomly generated instances, and a script to read cutting-stock test instances and solve them. An example file containing instances for one-dimensional cutting stock problems with multiple stock lengths can be found under the data folder. This file was obtained from the CaPaD-group (http://www.math.tu-dresden.de/~capad/), from which more and larger instances can be found.


References:
Cutting stock instances - http://www.math.tu-dresden.de/~capad/
