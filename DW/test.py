from mpi4py import MPI
import numpy as np
import numpy.linalg as npl

comm = MPI.COMM_WORLD
size = comm.Get_size()

K = int(1e8/size)
for i in range(K):
	a = np.random.randint(1,20,size = (10,10))
