#!/usr/bin/env python3
"""
Example script to run dist_dw and solve cutting stock instances.

Example command to run: mpiexec -np 5 python runCSP.py.
"""
import numpy as np
import generate_instances as gi
from dist_dw import DistDW
from mpi4py import MPI
import time
from csp_block import CSPBlock
from math import log, sqrt
import sys
import os
import csv
from numpy.random import RandomState

prng = RandomState(0)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
MPISolve = comm.Get_size() > 1

filepath = "./data/1_20"  # 5 stock lengths
instance_num = 1
method = "admm"

stockLengths, weights, demands = gi.generateCSP(filepath,
                                                instanceNum=instance_num)
comm.barrier()

numStockTypes = len(stockLengths)
currNumItems = len(weights)
if  MPISolve:
    assert len(stockLengths) == comm.Get_size(), (
        "There are %r stock lengths but only %r processes." 
        %(len(stockLengths), comm.Get_size())
        )

# Construct cost of using stocks of different lengths
costs = [log(stockLengths[i]) for i in range(numStockTypes)]
# Initialize blocks
if MPISolve:
    blocks = CSPBlock(numStockTypes, costs[rank], weights, demands,
                        stockLengths[rank], blockID=rank)
else:
    blocks = []
    for i in range(numStockTypes):
        blocks.append(CSPBlock(numStockTypes, costs[i], weights,
                                demands, stockLengths[i], blockID=rank))
linkConstrsSenses = [">"] * currNumItems
instance = DistDW(blocks, numStockTypes, currNumItems, method=method,
                    convConstrsPresent=False,
                    linkConstrsSenses=linkConstrsSenses,
                    rhs=demands)
comm.barrier()

t1 = time.time()
instance.solve()
if rank == 0:
    runtime = time.time() - t1

obj = instance.getObjVal()
lhs = instance.getLhsVal()
if rank == 0:
    diff = np.subtract(demands, lhs)
    normalizedDiff = np.divide(diff, demands)
    feasViol = max(np.max(normalizedDiff), 0)

    print("Objective Value:", obj)
    print("Time to solve:", runtime)
    print("Feasibility violation:", feasViol)
    
