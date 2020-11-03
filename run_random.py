#!/usr/bin/env python3
"""
Example script to create and solve randomly generated instances.

Example command to run: mpiexec -np 4 python runDistDW.py.
"""


import numpy as np
from argparse import ArgumentParser
import generate_instances as gi
from dist_dw import DistDW
import helpers as h
from mpi4py import MPI
import time, sys
from block import Block
from numpy.random import RandomState
import os
import csv

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = ArgumentParser()
    parser.add_argument('-g', help='Solve using Gurobi', action='store_true')
    args = vars(parser.parse_args())
    if args["g"]:
        assert comm.Get_size() == 1

    output_path = os.path.join("output", "random_instances")

    numBlocks = 4
    numLinkConstrs = 5
    numVars = 1000
    method = "admm" if not args["g"] else "g"

    # Each block has the same number of variables, except the last block
    # if numVars is not a multiple of numBlocks
    varsPerBlock = [int(numVars / numBlocks)] * numBlocks
    varsPerBlock[-1] = numVars - (numBlocks - 1) * varsPerBlock[0]
    mb = [min(int(varsPerBlock[i]), 2500) for i in range(numBlocks)]
    linkSenses = ['>'] * numLinkConstrs

    if rank == 0:
        print("Initializing blocks")
    seed = numLinkConstrs * numVars * numBlocks
    # Generate block data and right-hand side of link constraints
    if comm.Get_size() > 1:
        blockData, rowTotals = gi.generateRandomBlock(
            numLinkConstrs, mb[rank], varsPerBlock[rank], seed + int(rank))
        rowTotals = comm.gather(rowTotals, root=0)
        if rank == 0:
            prng = RandomState(seed)
            linkRHS = gi.getLinkRHS(rowTotals, prng=prng)
        else:
            linkRHS = np.empty(numLinkConstrs)
        # Broadcast right-hand side of linking constraints to all processors
        linkRHS = comm.bcast(linkRHS, root=0)
        linkingData = {"linkSense": linkSenses, "b": linkRHS,
                        "numBlocks": numBlocks}
        blocks = Block(numBlocks, linkingData, blockData)
    else:
        blockData = []
        rowTotals = []
        for i in range(numBlocks):
            currBlock, currRowTotals = gi.generateRandomBlock(
                numLinkConstrs, mb[rank], varsPerBlock[rank], seed + i)
            blockData.append(currBlock)
            rowTotals.append(currRowTotals)
        prng = RandomState(seed)
        linkRHS = gi.getLinkRHS(rowTotals, prng=prng)
        linkingData = {"linkSense": linkSenses, "b": linkRHS,
                        "numBlocks": numBlocks}
        blocks = []
        for i in range(numBlocks):
            blocks.append(Block(numBlocks, linkingData, blockData[i]))

    ddw = DistDW(blocks, numBlocks, numLinkConstrs, method=method,
                    linkConstrsSenses=linkingData["linkSense"],
                    rhs=linkingData["b"])
    
    if method != "g":
        comm.barrier()
        if rank == 0:
            t1 = time.time()
            print(method, "starting...")
        ddw.solve()
        if rank == 0:
            print(method, "done.")
            sys.stdout.flush()
            t2 = time.time()
            runtime = t2 - t1
        objVal = ddw.getObjVal()
        lhs = ddw.getLhsVal()
        if rank == 0:
            rhs = linkingData["b"].flatten()
            diff = np.subtract(rhs, lhs)
            normalizedDiff = np.divide(diff, rhs)
            feasViol = max(np.max(normalizedDiff), 0)
            print("Objective Value:", objVal)
            print("Runtime:", runtime)
            print("Feasibility violation:", feasViol)
            sys.stdout.flush()
    elif method == "g":
        gurobiOut = h.optimalDWSolve(linkingData, blockData)
        objVal = gurobiOut['objval']
        runtime = gurobiOut["runtime"]
        print("Gurobi objective value:", objVal)

        filename = "gurobi_output.csv"
        filepath = os.path.join(output_path, filename)