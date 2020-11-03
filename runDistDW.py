#!/usr/bin/env python
"""
"""


import numpy as np
from argparse import ArgumentParser
import generateInstances as gi
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
    if rank == 0:
        print(args)
    output_path = os.path.join("output", "random_instances")

    numBlocks = [2]  #, 4, 8, 10, 15]
    num_link_constrs = [5] #[1, 2, 5, 10]
    num_vars = [100] #, 1000, 5000, 10000, 20000, 25000, 50000]
    experiments = [(N, m, n) 
                   for N in numBlocks
                   for m in num_link_constrs for n in num_vars]
    methods = ["admm", "dw"] if not args["g"] else ["g"]
    for numBlocks, m, n in experiments:
        if rank == 0:
            print("num_link:", m)
            print("num_vars:", n)
        for method in methods:
            comm.barrier()
            varsPerBlock = [int(n / numBlocks)] * numBlocks
            varsPerBlock[-1] = n - (numBlocks - 1) * varsPerBlock[0]
            mb = [min(int(varsPerBlock[i]), 2500) for i in range(numBlocks)]
            link_sense = ['>'] * m

            if rank == 0:
                print("Initializing blocks")
            seed = m * n * numBlocks
            if comm.Get_size() > 1:
                block_data, row_totals = gi.generate_random_block(
                    m, mb[rank], varsPerBlock[rank], seed + int(rank))
                row_totals = comm.gather(row_totals, root=0)
                if rank == 0:
                    prng = RandomState(seed)
                    link_rhs = gi.get_link_rhs(row_totals, prng=prng)
                else:
                    link_rhs = np.empty(m)
                link_rhs = comm.bcast(link_rhs, root=0)
                linkingData = {"linkSense": link_sense, "b": link_rhs,
                               "numBlocks": numBlocks}
                blocks = Block(numBlocks, linkingData, block_data, method)
            else:
                block_data = []
                row_totals = []
                for i in range(numBlocks):
                    curr_block, curr_row_totals = gi.generate_random_block(
                        m, mb[rank], varsPerBlock[rank], seed + i)
                    block_data.append(curr_block)
                    row_totals.append(curr_row_totals)
                prng = RandomState(seed)
                link_rhs = gi.get_link_rhs(row_totals, prng=prng)
                linkingData = {"linkSense": link_sense, "b": link_rhs,
                               "numBlocks": numBlocks}
                blocks = []
                for i in range(numBlocks):
                    blocks.append(Block(numBlocks, linkingData, block_data[i], method))

            ddw = DistDW(blocks, numBlocks, m, method=method,
                         linkConstrsSenses=linkingData["linkSense"],
                         rhs=linkingData["b"])
            if method != "g":
                comm.barrier()
                if rank == 0:
                    t1 = time.time()
                    print(method, "start")
                ddw.solve()
                if rank == 0:
                    print(method, "done")
                    sys.stdout.flush()
                    t2 = time.time()
                    runtime = t2 - t1
                objVal = ddw.getObjVal()
                lhs = ddw.getLhsVal()
                # util = ddw.cpuTime/(ddw.cpuTime+ddw.commTime)
                # print('Rank ' + str(rank) + ' utilization: ' + str(util))
                # totalUtil = comm.reduce(util, op = MPI.SUM, root = 0)
                if rank == 0:
                    # print(totalUtil/float(comm.Get_size()))
                    # print('Mean core utilization: ' + str(sum(util)/float(len(util))))

                    rhs = linkingData["b"].flatten()
                    diff = np.subtract(rhs, lhs)
                    normalizedDiff = np.divide(diff, rhs)
                    feasViol = max(np.max(normalizedDiff), 0)
                    print('Objective Value')
                    print(objVal)
                    print('Time: ' + str(runtime))
                    print("Feasibility violation:", feasViol)
                    sys.stdout.flush()
                    filename = method + "_output.csv"
                    filepath = os.path.join(output_path, filename)
            elif method == "g":
                gurobiOut = h.optimalDWSolve(linkingData, block_data)
                del block_data
                del linkingData
                objVal = gurobiOut['objval']
                runtime = gurobiOut["runtime"]
                print("Gurobi objective value:", objVal)

                filename = "gurobi_output.csv"
                filepath = os.path.join(output_path, filename)