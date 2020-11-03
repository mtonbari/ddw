"""
Implementation of ADMM to solve the dual of the restricted master problem.
"""


import numpy as np
import numpy.linalg as npl
from block import Block
import math
from mpi4py import MPI
import parameters
import sys
import time


def solve(dw):
    """
    Solve dual of restricted master problem using ADMM.

    Parameters
    ----------
    dw : DistDW object

    Returns
    -------
    admmIter : int
        Number of iterations for ADMM to converge.
    """
    rank = dw.comm.Get_rank()
    dw.rho = dw.params.rhoStart
    admmIter = 0
    admmDone = 0
    while(not admmDone):
        if rank == 0:
            dualLinkValPrev = dw.dualLinkVal
        updateDual(dw, admmIter)
        updatea(dw)
        if admmIter % 1 == 0:
            if dw.MPISolve:
                dualResidBlock_2 = dw.blocks.computeDualResid(dw.dualLinkVal)
                dualResid_2 = np.empty(1, dtype='float')  # squared dual residual
                dw.comm.Reduce([dualResidBlock_2, MPI.DOUBLE],
                               [dualResid_2, MPI.DOUBLE],
                               op=MPI.SUM, root=0)
            else:
                dualResid_2 = sum(dw.blocks[ii].dualResid
                                  for ii in range(dw.numBlocks))
            if rank == 0:
                dw.dualResid = math.sqrt(max(dualResid_2, 0))
                dw.primalResid = dw.rho * npl.norm(dw.dualLinkVal
                                                   - dualLinkValPrev)
                # update rho
                dw.updateRho(dw.primalResid, dw.dualResid)

                admmDone = ((dw.primalResid <= dw.tolPrimal)
                            and (dw.dualResid <= dw.tolDual))
            if dw.MPISolve:
                dw.rho = dw.comm.bcast(dw.rho, root=0)
                admmDone = dw.comm.bcast(admmDone, root=0)
        admmIter += 1
    return admmIter


def updateDual(dw, admmIter):
    """
    Update steps of duals associated with the linking constraints of the master.
    
    Parameters
    ----------
    dw : DistDw object
    admmIter : int
        Current ADMM iteration.
    """
    rank = dw.comm.Get_rank()
    if dw.MPISolve:
        if rank == 0:
            # Receiving buffer
            dualLinkValRecv = np.empty([dw.comm.Get_size(), dw.numLinkConstrs])
        else:
            dualLinkValRecv = None
        dw.blocks.solveDualRMP(dw.dualLinkVal, dw.rho)
        dw.comm.Gather(dw.blocks.dualLinkVal, dualLinkValRecv, root=0)
    else:
        for i in range(dw.numBlocks):
            dw.blocks[i].solveDualRMP(dw.dualLinkVal, dw.rho)
        dualLinkValRecv = [dw.blocks[ii].dualLinkVal
                           for ii in range(dw.numBlocks)]
    if rank == 0:
        dualLinkValPrev = dw.dualLinkVal
        dw.dualLinkVal = np.mean(dualLinkValRecv, axis=0)

    if admmIter == 0:
        if dw.MPISolve:
            if rank == 0:
                aMat = np.empty([dw.comm.Get_size(), dw.numLinkConstrs])
            else:
                aMat = None
            dw.comm.Gather(dw.blocks.a, aMat, root=0)
        else:
            aMat = np.array([dw.blocks[ii].a
                             for ii in range(dw.numBlocks)])
        if rank == 0:
                aSum = aMat.sum(axis=0, dtype='float')
                dw.dualLinkVal = (dw.dualLinkVal
                                   + aSum / (dw.numBlocks * dw.rho))
    return


def updatea(dw):
    """
    Update step of Lagrangian multipliers associated with the copy constraints
    in the dual of the master.

    Parameters
    ----------
    dw : DistDw object
    """
    if dw.MPISolve:
        dw.comm.Bcast([dw.dualLinkVal, MPI.DOUBLE], root = 0) # broadcast dual variable to all workers
        dw.blocks.a = dw.blocks.a - dw.rho*(dw.dualLinkVal - dw.blocks.dualLinkVal)
    else:
        for i in range(dw.numBlocks):
            dw.blocks[i].a = (
                dw.blocks[i].a
                - dw.rho
                * (dw.dualLinkVal - dw.blocks[i].dualLinkVal))
            dw.blocks[i].computeDualResid(dw.dualLinkVal)
    return
