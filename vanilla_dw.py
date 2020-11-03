"""
Implementation of classical Dantzig-Wolfe decomposition. This is called
by dist_dw.
"""


import numpy as np
import helpers as hp
import sys
from mpi4py import MPI
from gurobipy import *


def phase1(dw):
    """
    Solve phase 1 of the restricted master problem to obtain a feasible model.

    Parameters
    ----------
    dw : DistDw object
    
    Returns
    -------
    phase1_model : Gurobi model
    """
    linkConstrsSenses = dw.linkConstrsSenses
    rank = dw.comm.Get_rank()
    Av = np.array(dw.blocks.Av[0])
    AvRecv = np.empty([dw.numBlocks, dw.numLinkConstrs])
    dw.comm.Gather(Av, AvRecv, root=0)

    if rank == 0:
        phase1Model = Model("Phase 1")
        phase1Model.setParam('OutputFlag', 0)
        lams = phase1Model.addVars(dw.numBlocks, lb=0, ub=np.inf, vtype='C',
                                   name='block').values()
        for i in range(dw.numLinkConstrs):
            expr = LinExpr(AvRecv[:, i], lams)
            phase1Model.addConstr(expr, linkConstrsSenses[i], dw.rhs[i],
                                  name='link' + str(i))
        if dw.convConstrsPresent:
            for i in range(dw.numBlocks):
                phase1Model.addConstr(lams[i], '==', 1, name='conv' + str(i))
        phase1Model.update()

        objExpr = 0
        for i in range(dw.numLinkConstrs):
            linkConstr = phase1Model.getConstrByName('link' + str(i))
            if linkConstrsSenses[i] == '<':
                si = phase1Model.addVar(
                    lb=0, ub=np.inf, column=Column(-1, linkConstr),
                    name = 'dummy')
                objExpr += si
            elif linkConstrsSenses[i] == '>':  
                si = phase1Model.addVar(
                    lb = 0, ub = np.inf, column = Column(1, linkConstr),
                    name = 'dummy')
                objExpr += si
            elif linkConstrsSenses[i] == '==':
                si1 = phase1Model.addVar(
                    lb = 0, ub = np.inf, column = Column(1, linkConstr),
                    name = 'dummy')
                si2 = phase1Model.addVar(
                    lb = 0, ub = np.inf, column = Column(-1, linkConstr),
                    name = 'dummy')
                objExpr += si1 + si2
        phase1Model.setObjective(objExpr, GRB.MINIMIZE)
    else:
        phase1Model = None
    phase1Model = colGen(dw, phase1Model, isPhase1=True)
    return phase1Model


def colGen(dw, model, is_phase1):
    """
    Solve model using column generation.

    Parameters
    ----------
    model : Gurobi model
        Restricted master problem.
    is_phase1 : bool
        This is required to call the correct separation problem.
    
    Returns
    model : Gurobi model
    """
    rank = dw.comm.Get_rank()
    dwDone = False
    dualLinkVal = np.empty(dw.numLinkConstrs)
    dualConvVal = np.empty(dw.numBlocks)
    while not dwDone:
        if rank == 0:
            # Solve restricted master centrally and retrive duals
            model.optimize()
            constrs = model.getConstrs()
            linkConstrs = [c for i, c in enumerate(constrs)
                           if "link" in constrs[i].ConstrName]
            dualLinkVal = np.array(model.getAttr('Pi', linkConstrs))
            if dw.convConstrsPresent:
                convConstrs = [c for i, c in enumerate(constrs)
                               if "conv" in constrs[i].ConstrName]
                dualConvVal = np.array(model.getAttr('Pi', convConstrs))
            else:
                dualConvVal = np.zeros(dw.numBlocks)

        dw.comm.Bcast([dualLinkVal, MPI.DOUBLE], root=0)
        dw.comm.Bcast([dualConvVal, MPI.DOUBLE], root=0)

        # Call appropriate separation procedure.
        if isPhase1:
            reducedCost, v = dw.blocks.solvePricingPhase1(dualLinkVal,
                                                          dualConvVal[rank])
        else:
            reducedCost, v = dw.blocks.solvePricing(dualLinkVal,
                                                    dualConvVal[rank])
        isDualFeasibleLocal = reducedCost > -1e-6

        # Add new columns if reduced cost is negative
        if not isDualFeasibleLocal:
            newAv, newcv = dw.blocks.getMasterColumn(v)
            newAv = np.array(newAv)
            newcv = np.array(newcv)
        isDualFeasibleList = dw.comm.gather(isDualFeasibleLocal, root=0)
        if rank == 0 and sum(isDualFeasibleList) == dw.numBlocks:
            dwDone = True
        dwDone = dw.comm.bcast(dwDone, root=0)
        if dwDone:  # all processors break if dwDone is True
            break

        # Split MPI comm based on whether block has added a column.
        # Color is 0 if new column has been added and 1 otherwise.
        # Rank 0 has color 0 regardless of whether a column was added for 
        # block associated with rank 0. This is necessary to be able to access 
        # the master problem model and update it.
        if rank == 0:
            color = 0
            # add dummy values if no columns added at root processor
            if isDualFeasibleLocal:
                newAv = np.zeros(dw.numLinkConstrs)
                newcv = np.zeros(1, dtype='float64')
        else:
            color = isDualFeasibleLocal

        newcomm = dw.comm.Split(color, rank)
        if color == 0:  # if processor added a column
            # Share new columns with processor 0.
            newAvRecv = np.empty([newcomm.Get_size(), dw.numLinkConstrs])
            newcvRecv = np.empty(newcomm.Get_size())
            newcomm.Gather(newAv, newAvRecv, root=0)
            newcomm.Gather(newcv, newcvRecv, root=0)
            if rank == 0 and isDualFeasibleLocal:
                # remove dummy values if root processor didn't add a column
                newAvRecv = newAvRecv[1:, :]
                newcvRecv = newcvRecv[1:]
        newcomm.Free()

        if rank == 0:
            # Update restricted master model.
            addedColInds = [ii for ii in range(dw.numBlocks)
                            if not isDualFeasibleList[ii]]
            for newRank, worldRank in enumerate(addedColInds):
                newAvCurr = newAvRecv[newRank, :].flatten()
                if dw.convConstrsPresent:
                    convConstr = [model.getConstrByName(
                        'conv' + str(worldRank))]
                    newCol = np.hstack([newAvCurr, 1])
                    newGurobiCol = Column(newCol, linkConstrs + convConstr)
                else:
                    newGurobiCol = Column(newAvCurr, linkConstrs)

                newVarName = 'block[' + str(worldRank) + ']'
                newVar = model.addVar(lb=0, ub=np.inf, column=newGurobiCol,
                                      name=newVarName)
                if not isPhase1:
                    model.update()
                    newVar.Obj = newcvRecv[newRank]

    return model

