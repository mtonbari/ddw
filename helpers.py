"""
Simple helper functions for Gurobi.
"""


from gurobipy import *
import numpy as np
from scipy.linalg import block_diag
import math
import time


def optimalDWSolve(linkingData, blockData):
    """
    Solve block-structured problems using Gurobi.

    Parameters
    ----------
    linkingData : dict
        Maps:
            -"linkSense" to a list of constraint senses ("<", ">" or "==")
                of the linking constraints
            -"b" to the right-hand side of the linking constraints
                (array or list)
    blockData : dict
        Maps:
            -"An" to the block's constraint matrix in the linking
                constraints (2-D numpy array)
            -"Bn" to the block's local constraint matrix (2-D numpy array)
            -"bn" to the right-hand side of the block's local constraints
                (array or list)
            -"cn" to the costs associated with the block's variables
                (array or list)
            -"blockSense" to a list of constraint senses ("<", ">" or "==")
                of the block's local constraints.
            -"lb" to a list of lower bounds of the block's variables
            -"ub" to a list of upper bounds of the block's variables
            -"varType" to a list of Gurobi types for the block's
                 variables
    
    Returns
    -------
    optimalResults : dict
        Maps "objval", "runtime", and "x" to the objective value, the runtime
        and the optimal solution, respectively.
    """
    model = Model()
    model.setParam("OutputFlag", 0)
    numBlocks = len(blockData)
    c = np.hstack(blockData[i]["cn"].flatten() for i in range(numBlocks))
    t = linkingData['b'].flatten()
    linkSense = linkingData['linkSense']
    An = []
    for i in range(numBlocks):
        An.append(blockData[i]['An'])
        varType = blockData[i]['varType']
        lb = blockData[i]['lb']
        ub = blockData[i]['ub']
        x = model.addVars(len(lb), lb = lb, ub = ub, vtype = varType).values()
        Bn = blockData[i]['Bn']
        bn = blockData[i]['bn']
        blockSense = blockData[i]['blockSense']
        if len(Bn.shape) == 1:
            model.addConstr(LinExpr(Bn, x), blockSense[0], bn.item(0))
        else:
            for j in range(Bn.shape[0]):
                constr = Bn[j,:]
                expr = LinExpr(constr, x)
                model.addConstr(expr, blockSense[j], bn.item(j))
    model.update()
    A = np.hstack(An)
    x = model.getVars()
    for i in range(A.shape[0]):
        constr = A[i,:]
        expr = LinExpr(constr, x)
        model.addConstr(expr, linkSense[i], t.item(i))
    objExpr = LinExpr(c, x)
    model.setObjective(objExpr, GRB.MINIMIZE)
    t1 = time.time()
    model.optimize()
    t2 = time.time()
    optimalResults = {}
    try:
        optimalResults['objval'] = model.getObjective().getValue()
        optimalResults['runtime'] = t2 - t1
        optimalResults['x'] = model.getAttr('X', model.getVars())
    except AttributeError:
        return None, 0
    return optimalResults


def getVarsByName(vars, prefix):
    vars_by_name = [v for i, v in enumerate(vars) if prefix + '[' in vars[i].VarName]
    return vars_by_name

def getConstrsByName(constrs, prefix):
    constrs_by_name = [c for i, c in enumerate(constrs) if prefix + '[' in constrs[i].ConstrName]
    return constrs_by_name
     