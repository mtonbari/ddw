"""
Functions to generate block-structured instances. These include
randomly generated instances and the cutting stock problem.
"""


from gurobipy import *
import numpy as np
from os import listdir
import helpers as h
from numpy.random import RandomState


def generateRandomBlock(numLinkConstrs, numBlockConstrs, numVars,
                          blockID,
                          constrBounds=[-10, 20],
                          constrBlockBounds=[-10, 20],
                          cost_bounds=[-10, 30], prng=None):
    """
    Generate blocks with random constraint matrices and random costs.

    The returned blockData can be used when initializing Block objects.
    
    Parameters
    ----------
    numLinkConstrs : int
        Number of linking constraints.
    numBlockConstrs : int
        Number of local constraints for this block.
    numVars : int
        Number of variables for this block.
    blockID : int
        If prng is None, blockID is used as the random seed.
    constrs_bounds, constrs_block_bounds, cost_bounds : list or tuple
        Contains two values: a lowerbound and upperbound for coeffiencts in the
        linking constraints, local constraints and cost vector, respectively.
    prng : RandomState, optional

    Returns
    -------
    blockData : dict
        Dictionary contains block data that can be used to initialize a Block
        object.
    linkRowTotals : array
        Row sum of coefficients of linking constraints. This is needed to
        randomly generate reasonable right-hand sides in the getLinkRHS
        function.
    """
    if prng is None:
        prng = RandomState(blockID)
    linkConstrsMat = prng.randint(constrBounds[0], constrBounds[1],
                                    size=(numLinkConstrs, numVars))
    blockConstrsDim = (numBlockConstrs, numVars)
    blockConstrsMat = prng.randint(
        constrBlockBounds[0], constrBlockBounds[1], size=blockConstrsDim)
    block_costs = prng.randint(cost_bounds[0], cost_bounds[1], size=numVars)
    block_rhs = np.zeros(numBlockConstrs)
    for i in range(numBlockConstrs):
        row_total = blockConstrsMat[i, :].sum(dtype='float')
        if row_total > 0:
            block_rhs[i] = prng.randint(row_total * 2, row_total * 3)
        elif row_total < 0:
            block_rhs[i] = prng.randint(row_total * 3, row_total * 2)
        else:
            block_rhs[i] = 0
    linkRowTotals = np.zeros(numLinkConstrs)
    for i in range(numLinkConstrs):
        row = linkConstrsMat[i, :]
        linkRowTotals[i] = np.sum(row)
    blockData = {}
    blockData["An"] = linkConstrsMat
    blockData["Bn"] = blockConstrsMat
    blockData["bn"] = block_rhs
    blockData["cn"] = block_costs
    blockData["blockSense"] = ["<"] * numBlockConstrs
    blockData["lb"] = [0] * numVars
    blockData["ub"] = [30] * numVars
    blockData["varType"] = ['C'] * numVars
    return blockData, linkRowTotals


def getLinkRHS(rowTotals, prng=None):
    """
    Randomly generate the right-hand side of the linking constraints.
    
    Parameters
    ----------
    rowTotals : array
        Row sum of coefficients of linking constraints.
    prng : RandomState, optional 
    
    Returns
    -------
    linkRHS : numpy 1-D array
        Randomly generated right-hand side of the linking constraints.
    """
    if prng is None:
        prng = RandomState(0)
    # Sum components if rowTotals is a list of arrays
    if isinstance(rowTotals, list):
        rowTotals = sum(rowTotals[i] for i in range(len(rowTotals)))
    linkRHS = np.zeros((len(rowTotals), 1))
    for i, t in enumerate(rowTotals):
        if t > 0:
            linkRHS[i] = prng.randint(t * 2, t * 3)
        elif t < 0:
            linkRHS[i] = prng.randint(t * 3, t * 2)
        else:
            linkRHS[i] = 0
    return linkRHS


def test():
    """Dummy instance for testing purposes."""
    cn = []
    An = []
    Bn = []
    bn = []
    blocks_sensen = []
    An.append(np.array([6,2]))
    An.append(np.array([4,3]))
    Bn.append(np.array([[1,1],[6,3]]))
    Bn.append(np.array([[1,1],[6,2]]))
    bn.append(np.array([6,24]))
    bn.append(np.array([4,12]))
    cn.append(np.array([-3,-4]))
    cn.append(np.array([-2,-6]))
    c = np.hstack(cn)
    blocks_sensen.append(['<','<'])
    blocks_sensen.append(['<','<'])
    t = np.array([30])
    link_sense = ['==']
    num_blocks = 2
    tol = 1e-3
    blockData = []
    for i in range(num_blocks):
       blockData.append(
           {'An':An[i],'Bn':Bn[i],'bn':bn[i],'cn':cn[i],
            'blockSense':blocks_sensen[i],'block_i':i,
            'varType': ['C']*2, 'numBlocks': 2, 'lb': [0,0],
            'ub': [np.inf,np.inf]})

    linkingData = {'c': c, 'b': t, 'linkSense': link_sense}

    return linkingData, blockData


def generateCSP(filepath, instanceNum=1, numAdditionalLengths=0, prng=None):
    """Read and parse a CSP instance.
    
    Parameters
    ----------
    filepath : str
        Path to file containing instances.
    instanceNum : int
        Instance number to parse within file.
    numAdditionalLengths : int, optional
        Number of stock lengths to add to the problem. Default is 0.
    prng : numpy.RandomState

    Returns
    -------
    stockLengths : list
        Lengths of stocks that can be used.
    weights : list
        Weights of items
    demands : list
        Demands of items.
    """
    if prng is None:
        prng = RandomState()
    foundInstance = False
    weights = []
    demands = []
    stockLengths = []
    stockSupplies = []
    counter = 0
    with open(filepath, "r") as f:
        for _, line in enumerate(f):
            if line.strip():
                s = line.split()
                if "NN=" in s[0] and not foundInstance:
                    info = s[0].split("=")
                    currInstanceNum = int(info[1])
                    if currInstanceNum == instanceNum:
                        foundInstance = True
                elif foundInstance:
                    if "NN=" in s[0]:  # break when we get to a new instance
                        break
                    if counter == 0:
                        numItems = int(s[1])
                        numStockTypes = int(s[2])
                        counter += 1
                    elif counter <= numItems:
                        weights.append(int(s[0]))
                        demands.append(int(s[1]))
                        counter += 1
                    elif counter >= numItems + 1:
                        stockLengths.append(int(s[0]))
                        stockSupplies.append(int(s[1]))
    minLength = min(stockLengths)
    maxLength = max(stockLengths)
    new_stock_lengths = prng.randint(
        minLength, maxLength + 1, numAdditionalLengths).tolist()
    stockLengths += new_stock_lengths
    return stockLengths, weights, demands
