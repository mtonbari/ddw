"""
Functions to generate block-structured instances. These include
randomly generated instances and the cutting stock problem.
"""


from gurobipy import *
import numpy as np
from os import listdir
import helpers as h
from numpy.random import RandomState


def generate_random_block(num_link_constrs, num_block_constrs, num_vars,
                          block_id,
                          constr_bounds=[-10, 20],
                          constr_block_bounds=[-10, 20],
                          cost_bounds=[-10, 30], prng=None):
    """
    Generate blocks with random constraint matrices and random costs.

    The returned block_data can be used when initializing Block objects.
    
    Parameters
    ----------
    num_link_constrs : int
        Number of linking constraints.
    num_block_constrs : int
        Number of local constraints for this block.
    num_vars : int
        Number of variables for this block.
    constrs_bounds, constrs_block_bounds, cost_bounds : list or tuple
        Contains two values: a lowerbound and upperbound for coeffiencts in the
        linking constraints, local constraints and cost vector, respectively.
    prng : RandomState, optional

    Returns
    -------
    block_data : dict
        Dictionary contains block data that can be used to initialize a Block
        object.
    link_row_totals : array
        Row sum of coefficients of linking constraints. This is needed to
        randomly generate reasonable right-hand sides in the get_link_rhs
        function.
    """
    if prng is None:
        prng = RandomState(block_id)
    link_constrs_mat = prng.randint(constr_bounds[0], constr_bounds[1],
                                    size=(num_link_constrs, num_vars))
    block_constrs_dim = (num_block_constrs, num_vars)
    block_constrs_mat = prng.randint(
        constr_block_bounds[0], constr_block_bounds[1], size=block_constrs_dim)
    block_costs = prng.randint(cost_bounds[0], cost_bounds[1], size=num_vars)
    block_rhs = np.zeros(num_block_constrs)
    for i in range(num_block_constrs):
        row_total = block_constrs_mat[i, :].sum(dtype='float')
        if row_total > 0:
            block_rhs[i] = prng.randint(row_total * 2, row_total * 3)
        elif row_total < 0:
            block_rhs[i] = prng.randint(row_total * 3, row_total * 2)
        else:
            block_rhs[i] = 0
    link_row_totals = np.zeros(num_link_constrs)
    for i in range(num_link_constrs):
        row = link_constrs_mat[i, :]
        link_row_totals[i] = np.sum(row)
    block_data = {}
    block_data["An"] = link_constrs_mat
    block_data["Bn"] = block_constrs_mat
    block_data["bn"] = block_rhs
    block_data["cn"] = block_costs
    block_data["blockSense"] = ["<"] * num_block_constrs
    block_data["lb"] = [0] * num_vars
    block_data["ub"] = [30] * num_vars
    block_data["varType"] = ['C'] * num_vars
    return block_data, link_row_totals


def get_link_rhs(row_totals, prng=None):
    """
    Randomly generate the right-hand side of the linking constraints.
    
    row_totals : array
        Row sum of coefficients of linking constraints.
    prng : RandomState, optional 
    """
    if prng is None:
        prng = RandomState(0)
    # Sum components if row_totals is a list of arrays
    if isinstance(row_totals, list):
        row_totals = sum(row_totals[i] for i in range(len(row_totals)))
    link_rhs = np.zeros((len(row_totals), 1))
    for i, t in enumerate(row_totals):
        if t > 0:
            link_rhs[i] = prng.randint(t * 2, t * 3)
        elif t < 0:
            link_rhs[i] = prng.randint(t * 3, t * 2)
        else:
            link_rhs[i] = 0
    return link_rhs


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


def generateCSP(filepath, instanceNum=1, num_add_stock_lengths=0, prng=None):
    """Read and parse a CSP instance.
    
    Parameters
    ----------
    filepath : str
        Path to file containing instances.
    instanceNum : int
        Instance number to parse within file.
    num_add_stock_lengths : int, optional
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
    min_length = min(stockLengths)
    max_length = max(stockLengths)
    new_stock_lengths = prng.randint(
        min_length, max_length + 1, num_add_stock_lengths).tolist()
    stockLengths += new_stock_lengths
    return stockLengths, weights, demands
