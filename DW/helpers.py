# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 09:07:10 2018

@author: Mohamed
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 01 15:11:50 2018

@author: Mohamed
"""
from gurobipy import *
import numpy as np
from scipy.linalg import block_diag
import math
import time

def get_objective(m):
    dvars = m.getVars()
    var_indices = {v: i for i, v in enumerate(dvars)}
    num_vars = m.getAttr('NumVars')
    obj_expr = m.getObjective()
    obj = np.zeros([num_vars,1])
    for i in range(obj_expr.size()):
        dvar = obj_expr.getVar(i)
        j = var_indices[dvar]
        obj[j] = obj_expr.getCoeff(i)
    return obj

#constr_range: slice object
def get_constrs(m, constrs_slice = slice(0,None)):
    dvars = m.getVars()
    constrs = m.getConstrs()[constrs_slice]
    var_indices = {v: i for i, v in enumerate(dvars)}
    num_vars = m.getAttr('NumVars')
    num_constrs = len(constrs)
    A = np.zeros([num_constrs,num_vars])
    b = np.zeros([num_constrs,1])
    
    for i, constr in enumerate(constrs):
        b[i] = constr.RHS
        constr_expr = m.getRow(constr)
        for ii in range(constr_expr.size()):
            dvar = constr_expr.getVar(ii)
            j = var_indices[dvar]
            A[i,j] = constr_expr.getCoeff(ii)
    sense = m.getAttr('sense',constrs)
    for i in range(len(sense)):
        if sense[i] == '=':
            sense[i] = '=='
    return A, b, sense

def set_objective(model,c,q,obj_sense):
    x = model.getVars()
    n = c.size
    if q.size > 0:
        lexpr = sum(c[i]*x[i] for i in range(n))
        qexpr = sum(q[i]*x[i]*x[i] for i in range(n))
        obj = QuadExpr(qexpr + lexpr)
    else:
        obj = LinExpr(sum(c[i]*x[i] for i in range(n)))
        
    model.setObjective(obj, obj_sense)
    return model

#model: Gurobi model
#A: np.array constraint matrix
#x: Gurobi variables
#b: np.array RHS
#sense: list containing sense of inequalities (e.g sense= ['<','='])
def add_constraints(model,A,x,b,sense):
    # Create the constraints
    n = A.shape[1]
    m = A.shape[0]
    for j in range(m):
        model.addConstr(
                sum(A[j][i]*x[i] for i in range(n)), sense[j], b[j])
    return model

def create_model(c,q,A,b,sense,lb_,ub_,var_type,obj_sense):
    model = Model('Model')
    n = int(A.shape[1])
    m = int(A.shape[0])
#    b = b.reshape(m,1)
#    c = c.reshape(n,1)
    model.addVars(n, lb = lb_, ub = ub_, vtype=var_type)
    model.update()
    x = model.getVars()
    #add constraints
    A = A.tolist()
    for i in range(m):
        coeffs = A[i]
        lhs = LinExpr(coeffs,x)
        model.addConstr(lhs, sense[i], b.item(i))
    
    #add objective function
    if q.size > 0:
        lexpr = sum(c[i,0]*x[i] for i in range(n))
        qexpr = sum(q[i]*x[i]*x[i] for i in range(n))
        obj = QuadExpr(qexpr + lexpr)
    else:
        obj = LinExpr(c,x)
        
    model.setObjective(obj, obj_sense)
    model.setParam('OutputFlag', 0)
    model.setParam('Method',0)
    model.setParam('DualReductions',0)
    model.update()
        
    return model   
 
def optimalDWSolve(dw):
    model = Model()
    linkingData = dw.linkingData
    c = linkingData['c'].flatten()
    blockData = dw.blockData
    t = linkingData['b'].flatten()
    linkSense = linkingData['linkSense']
    An = []
    numBlocks = len(blockData)
    for i in range(numBlocks):
        An.append(blockData[i]['An'])
        varType = blockData[i]['varType']
        lb = blockData[i]['lb']
        ub = blockData[i]['ub']
        x = model.addVars(len(lb), lb = lb, ub = ub, vtype = varType).values()
        Bn = blockData[i]['Bn']
        bn = blockData[i]['bn']
        blockSense = blockData[i]['blockSense']
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
        optimalResults['Runtime'] = t2 - t1
        optimalResults['x'] = model.getAttr('X', model.getVars())
    except AttributeError:
        return None, 0
        #    optimal_results['pi'] =  optimal_model.getAttr('Pi', optimal_model.getConstrs())
    return optimalResults


def getVarsByName(vars, prefix):
    vars_by_name = [v for i, v in enumerate(vars) if prefix in vars[i].VarName]
    return vars_by_name

def getConstrsByName(constrs, prefix):
    constrs_by_name = [c for i, c in enumerate(constrs) if prefix in constrs[i].ConstrName]
    return constrs_by_name
        
        