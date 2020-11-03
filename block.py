"""Example of a block class that can be used in DistDw.

Each function in this example, except solvePricingPhase1 and getMasterColumn,
is required in any block class to solve Dantzig-Wolfe decomposition in
a distributed way (i.e. using ADMM). These functions are used to handle block
data, solve the block problem of the augmented Lagrangian Dual
and solve the seperation problem to add new columns.
"""


from gurobipy import *
import numpy as np
import numpy.linalg as npl
import math


class Block():
    """
    General block class to model any block-structured problem.
    """
    def __init__(self, numBlocks, linkingData, blockData):
        """
        Parameters
        ----------
        numBlocks : int
            number of blocks in the problem
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
        """
        self.linkingData = linkingData
        self.numBlocks = numBlocks
        self.numBlockConstrs = len(blockData['blockSense'])
        self.An = blockData['An']
        self.Bn = blockData['Bn']
        self.bn = blockData['bn'].flatten()
        self.cn = blockData['cn'].flatten()
        self.b = linkingData["b"].flatten()
        self.sense = blockData['blockSense']
        self.lb = blockData['lb']
        self.ub = blockData['ub']
        self.varType = blockData['varType']
        self.V = []  # list of extreme points of block
        self.Av = []  # list of arrays
        self.cv = []
        self.dualLinkVal = np.zeros(len(self.linkingData['linkSense']))
        self.a = np.zeros(self.dualLinkVal.size)
        self.primalSol = None
        self.MPISolve = None
        return

    def setDualBounds(self):
        """Set bound on dual variables based on cost vector"""
        M = npl.norm(self.cn)
        self.dualBounds = [[], []]
        for s in self.linkingData['linkSense']:
            if s == '==':
                self.dualBounds[0].append(-M)
                self.dualBounds[1].append(M)
            elif s == '<':
                self.dualBounds[0].append(-M)
                self.dualBounds[1].append(0)
            elif s == '>':
                self.dualBounds[0].append(0)
                self.dualBounds[1].append(M)
        return

    def initializePricing(self):
        """Construct Gurobi object modeling the separation problem"""
        numThreadsParam = 1
        # initialize pricing model
        self.pricingModel = Model('Pricing')
        self.pricingModel.setParam('OutputFlag', 0)
        self.pricingModel.setParam('Threads', numThreadsParam)
        x = self.pricingModel.addVars(self.cn.size, lb=self.lb, ub=self.ub,
                                      vtype=self.varType).values()
        objExpr = LinExpr(self.cn, x)
        self.pricingModel.setObjective(objExpr, GRB.MINIMIZE)
        if self.numBlockConstrs == 1:
            self.pricingModel.addConstr(LinExpr(self.Bn, x), self.sense[0],
                                        self.bn)
        else:
            for i in range(self.numBlockConstrs):
                expr = LinExpr(self.Bn[i, :], x)
                self.pricingModel.addConstr(expr, self.sense[i], self.bn[i])
        self.pricingModel.optimize()
        v = np.array(self.pricingModel.getAttr('X', x))
        self.V.append(np.array(v))
        self.Av.append(self.An.dot(v))
        self.cv.append(self.cn.dot(v))
        return v


    def initializeDualRMP(self, v):
        """Construct Gurobi object modeling the block problem in the Lagrangian
        dUal."""
        numThreadsParam = 1
        # Initialize dual RMP model
        self.setDualBounds()
        self.dualRMPModel = Model('DualRMP')
        self.dualRMPModel.setParam('OutputFlag', 0)
        self.dualRMPModel.setParam('Threads', numThreadsParam)
        numLinkConstrs = len(self.linkingData["linkSense"])
        dualLink = self.dualRMPModel.addVars(
            numLinkConstrs, lb = self.dualBounds[0], ub=self.dualBounds[1],
            name='dualLink').values()
        dualConv = self.dualRMPModel.addVar(lb=-np.inf, ub=np.inf, name='dualConv')
        self.dualRMPModel.addConstr(
            LinExpr(self.An.dot(v).T, dualLink) + dualConv <= self.cn.T.dot(v))
        self.dualRMPModel.update()
        return

    def solvePricing(self, dualLinkVal, dualConvVal):
        """Solve separation problem.

        Parameters
        ----------
        dualLinkVal : list or array
            Value of the duals associated with the linking constraints.
        dualConvVal : float
            Value of the dual associated with the convexity constraint of this
            block.
        
        Returns
        -------
        reducedCost : float
        v : list
            Potential new extreme point.
        """
        x = self.pricingModel.getVars()
        objExpr = LinExpr(self.cn - self.An.T.dot(dualLinkVal), x)
        self.pricingModel.setObjective(objExpr, GRB.MINIMIZE)
        self.pricingModel.optimize()
        reducedCost = self.pricingModel.getObjective().getValue() - dualConvVal
        v = self.pricingModel.getAttr('X', self.pricingModel.getVars())
        return reducedCost, v

    def solvePricingPhase1(self, dualLinkVal, dualConvVal):
        """Solve separation problem associated with the phase 1 problem.

        This function is only needed if solving the problem using
        vanilla Dantzig-Wolfe decomposition. The phase 1 problem is solved
        until enough columns are added to make the problem feasible. Note that
        only the objective is different from the regular separation problem.
        
        Parameters
        ----------
        dualLinkVal : list or array
            Value of the duals associated with the linking constraints.
        dualConvVal : float
            Value of the dual associated with the convexity constraint of this
            block.
        
        Returns
        -------
        reducedCost : float
        v : list
            Potential new extreme point.
        """   
        x = self.pricingModel.getVars()
        objExpr = LinExpr(- self.An.T.dot(dualLinkVal), x)
        self.pricingModel.setObjective(objExpr, GRB.MINIMIZE)
        self.pricingModel.optimize()
        reducedCost = self.pricingModel.getObjective().getValue() - dualConvVal
        v = self.pricingModel.getAttr('X', self.pricingModel.getVars())
        return reducedCost, v

    def addColumn(self, v):
        """Add new constraint associated with new extreme point.
        
        Parameters
        ----------
        v : list
           Values of new extreme point.
        """ 
        self.V.append(v)
        self.Av.append(self.An.dot(v))
        self.cv.append(self.cn.dot(v))
        xRDM = self.dualRMPModel.getVars()
        dualLink = xRDM[:-1]
        dualConv = xRDM[-1]
        self.dualRMPModel.addConstr(LinExpr(self.An.dot(v).T, dualLink)
                                    + dualConv <= self.cn.T.dot(v))
        return

    def getMasterColumn(self, v):
        """Return the cost and constraint coefficients of extreme point v.
        
        This function is only needed if solving the problem using
        vanilla Dantzig-Wolfe decomposition.

        Parameters
        ----------
        v : list
            Values of extreme point.
        
        Returns
        -------
        newAv : numpy array
            Product of An and v.
        newcv : float
            Dot product of cn and v.
        """
        newAv = self.An.dot(v)
        newcv = self.cn.dot(v)
        self.Av.append(newAv)
        self.cv.append(newcv)
        return newAv, newcv

    def solveDualRMP(self, dualLinkVal, rho):
        """Solve block problem of the augmented Lagrangian Dual.
        
        Also updates attributes dualLinkVal and dualConvVal.
        """
        xRDM = self.dualRMPModel.getVars()
        dualLink = xRDM[:-1]
        dualConv = xRDM[-1]

        objExpr1 = LinExpr((1.0/self.numBlocks)*self.b
                           - self.a + rho*dualLinkVal, dualLink) + dualConv
        objExpr2 = 0
        for i in range(len(dualLink)):
            objExpr2 = objExpr2 + (-rho / 2) * dualLink[i] * dualLink[i]
        objExpr = QuadExpr(objExpr1 + objExpr2)
        self.dualRMPModel.setObjective(objExpr, GRB.MAXIMIZE)
        self.dualRMPModel.optimize()
        dualVal = np.array(
            self.dualRMPModel.getAttr('X', self.dualRMPModel.getVars()))
        self.dualLinkVal = dualVal[:-1]
        self.dualConvVal = dualVal[-1]
        return

    def computeDualResid(self, dualLinkVal):
        self.dualResid = npl.norm(self.dualLinkVal - dualLinkVal)**2
        return self.dualResid

    def getPrimalSol(self):
        """Compute primal solution given current convex multipliers."""
        constrs = self.dualRMPModel.getConstrs()
        multipliers = np.array(self.dualRMPModel.getAttr('Pi', constrs))
        V = np.array(self.V).T
        self.primalSol = V.dot(multipliers)
        return self.primalSol

    def getObjContribution(self):
        """Compute cost contribution of the variables associated with this block.
        
        This is the dot product of cn and the values of the variables of this
        block.

        Returns
        -------
        objContribution : float
        """
        if self.primalSol is None:
            self.getPrimalSol()
        objContribution = self.primalSol.dot(self.cn)
        return objContribution

    def getLhsContribution(self):
        """Compute left-hand side contribution in the linking constraints 
        of the variables of this block.
        
        This is the product of An and the values of the variables of this block.

        Returns
        -------
        LHSContribution : float
        """
        if self.primalSol is None:
            self.getPrimalSol()
        lhs = self.An.dot(self.primalSol)
        LHSContribution = lhs.squeeze()
        return LHSContribution
