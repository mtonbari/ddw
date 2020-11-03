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


class CSPBlock():
    """
    Block class to solve the cutting stock problem."""
    def __init__(self, numBlocks, cost, weights, demands,
                 stockLength, maxRolls=None, blockID=None):
        """
        Parameters
        ----------
        numBlocks : int
            Number of blocks in the problem. In the cutting stock problem,
            this is equivalent to the number of different stock lengths.
        cost : float
            Cost of using the stock associated with this block. Note that
            each block has an associated stock length.
        weights : array or list
            Item lengths.
        demands : array or list
        stockLength : float
            Length of stock associated with this block.
        maxRolls : float, optional
            Maximum number of rolls of this length that can be used.
        blockID : float, optional
            Use for debugging purposes when using MPI.
        """
        self.numBlocks = numBlocks 
        self.numItems = len(weights)
        self.cost = cost
        self.weights = weights
        self.demands = demands
        self.stockLength = stockLength
        if maxRolls is None:
            self.maxRolls = self.numItems
        else:
            self.maxRolls = maxRolls
        self.V = []
        self.Av = []
        self.cv = []
        self.maxAv = None
        self.dualLinkVal = np.zeros(self.numItems)
        self.a = np.zeros(self.numItems)
        self.primalSol = None
        self.multipliers = None
        self.blockID = blockID
        return

    def setDualBounds(self):
        """Set bound on dual variables based on cost vector"""
        M = 100 * npl.norm(self.cost)
        lb = [0 for i in range(self.numItems)]
        ub = [M for i in range(self.numItems)]
        self.dualBounds = [lb, ub]
        return self.dualBounds

    def initializePricing(self):
        """Construct Gurobi object modeling the separation problem"""
        self.setDualBounds()
        # initialize pricing model
        self.pricingModel = Model('Pricing')
        self.pricingModel.setParam('OutputFlag', 0)
        self.pricingModel.setParam('Threads', 1)
        x = self.pricingModel.addVars(self.numItems, lb=0, vtype="I").values()
        self.pricingModel.addConstr(LinExpr(self.weights, x),
                                    "<", self.stockLength)

        objExpr = LinExpr(-np.ones(len(x)), x)
        self.pricingModel.setObjective(objExpr, GRB.MINIMIZE)

        self.pricingModel.optimize()
        v = self.pricingModel.getAttr('X', x)
        self.maxAv = sum(v)
        v.append(1)
        self.V.append(v)
        self.Av.append(v[:-1])
        self.cv.append(self.cost)
        return v

    def initializeDualRMP(self, v):
        """Construct Gurobi object modeling the block problem in the Lagrangian
        dUal."""
        # initialize dual RMP model
        self.dualRMPModel = Model('DualRMP')
        self.dualRMPModel.setParam('OutputFlag', 0)
        self.dualRMPModel.setParam('Threads', 1)
        dualLink = self.dualRMPModel.addVars(self.numItems,
                                             lb=self.dualBounds[0],
                                             ub=self.dualBounds[1],
                                             name='dualLink').values()
        self.dualRMPModel.addConstr(LinExpr(v[:-1], dualLink) <= self.cost)
        self.dualRMPModel.update()
        return

    def solvePricing(self, dualLinkVal, dualConvVal=0):
        """Solve separation problem.

        In the cutting stock problem, we can omit the convexity constraints
        in the master problem, thus not requiring dualConvVal.

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
        objExpr = LinExpr(-dualLinkVal, x)
        self.pricingModel.setObjective(objExpr, GRB.MINIMIZE)
        self.pricingModel.optimize()
        objVal = self.pricingModel.getObjective().getValue()
        reducedCost = self.cost + objVal
        v = self.pricingModel.getAttr('X', x) + [1]
        return reducedCost, v

    def solvePricingPhase1(self, dualLinkVal, dualConvVal=0):
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
        objExpr = LinExpr(-dualLinkVal, x)
        self.pricingModel.setObjective(objExpr, GRB.MINIMIZE)
        self.pricingModel.optimize()
        objVal = self.pricingModel.getObjective().getValue()
        reducedCost = objVal
        v = self.pricingModel.getAttr('X', x) + [1]
        return reducedCost, v

    def addColumn(self, v):
        """Add new constraint associated with new extreme point.
        
        Parameters
        ----------
        v : list
           Values of new extreme point.
        """ 
        self.V.append(v)
        self.Av.append(v[:-1])
        self.cv.append(self.cost)
        dualLink = self.dualRMPModel.getVars()
        sys.stdout.flush()
        self.dualRMPModel.addConstr(LinExpr(v[:-1], dualLink) <= self.cost)
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
        newAv = v[:-1]
        newcv = self.cost
        self.Av.append(newAv)
        self.cv.append(newcv)
        return newAv, newcv

    def solveDualRMP(self, dualLinkVal, rho):
        """Solve block problem of the augmented Lagrangian Dual.
        
        Also updates attributes dualLinkVal and dualConvVal.
        """
        dualLink = self.dualRMPModel.getVars()
        objExpr = LinExpr(np.dot((1.0 / self.numBlocks), self.demands)
                          - self.a + rho * dualLinkVal, dualLink)
        objExpr = QuadExpr(objExpr)
        for i in range(len(dualLink)):
            objExpr.addTerms(-rho / 2, dualLink[i], dualLink[i])
        self.dualRMPModel.setObjective(objExpr, GRB.MAXIMIZE)
        self.dualRMPModel.optimize()
        self.dualLinkVal = np.array(self.dualRMPModel.getAttr('X', dualLink))
        return

    def computeDualResid(self, dualLinkVal):
        self.dualResid = npl.norm(self.dualLinkVal - dualLinkVal)**2
        return self.dualResid

    def getPrimalSol(self):
        """Compute primal solution given current convex multipliers."""
        constrs = self.dualRMPModel.getConstrs()
        if self.multipliers is None:
            self.multipliers = self.dualRMPModel.getAttr('Pi', constrs)
        V = np.array(self.V).T
        self.primalSol = V.dot(self.multipliers)
        return self.primalSol

    def getMultipliers(self):
        self.multipliers = self.dualRMPModel.getAttr('Pi', constrs)
        return self.multipliers

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
        y = self.primalSol[-1]
        objContribution = self.cost * y
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
        lhs = self.primalSol[:-1]
        return lhs
