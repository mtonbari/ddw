"""
@author = Mohamed El Tonbari

Module to solve large scale block-structured linear programs using
Dantzig-Wolfe decomposition. The master problem can either be solved
centrally as in classical Dantzig-Wolfe or in a distributed fashion
using ADMM. The algorithms are implemented using MPI.
"""


import numpy as np
import numpy.linalg as npl
from block import Block
import math
from mpi4py import MPI
import parameters
import sys
import time
import admm
import vanilla_dw as dw
import helpers as hp
from gurobipy import *


class DistDW():
    def __init__(self, blocks, numBlocks, numLinkConstrs,
                 convConstrsPresent=True, params=None, method="admm",
                 linkConstrsSenses=None, rhs=None):
        """
        Parameters
        ----------
        blocks : Block object or list of Block objects
            If not solving using MPI, then must be a list of Block objects.
            Otherwise, blocks is a single Block object associated with
            the rank.
        numBlocks : int
            Number of blocks. Necessary if solving using MPI.
        numLinkConstrs : int
            number of linking constraints
        convConstrsPresent : bool, optional
            If False, convexity constraints will not be included
        params : Params object, optional
        method : {"admm", "dw"}, optional
            Method to solve the master problem. Set to "dw"To solve the master
            problem centrally (as in classical Dantzig-Wolfe).
        linkConstrsSenses : list, optional
            list of constraint senses ("<", ">" or "==") of the linking
            constraints. Only needed if method is set to "dw".
        rhs : list or array, optional
            Right-hand side of linking constraints. Only needed if method
            is set to "dw".

        """
        if params is None:
            self.params = parameters.Params(
                tolPrimalStart=5e1, tolDualStart=5e1, tolPrimalEnd=5e-2,
                tolDualEnd=5e-3, mu=100, tauInc=2, tauDec=2, rhoStart=100)
        else:
            self.params = params

        self.comm = MPI.COMM_WORLD
        self.MPISolve = self.comm.Get_size() > 1  # set to 1 if using MPI
        if self.MPISolve:
            blocks.MPISolve = True
        else:
            for i in range(numBlocks):
                blocks[i].MPISolve = False
        
        if method == "dw":
            assert self.MPISolve, "Only MPI is supported for classical Dantzig-Wolfe."
        
        self.convConstrsPresent = convConstrsPresent
        self.numBlocks = numBlocks
        self.numLinkConstrs = numLinkConstrs
        self.blocks = blocks
        self.method = method
        if self.method == "dw":
            assert linkConstrsSenses is not None and rhs is not None
            self.masterModel = None
            self.linkConstrsSenses = linkConstrsSenses
            self.rhs = rhs

        self.smallRhoTrack = 0
        return

    def updateRho(self, primalResid, dualResid):
        """Update penalty parameter "rho" in ADMM according to residuals"""
        if npl.norm(dualResid) > self.params.mu * npl.norm(primalResid):
            self.rho = self.params.tauInc * self.rho
        elif npl.norm(primalResid) > self.params.mu * npl.norm(dualResid):
            self.rho = math.ceil(self.rho / float(self.params.tauDec))
        if self.rho < 10:
            self.smallRhoTrack += 1
        if self.smallRhoTrack == 50:
            self.rho = 10
            self.smallRhoTrack = 1
        return self.rho

    def solve(self):
        """Call appropriate solve function based on set method."""
        if self.method == "admm":
            self.solveDistributed()
        elif self.method == "dw":
            self.solveCentral()
        else:
            sys.exit("Not a valid method! Please choose between \"admm\" and \"dw\".")
        return

    def solveDistributed(self):
        """Solve restricted master by solving its dual using ADMM."""
        rank = self.comm.Get_rank()
        dwIter = 0

        if self.MPISolve:
            v = self.blocks.initializePricing()
            self.blocks.initializeDualRMP(v)
            # dualConvVal does not exist if convConstrsPresent is False
            if not self.convConstrsPresent:
                self.blocks.dualConvVal = 0
        else:
            for i in range(self.numBlocks):
                v = self.blocks[i].initializePricing()
                self.blocks[i].initializeDualRMP(v)
                if not self.convConstrsPresent:
                    self.blocks[i].dualConvVal = 0

        self.dualLinkVal = np.zeros(self.numLinkConstrs)
        self.tolPrimal = self.params.tolPrimalStart
        self.tolDual = self.params.tolDualStart
        dwDone = False
        if rank == 0:
            print("##########################################")
            print("Starting tolerances:")
            print("\t Primal ADMM tolerance:", self.tolPrimal)
            print("\t Dual ADMM tolerance:", self.tolDual)
            sys.stdout.flush()
        while(not dwDone):
            admmIter = admm.solve(self)
            # Solve princing subproblems and add new column if reduced cost
            # is less than threshold.
            if self.MPISolve:
                isDualFeasibleSum = np.empty(1, dtype='i')
                reducedCost, v = self.blocks.solvePricing(
                    self.dualLinkVal, self.blocks.dualConvVal)
                # isDualFeasibleLocal = (reducedCost >
                #                        - self.params.tolDualEnd
                #                        * npl.norm(self.blocks.Av))
                isDualFeasibleLocal = reducedCost > -1
                if not isDualFeasibleLocal and admmIter != 1:
                    self.blocks.addColumn(v)
                isDualFeasibleLocal = np.array(isDualFeasibleLocal,
                                                  dtype='i')
                self.comm.Reduce([isDualFeasibleLocal, MPI.INT],
                                 [isDualFeasibleSum, MPI.INT],
                                 op=MPI.SUM, root=0)
            else:
                isDualFeasibleLocal = []
                for i in range(self.numBlocks):
                    reducedCost, v = self.blocks[i].solvePricing(
                        self.dualLinkVal, self.blocks[i].dualConvVal)
                    isDualFeasibleLocalCurr = (reducedCost >
                                               - self.params.tolDualEnd
                                               * npl.norm(self.blocks[i].Av))
                    if not isDualFeasibleLocalCurr:
                        self.blocks[i].addColumn(v)
                    isDualFeasibleLocal.append(isDualFeasibleLocalCurr)
                isDualFeasibleSum = np.sum(isDualFeasibleLocal)

            # Check terminating conditions
            if rank == 0 and self.method == "admm":
                tolChange = False
                if ((isDualFeasibleSum == self.numBlocks or admmIter == 1)
                        and self.tolPrimal > self.params.tolPrimalEnd):
                    self.tolPrimal = self.tolPrimal / 10.0
                    tolChange = True
                if ((isDualFeasibleSum == self.numBlocks or admmIter == 1)
                        and self.tolDual > self.params.tolDualEnd):
                    self.tolDual = self.tolDual / 10.0
                    tolChange = True
                elif ((isDualFeasibleSum == self.numBlocks or admmIter == 1)
                        and self.tolPrimal == self.params.tolPrimalEnd
                        and self.tolDual == self.params.tolDualEnd):
                    dwDone = True
                if not dwDone and tolChange:
                    print("##############")
                    print("Tolerances updated...")
                    print("\t Primal tolerance:", self.tolPrimal)
                    print("\t Dual tolerance:", self.tolDual)
            if self.MPISolve:
                dwDone = self.comm.bcast(dwDone, root=0)
            dwIter += 1
        if rank == 0:
            print("##########################################")
        return

    def solveCentral(self):
        """Solve master problem centrally as in classical Dantzig-Wolfe."""
        rank = self.comm.Get_rank()

        self.blocks.initializePricing()

        # Run phase 1 to get a feasible master
        phase1Model = dw.phase1(self)

        # Remove dummy variables used in phase 1 and initialize objective
        cv = self.comm.gather(self.blocks.cv, root=0)
        if rank == 0:
            self.masterModel = phase1Model.copy()
            currVars = self.masterModel.getVars()
            dummyVars = [v for i, v in enumerate(currVars)
                         if "dummy" in v.VarName]
            self.masterModel.remove(dummyVars)
            self.masterModel.update()

            allVars = self.masterModel.getVars()
            objExpr = 0
            for i in range(self.numBlocks):
                currVarName = 'block[' + str(i) + ']'
                currVars = [v for i, v in enumerate(allVars)
                            if currVarName in v.VarName]
                cvCurr = np.hstack(cv[i][jj] for jj in range(len(cv[i])))
                expr = LinExpr(cvCurr, currVars)
                objExpr += expr
            self.masterModel.setObjective(objExpr, GRB.MINIMIZE)
        else:
            self.masterModel = None

        # Run classical Dantzig-Wolfe algorithm
        self.masterModel = dw.colGen(self, self.masterModel,
                                     isPhase1=False)

    def getPrimalSol(self):
        """Return primal solution at current iteration."""
        try:
            return self.primalSol
        except AttributeError:
            if self.MPISolve:
                rank = self.comm.Get_rank()
                primalSol = self.blocks.getPrimalSol()
                primalSol = self.comm.gather(primalSol, root=0)
                if rank == 0:
                    self.primalSol = np.hstack(
                        [primalSol[i] for i in range(len(primalSol))]).flatten()
                else:
                    self.primalSol = None
            else:
                primalSol = []
                for i in range(self.numBlocks):
                    primalSol.append(self.blocks[i].getPrimalSol())
                self.primalSol = np.hstack([primalSol[i]
                                            for i in range(len(primalSol))])
            return self.primalSol

    def getObjVal(self):
        """Return objective value at current solution."""
        rank = self.comm.Get_rank()
        if self.method == "dw":
            if (self.MPISolve and rank == 0) or not self.MPISolve:
                self.objVal = self.masterModel.getObjective().getValue()
            elif self.MPISolve and rank != 0:
                self.objVal = None

        if self.method == "admm":
            if self.MPISolve:
                objContribution = self.blocks.getObjContribution()
                objContributions = self.comm.gather(objContribution, root=0)
                if rank == 0:
                    self.objVal = sum(objContributions)
                else:
                    self.objVal = None
            else:
                self.objVal = sum(self.blocks[i].getObjContribution()
                                  for i in range(self.numBlocks))
        return self.objVal

    def getLhsVal(self):
        """Return left-hand side of linking constraints.
        
        Returns the product of the constraint matrix and the current
        solution.This is needed to compute the feasibility violation.

        Returns
        -------
        self.lhs : 1-D array
        """
        rank = self.comm.Get_rank()
        if self.method == "dw" and rank == 0:
            constrs = self.masterModel.getConstrs()
            linkConstrs = [c for c in constrs if "link" in c.ConstrName]
            self.lhs = [c.RHS - c.Slack for c in linkConstrs]
        else:
            self.lhs = 0

        if self.method == "admm":
            if self.MPISolve:
                lhsConstribution = self.blocks.getLhsContribution()
                lhsConstributions = self.comm.gather(lhsConstribution, root=0)
            else:
                lhsConstributions = []
                for bl in self.blocks:
                    lhsConstributions.append(bl.getLhsContribution())
                self.lhs = np.vstack([lhs for lhs in lhsConstributions])
                self.lhs = np.sum(self.lhs, axis=0)

            if rank == 0:
                self.lhs = np.vstack([lhs for lhs in lhsConstributions])
                self.lhs = np.sum(self.lhs, axis=0)
            else:
                self.lhs = None
        return self.lhs