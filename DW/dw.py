import numpy as np
import numpy.linalg as npl
from block import Block
import math
from mpi4py import MPI
import parameters
import sys
import time
import helpers as hp
from gurobipy import *


class DW():
    def __init__(self, linkingData, blockData):
        self.params = parameters.Params()
        self.comm = MPI.COMM_WORLD
        self.MPISolve = self.comm.Get_size() > 1    # set to 1 if solving using MPI
        self.numBlocks = len(blockData)
        self.numLinkConstrs = len(linkingData['linkSense'])
        self.linkingData = linkingData
        self.blockData = blockData
        self.preprocess()

        self.linkSense = linkingData['linkSense']
        return


    def preprocess(self):
        for i in range(self.numBlocks):
            if self.blockData[i]['An'].ndim == 1:
                self.blockData[i]['An'] = self.blockData[i]['An'][:,None].T
            if self.blockData[i]['Bn'].ndim == 1:
                self.blockData[i]['Bn'] = self.blockData[i]['Bn'][:,None].T
        self.c = self.linkingData['c'].flatten()
        self.b = self.linkingData['b'].flatten()
        return


    def initBlocks(self):
        self.blocks = []
        if self.MPISolve:
            rank = self.comm.Get_rank()
            self.block = Block(self.blockData[rank])
        else:
            for i in range(self.numBlocks):
                self.blocks.append(Block(self.blockData[i]))


    def phase1(self):
        rank = self.comm.Get_rank()
        Av = self.block.Av[0]   
        cv = self.block.cv[0]

        AvRecv = np.empty([self.comm.Get_size(),self.numLinkConstrs])
        self.comm.Gather(Av, AvRecv, root = 0)

        if rank == 0:
            phase1Model = Model()
            phase1Model.setParam('OutputFlag', 0)
            lams = phase1Model.addVars(self.numBlocks, lb = 0, ub = np.inf, vtype = 'C', name = 'block').values()
            for i in range(self.numLinkConstrs):
                expr = LinExpr(AvRecv[:,i], lams)
                phase1Model.addConstr(expr, self.linkSense[i], self.b.item(i), name = 'link' + str(i))
            for i in range(self.numBlocks):
                phase1Model.addConstr(lams[i], '==', 1, name = 'conv' + str(i))
            phase1Model.update()
        
            numEq = sum([1 for i in range(self.numLinkConstrs) if self.linkSense == '=='])
            s = phase1Model.addVars(self.numLinkConstrs + numEq, lb = 0, ub = np.inf).values()
            objExpr = 0
            for i in range(self.numLinkConstrs):
                linkConstr = phase1Model.getConstrByName('link' + str(i))
                convConstr = phase1Model.getConstrByName('conv' + str(i))
                if self.linkSense[i] == '<':
                    si = phase1Model.addVar(lb = 0, ub = np.inf, column = Column(-1, linkConstr), name = 'dummy')
                    objExpr += si
                elif self.linkSense[i] == '>':
                    si = phase1Model.addVar(lb = 0, ub = np.inf, column = Column(1, linkConstr), name = 'dummy')
                    objExpr += si
                elif self.linkSense[i] == '==':
                    si1 = phase1Model.addVar(lb = 0, ub = np.inf, column = Column(1, linkConstr), name = 'dummy')
                    si2 = phase1Model.addVar(lb = 0, ub = np.inf, column = Column(-1, linkConstr), name = 'dummy')
                    objExpr += si1 + si2
            phase1Model.setObjective(objExpr, GRB.MINIMIZE)
        else:
            phase1Model = None
        phase1Model = self.colGen(phase1Model, isPhase1 = True)
        return phase1Model




    def solveDW(self):
        rank = self.comm.Get_rank()
        self.initBlocks()

        phase1Model = self.phase1()
        
        cv = self.comm.gather(self.block.cv, root = 0)

        # remove dummy variables and initialize objective
        if rank == 0:
            self.masterModel = phase1Model.copy()
            dummyVars = hp.getVarsByName(self.masterModel.getVars(), 'dummy')
            self.masterModel.remove(dummyVars)
            self.masterModel.update()

            allVars = self.masterModel.getVars()
            objExpr = 0
            for i in range(self.numBlocks):
                currVars = hp.getVarsByName(allVars, 'block['+ str(i) +']')
                cvCurr = np.hstack(cv[i][jj] for jj in range(len(cv[i])))
                sys.stdout.flush()
                expr = LinExpr(cvCurr, currVars)
                objExpr += expr
            self.masterModel.setObjective(objExpr, GRB.MINIMIZE)
        else:
            self.masterModel = None
        self.masterModel = self.colGen(self.masterModel, isPhase1 = False)
        return

    def colGen(self, model, isPhase1):
        rank = self.comm.Get_rank()
        dwDone = 0
        dualLinkVal = np.empty(self.numLinkConstrs)
        dualConvVal = np.empty(self.numBlocks)
        while not dwDone:
            if rank == 0:
                model.optimize()
                constrs = model.getConstrs()
                linkConstrs = hp.getConstrsByName(constrs, 'link')
                convConstrs = hp.getConstrsByName(constrs, 'conv')
                dualLinkVal = np.array(model.getAttr('Pi', linkConstrs))
                dualConvVal = np.array(model.getAttr('Pi', convConstrs))
                print(model.getObjective().getValue())


            self.comm.Bcast([dualLinkVal, MPI.DOUBLE], root = 0)
            self.comm.Bcast([dualConvVal, MPI.DOUBLE], root = 0)
            newAv, newcv, isDualFeasible = self.block.solvePricing(dualLinkVal, dualConvVal[rank], isPhase1)
            sys.stdout.flush()
            isDualFeasibleList = self.comm.gather(isDualFeasible, root = 0)

            if rank == 0 and sum(isDualFeasibleList) == self.numBlocks:
                dwDone = 1
            dwDone = self.comm.bcast(dwDone, root = 0)
            if dwDone: # all processors break if dwDone == 1
                break   

            if rank == 0:
                color = 0
                # add dummy values if no columns added at root processor
                if isDualFeasible:
                    newAv = np.zeros(self.numLinkConstrs)
                    newcv = np.zeros(1, dtype = 'float64')
            else:
                color = isDualFeasible

            newcomm = self.comm.Split(color, rank)
            if color == 0: # if processor added a column
                newAvRecv = np.empty([newcomm.Get_size(),self.numLinkConstrs])
                newcvRecv = np.empty(newcomm.Get_size())
                newcomm.Gather(newAv, newAvRecv, root = 0)
                newcomm.Gather(newcv, newcvRecv, root = 0)
                if rank == 0 and isDualFeasible:
                    # remove dummy values if root processor didn't add a column
                    newAvRecv = newAvRecv[1:,:]
                    newcvRecv = newcvRecv[1:]
            newcomm.Free()

            if rank == 0:
                addedColInds = [ii for ii in range(self.numBlocks) if not isDualFeasibleList[ii]]
                for newRank, worldRank in enumerate(addedColInds):
                    newAvCurr = newAvRecv[newRank,:].flatten()
                    newCol = np.hstack((newAvCurr, 1))

                    convConstr = [model.getConstrByName('conv' + str(worldRank))]
                    newVarName = 'block[' + str(worldRank) + ']'
                    newVar = model.addVar(lb = 0, ub = np.inf, column = Column(newCol, linkConstrs + convConstr), name = newVarName)
                    if not isPhase1:
                        model.update()
                        newVar.Obj = newcvRecv[newRank]
        return model