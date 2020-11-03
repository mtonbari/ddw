from gurobipy import *
import numpy as np
import numpy.linalg as npl
import math
import helpers as hp

class Block():
	def __init__(self, blockData):
		self.numBlockConstrs = len(blockData['blockSense'])
		self.An = blockData['An']
		self.Bn = blockData['Bn']
		self.bn = blockData['bn'].flatten()
		self.cn = blockData['cn'].flatten()
		self.sense = blockData['blockSense']
		self.lb = blockData['lb']
		self.ub = blockData['ub']
		self.varType = blockData['varType']
		self.V = []
		self.Av = []
		self.cv = []
		self.initializePricing()
		return


	def initializePricing(self):
		self.pricingModel = Model('Pricing')
		self.pricingModel.setParam('OutputFlag', 0)
		self.pricingModel.setParam('Threads', 0)
		x = self.pricingModel.addVars(self.cn.size, lb = self.lb, ub = self.ub,
								vtype = self.varType).values()
		objExpr = LinExpr(self.cn, x)
		self.pricingModel.setObjective(objExpr, GRB.MINIMIZE)
		for i in range(self.numBlockConstrs):
			expr = LinExpr(self.Bn[i,:], x)
			self.pricingModel.addConstr(expr, self.sense[i], self.bn[i])
		self.pricingModel.optimize()
		v = np.array(self.pricingModel.getAttr('X', x))
		self.V.append(np.array(v))
		self.Av.append(self.An.dot(v))
		self.cv.append(self.cn.dot(v))
		return

	def solvePricing(self, dualLinkVal, dualConvVal, isPhase1):
		x = self.pricingModel.getVars()
		if isPhase1:
			obj = 0
		else:
			obj = self.cn
		objExpr = LinExpr(obj - self.An.T.dot(dualLinkVal), x)
		self.pricingModel.setObjective(objExpr, GRB.MINIMIZE)
		self.pricingModel.optimize()
		if(self.pricingModel.getObjective().getValue() - dualConvVal <= -1e-8):
			v = np.array(self.pricingModel.getAttr('X', self.pricingModel.getVars()))
			self.V.append(v)
			self.Av.append(self.An.dot(v))
			self.cv.append(self.cn.dot(v))
			return self.Av[-1], self.cv[-1], 0
		else:
			return None, None, 1


	def getPrimalSol(self):
		multipliers = np.array(self.dualRMPModel.getAttr('Pi', self.dualRMPModel.getConstrs()))
		V = np.array(self.V).T
		self.primalSol = V.dot(multipliers)
		return self.primalSol