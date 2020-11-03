class Params():

	def __init__(self, tolPrimalStart = 5e-1, tolDualStart = 5e-1, tolPrimalEnd = 5e-2, 
					tolDualEnd = 5e-4, mu = 100, tauInc = 2, tauDec = 2, rhoStart = 100):
		self.tolPrimalStart = tolPrimalStart
		self.tolDualStart = tolDualStart
		self.tolPrimalEnd = tolPrimalEnd
		self.tolDualEnd	 = tolDualEnd
		self.mu = mu
		self.tauInc = tauInc
		self.tauDec = tauDec
		self.rhoStart = rhoStart
		return