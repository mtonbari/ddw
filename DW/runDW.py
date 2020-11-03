import numpy as np
from argparse import ArgumentParser
import generateInstances as gi
from dw import DW
import helpers as hp
from mpi4py import MPI
import time, sys

if __name__ == '__main__':
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	parser = ArgumentParser()

	parser.add_argument('-g', help = 'Solve using Gurobi', action = 'store_true')
	args = vars(parser.parse_args())
	if rank == 0:
		print(args)

	# linkingData, blockData = gi.test()
	m, n = (5, 100)
	numBlocks = 5
	varsPerBlock = [int(n/float(numBlocks))]*numBlocks
	mb = varsPerBlock[0]
	linkingData, blockData = gi.randomInstance(m, n, mb, varsPerBlock)
	# linkingData, blockData = gi.test()
	instance = DW(linkingData, blockData)
		
	if not args['g']:
		# comm.barrier()
		if rank == 0:
			t1 = time.time()
			print('DW Start')
		instance.solveDW()
		if rank == 0:
			print('DW done')
			sys.stdout.flush()
			obj = instance.masterModel.getObjective().getValue()
			t2 = time.time()
			print('Objective Value')
			print(obj)
			print('\n')
			print('DW time: ' + str(t2-t1))
	elif args['g'] and comm.Get_size() == 1:
		gurobiOut = hp.optimalDWSolve(instance)
		optObj = gurobiOut['objval']
		print(optObj)
		print('Gurobi time: ' + str(gurobiOut['Runtime']))
