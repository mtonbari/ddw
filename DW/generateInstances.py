# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 15:20:57 2017

@author: Mohamed

"""
from gurobipy import *
import numpy as np
from os import listdir
import helpers as h

def randomInstance(m, n, mb, varsPerBlock, constrBounds = [-10,20], constrBlockBounds = [-10,20], costBounds = [-10,30]):
    np.random.seed(0)
    numBlocks = len(varsPerBlock)
    Bn = []
    An = []
    cn = []
    blockSense = []
    for i in range(numBlocks):
        vpb = varsPerBlock[i]
        An.append(np.random.randint(constrBounds[0], constrBounds[1], size=(m,vpb)))
        Bn.append(np.random.randint(constrBlockBounds[0], constrBlockBounds[1], size=(mb,vpb)))
        # Bn.append(np.identity(vpb))
        cn.append(np.random.randint(costBounds[0], costBounds[1], size = vpb))
        # cn.append(np.random.normal(0,1,size = vpb))
        blockSense.append(['<']*mb)
    c = np.hstack(cn)
    t = np.zeros(m)
    bn = []
    for i in range(m):
        totRow = 0
        for ii in range(numBlocks):
            totRow += An[ii][i,:].sum(dtype='float')
        if totRow > 0:
            t[i] = np.random.randint(totRow*2,totRow*3)
        elif totRow < 0:
            t[i] = np.random.randint(totRow*3,totRow*2)
        else:
            t[i] = 0
    t = t.reshape(m,1)
    for ii in range(numBlocks):
        bn_temp = np.zeros(Bn[ii].shape[0])
        for i in range(mb):
            totRow = Bn[ii][i,:].sum(dtype='float')
            if totRow > 0:
                bn_temp[i] = np.random.randint(totRow*2,totRow*3)
            elif totRow < 0:
                bn_temp[i] = np.random.randint(totRow*3,totRow*2)
            else:
                bn_temp[i] = 0
        bn.append(np.array(bn_temp.reshape(mb,1)))
       # bn.append(np.ones((mb,1)))
    blockData = []
    for i in range(numBlocks):
        varType = ['C']*varsPerBlock[i]
        blockData.append({'An':np.array(An[i]),'Bn':np.array(Bn[i]),'bn':bn[i],'cn':cn[i],
                                        'blockSense':blockSense[i], 'lb': [0]*Bn[i].shape[1], 
                                        'ub': [30]*Bn[i].shape[1], 'varType': varType})
    linkSense = ['>']*m
    linkingData = {'c':c, 'b':t, 'linkSense':linkSense, 'numBlocks':numBlocks}
    return linkingData, blockData

def generate_file_instance(folder_name):
    filenames = listdir(os.path.join(os.getcwd(),'Instances'))
    num_blocks = len(filenames)
    block_data = []
    for i in range(num_blocks):
        model = read(os.path.join(os.getcwd(),'Finished instances',filename))
        temp_dict = {}
        Bn_temp,bn_temp,sense_temp = h.get_constrs(model)
        cn_temp = h.get_objective(model)
        temp_dict['Bn'] = Bn_temp
        temp_dict['bn'] = bn_temp
        temp_dict['cn'] = cn_temp
        temp_dict['blocks_sensen'] = sense_temp
        temp_dict['block_i'] = i

def generate_mps(filepath):
    model = read(filepath)
    A, b, sense = h.get_constrs(model)
    x = model.getVars()
    var_type = [x[i].VType for i in range(len(x))]
    lb = [x[i].LB for i in range(len(x))]
    ub = [x[i].UB for i in range(len(x))]
    c = h.get_objective(model)
    block_data = []
    block_data.append({'An':np.matrix(A), 'cn':np.matrix(c)})
    global_data = {'c':c, 'b':b, 'link_sense':sense, 'var_type':var_type, 'num_blocks':1,
               'lb':lb, 'ub':ub}
    return global_data, block_data


def test():
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
       blockData.append({'An':An[i],'Bn':Bn[i],'bn':bn[i],'cn':cn[i],'blockSense':blocks_sensen[i],'block_i':i,
                            'varType': ['C']*2, 'numBlocks': 2, 'lb': [0,0],'ub': [np.inf,np.inf]})

    linkingData = {'c': c, 'b': t, 'linkSense': link_sense}

    return linkingData, blockData


def generateBPP(filepath):
    file = open(filepath, mode = 'r')
    data = []
    for line in file:
    	data.append(int(line))
    numItems = data[0]
    binCap = data[1]
    weights = data[2:]
    numBins = len(weights)
    numBlocks = numBins
    An = []
    Bn = []
    bn = []
    lb = []
    ub = []
    cn = []
    for i in range(numBlocks):
        temp = np.eye(numItems)
        An.append(np.hstack((temp,np.zeros([numItems,1]))))
        Bn.append(np.hstack((weights,-binCap)))
        bn.append(np.array(0))
        lb.append([0]*Bn[i].size)
        ub.append([1]*Bn[i].size)
        cn.append(np.hstack((np.zeros(numItems),1)))
    b = np.ones(numItems)
    c = np.hstack(cn[i] for i in range(numBlocks))
    linkSense = ['==']*numItems
    blockSense = ['<']
    blockData = []
    for i in range(numBlocks):
        blockData.append({'An':An[i],'Bn':Bn[i],'bn':bn[i],'cn':cn[i],'blockSense':blockSense,
                            'varType': ['C']*(numItems*numBins+numBins), 'lb': lb[i],'ub': ub[i]})
    linkingData = {'c': c, 'b':b, 'linkSense': linkSense}
    return linkingData, blockData
