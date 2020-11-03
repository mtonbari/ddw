def phase1(dw):
    rank = dw.comm.Get_rank()
    Av = dw.block.Av[0]   
    cv = dw.block.cv[0]

    AvRecv = np.empty([dw.comm.Get_size(),dw.numLinkConstrs])
    dw.comm.Gather(Av, AvRecv, root = 0)

    if rank == 0:
        phase1Model = Model()
        phase1Model.setParam('OutputFlag', 0)
        lams = phase1Model.addVars(dw.numBlocks, lb = 0, ub = np.inf, vtype = 'C', name = 'block').values()
        for i in range(dw.numLinkConstrs):
            expr = LinExpr(AvRecv[:,i], lams)
            phase1Model.addConstr(expr, dw.linkSense[i], dw.b.item(i), name = 'link' + str(i))
        for i in range(dw.numBlocks):
            phase1Model.addConstr(lams[i], '==', 1, name = 'conv' + str(i))
        phase1Model.update()
    
        numEq = sum([1 for i in range(dw.numLinkConstrs) if dw.linkSense == '=='])
        s = phase1Model.addVars(dw.numLinkConstrs + numEq, lb = 0, ub = np.inf).values()
        objExpr = 0
        for i in range(dw.numLinkConstrs):
            linkConstr = phase1Model.getConstrByName('link' + str(i))
            convConstr = phase1Model.getConstrByName('conv' + str(i))
            if dw.linkSense[i] == '<':
                si = phase1Model.addVar(lb = 0, ub = np.inf, column = Column(-1, linkConstr), name = 'dummy')
                objExpr += si
            elif dw.linkSense[i] == '>':  
                si = phase1Model.addVar(lb = 0, ub = np.inf, column = Column(1, linkConstr), name = 'dummy')
                objExpr += si
            elif dw.linkSense[i] == '==':
                si1 = phase1Model.addVar(lb = 0, ub = np.inf, column = Column(1, linkConstr), name = 'dummy')
                si2 = phase1Model.addVar(lb = 0, ub = np.inf, column = Column(-1, linkConstr), name = 'dummy')
                objExpr += si1 + si2
        phase1Model.setObjective(objExpr, GRB.MINIMIZE)
    else:
        phase1Model = None
    phase1Model = dw.colGen(phase1Model, isPhase1 = True)
    return phase1Model


def colGen(dw, model, isPhase1):
    rank = dw.comm.Get_rank()
    dwDone = 0
    dualLinkVal = np.empty(dw.numLinkConstrs)
    dualConvVal = np.empty(dw.numBlocks)
    while not dwDone:
        if rank == 0:
            model.optimize()
            constrs = model.getConstrs()
            linkConstrs = hp.getConstrsByName(constrs, 'link')
            convConstrs = hp.getConstrsByName(constrs, 'conv')
            dualLinkVal = np.array(model.getAttr('Pi', linkConstrs))
            dualConvVal = np.array(model.getAttr('Pi', convConstrs))
            print(model.getObjective().getValue())


        dw.comm.Bcast([dualLinkVal, MPI.DOUBLE], root = 0)
        dw.comm.Bcast([dualConvVal, MPI.DOUBLE], root = 0)
        newAv, newcv, isDualFeasible = dw.block.solvePricing(dualLinkVal, dualConvVal[rank], isPhase1)
        sys.stdout.flush()
        isDualFeasibleList = dw.comm.gather(isDualFeasible, root = 0)

        if rank == 0 and sum(isDualFeasibleList) == dw.numBlocks:
            dwDone = 1
        dwDone = dw.comm.bcast(dwDone, root = 0)
        if dwDone: # all processors break if dwDone == 1
            break   

        if rank == 0:
            color = 0
            # add dummy values if no columns added at root processor
            if isDualFeasible:
                newAv = np.zeros(dw.numLinkConstrs)
                newcv = np.zeros(1, dtype = 'float64')
        else:
            color = isDualFeasible

        newcomm = dw.comm.Split(color, rank)
        if color == 0: # if processor added a column
            newAvRecv = np.empty([newcomm.Get_size(),dw.numLinkConstrs])
            newcvRecv = np.empty(newcomm.Get_size())
            newcomm.Gather(newAv, newAvRecv, root = 0)
            newcomm.Gather(newcv, newcvRecv, root = 0)
            if rank == 0 and isDualFeasible:
                # remove dummy values if root processor didn't add a column
                newAvRecv = newAvRecv[1:,:]
                newcvRecv = newcvRecv[1:]
        newcomm.Free()

        if rank == 0:
            addedColInds = [ii for ii in range(dw.numBlocks) if not isDualFeasibleList[ii]]
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


def solveDW(dw):
        rank = dw.comm.Get_rank()
        dw.initBlocks()

        phase1Model = dw.phase1()
        
        cv = dw.comm.gather(dw.block.cv, root = 0)

        # remove dummy variables and initialize objective
        if rank == 0:
            dw.masterModel = phase1Model.copy()
            dummyVars = hp.getVarsByName(dw.masterModel.getVars(), 'dummy')
            dw.masterModel.remove(dummyVars)
            dw.masterModel.update()

            allVars = dw.masterModel.getVars()
            objExpr = 0
            for i in range(dw.numBlocks):
                currVars = hp.getVarsByName(allVars, 'block['+ str(i) +']')
                cvCurr = np.hstack(cv[i][jj] for jj in range(len(cv[i])))
                sys.stdout.flush()
                expr = LinExpr(cvCurr, currVars)
                objExpr += expr
            dw.masterModel.setObjective(objExpr, GRB.MINIMIZE)
        else:
            dw.masterModel = None
        dw.masterModel = dw.colGen(dw.masterModel, isPhase1 = False)
        return