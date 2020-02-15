## Script for Example 12.3 in Structural Analysis
#  plastic analysis of one story portal frame

#  =========================================================================================
#  FEDEASLab - Release 5.0, July 2018
#  Matlab Finite Elements for Design, Evaluation and Analysis of Structures
#  Professor Filip C. Filippou (filippou@berkeley.edu)
#  Department of Civil and Environmental Engineering, UC Berkeley
#  Copyright(c) 1998-2018. The Regents of the University of California. All Rights Reserved.
#  =========================================================================================

## clear memory and define global variables
mdl = em.Model(2,3)

## define structural model (coordinates, connectivity, boundary conditions, element types)
mdl.node('1',  0,  0) # first node
mdl.node('2',  0,  5) # second node, etc
mdl.node('3',  4,  5) #
mdl.node('4',  8,  5) # 
mdl.node('5',  8,  0) # 
  
# element connectivity array
mdl.beam('1', n['1'], n['2'])
mdl.beam('2', n['2'], n['3'])
mdl.beam('3', n['3'], n['4'])
mdl.beam('4', n['4'], n['5'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
BOUN(1, 1 1 1)
BOUN(5, 1 1 1)


## static matrix Bf for portal frame (Ex 3.3)
Bf = [1/5   1/5    0     0     0     0   1/5   1/5
       0     1     1     0     0     0    0     0 
       0     0   -1/4  -1/4   1/4   1/4   0     0 
       0     0     0     1     1     0    0     0 
       0     0     0     0     0     1    1     0]
   
# specify plastic capacities in vector Qpl
Qpl  = [150 150 120 120 120 120 150 150]
# specify reference load in vector Pref
Pref = [30 0 -50 0  0]

## call function for lower bound plastic analysis in FEDEASLab
[lambdac,Qc] = PlasticAnalysis_wLBT (Bf,Qpl,Pref)

disp([' The collapse load factor is ' num2str(lambdac)])
disp(' The basic forces Q at collapse are')
disp(Qc)

lambdac,Qc = PlasticAnalysis_wLBT (Bf,Qpl,Pref)

disp([' The collapse load factor is ' num2str(lambdac)])
disp(' The basic forces Q at collapse are')
disp(Qc)


## plotting
# display and label bending moment distribution with plastic hinges
# -------------------------------------------------------------------------
ScaleM = 1/4    # scale factor for bending moments
MDigt  = 1      # number of significant digits for bending moments
HngSF  = 0.50   # relative size of plastic hinges
HOfSF  = 0.50   # relative offset of plastic hinges from element end
# -------------------------------------------------------------------------
Create_Window (WinXr,WinYr)       # open figure window
PlotOpt.PlNod = 'no'
PlotOpt.PlBnd = 'no'
Plot_Model (Model)
[Model.Qmis{1:4}] = deal(1,1,1,1)
Plot_2dMomntDistr (Model,[],Qc,[],ScaleM)
# display plastic hinge locations
PlotOpt.HngSF = HngSF
PlotOpt.HOfSF = HOfSF
Plot_PlasticHinges (Model,Qpl,[],Qc,PlotOpt)
Label_2dMoments (Model,Qc,[],MDigt)

ip = [1 4 6 8]
ie = setdiff(1:8,ip)
Be = Bf(:,ie)
Bp = Bf(:,ip)

disp(['the rank of Be matrix is ' num2str(rank(Be))])
disp(['the rank of Pref -Be matrix is ' num2str(rank([Pref -Be]))])