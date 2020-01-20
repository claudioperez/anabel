## Script for Example 12.6 in Structural Analysis
#  upper bound theorem of plastic analysis of one story portal frame

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
mdl.node('1',  0., 0)  # first node
mdl.node('2',  0., 5)  # second node, etc
mdl.node('3',  4., 5)  #
mdl.node('4',  8., 5)  # 
mdl.node('5',  8., 0)  # 
  
# element connectivity array
mdl.beam('1',  1.,  2.)
mdl.beam('2',  2.,  3.)
mdl.beam('3',  3.,  4.)
mdl.beam('4',  4.,  5.)

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
mdl.boun(n['1'], [1, 1, 1])
mdl.boun(n['5'], [1, 1, 1])


## define plastic flexural capacity in column vector (3 per element)
# axial force capacity is "very large" for all elements
# avoid "double hinge" in girder by specifying 120.001 for end i of element c 
Qpl = [ 1e5 150 150 1e5 120 120 1e5 120.001 120 1e5 150 150]

## define loading
Pe(2,1) =  30
Pe(3,2) = -50
Pref = Create_NodalForces(Model,Pe)

## call function for upper bound plastic analysis in FEDEASLab
[lambdac,DUf,DVhp] = PlasticAnalysis_wUBT (Model,Qpl,Pref)
## plot the collapse mode
# -------------------------------------------------------------------------
HngSF = 0.60   # relative size of plastic hinges
OffSF = 0.60   # relative offset of plastic hinges from element end
MAGF  = 50     # magnification factor for collapse mode
# -------------------------------------------------------------------------
Create_Window (WinXr,WinYr)       # open figure window
PlotOpt.PlNod = 'no'
Plot_Model (Model,[],PlotOpt)
PlotOpt.MAGF = MAGF
Plot_Model (Model,DUf,PlotOpt)
PlotOpt.HngSF = HngSF
PlotOpt.HOfSF = OffSF
Plot_PlasticHinges (Model,[],DUf,DVhp,PlotOpt)