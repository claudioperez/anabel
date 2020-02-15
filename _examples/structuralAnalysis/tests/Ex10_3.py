# Script for Example 10.3 in Structural Analysis
#  displacement method of portal frame

#  =========================================================================================
#  FEDEASLab - Release 5.0, July 2018
#  Matlab Finite Elements for Design, Evaluation and Analysis of Structures
#  Professor Filip C. Filippou (filippou@berkeley.edu)
#  Department of Civil and Environmental Engineering, UC Berkeley
#  Copyright(c) 1998-2018. The Regents of the University of California. All Rights Reserved.
#  =========================================================================================



## define structural model (coordinates, connectivity, boundary conditions, element types)
# specify node coordinates (could only specify non-zero terms)
mdl.node('1',  0,  0)  # first node
mdl.node('2',  0,  8)  # second node, etc
mdl.node('3', 10,  8)  #
mdl.node('4', 20,  8)
mdl.node('5', 20, -2)

# connectivity array
mdl.beam('1', n['1'], n['2'])  # linear 2d frame element
mdl.beam('2', n['2'], n['3'])
mdl.beam('3', n['3'], n['4'])
mdl.beam('4', n['4'], n['5'])

# boundary conditions (1 = restrained,  0 = free)
mdl.boun(n['1'],[1, 1, 1])
mdl.boun(n['5'],[1, 1, 1])


# display model
Plot_Model (Model,[],PlotOpt)
PlotOpt.AxsSF = AxsSF
PlotOpt.LOfSF = 1.6.*NodSF
Label_Model (Model,PlotOpt)

## specify element properties
for el=1:Model.ne
   ElemData{el}.E = 1000        # elastic modulus
   ElemData{el}.A = 1e6         # large area for "inextensible" elements a-d
   ElemData{el}.I = 60          # moment of inertia

# insert release at base of element a
e['1'].Release = [0 1 0]

## 1. Load case: uniformly distributed load in elements a and b
# specify uniformly distributed load in elements a and b
e['1'].w['y']  =  -5
e['2'].w['y']  = -10


# plot and label distributed element load
Plot_ElemLoading (Model,ElemData,PlotOpt)
Plot_Model (Model,[],PlotOpt)    # plot model

# displacement method of analysis
em.S_DisplMethod()
# display and label bending moment distribution

# -------------------------------------------------------------------------
Create_Window(WinXr,WinYr)    # open figure window
Plot_Model(Model)
Plot_2dMomntDistr(Model,ElemData,Q,[],ScaleM)

# plot deformed shape
# display model
Create_Window (WinXr,WinYr)    # open figure window
Plot_DeformedStructure (Model,ElemData,Uf,Ve,PlotOpt)

## 2. Load case: thermal deformation of elements a and b
# no nodal forces
# clear equivalent nodal force vector of previous load case
clear Pwf
# clear distributed load from ElemData
e['1'].w['y']  =  0
e['2'].w['y']  =  0
# specify initial element deformations in ElemData
e['1'].e0['2'] = 0.002
e['2'].e0['2'] = 0.002
# ElemData{1}.Release = [010]

# plot and label initial deformations
Plot_ElemLoading (Model,ElemData,PlotOpt)
Plot_Model  (Model,[],PlotOpt)    # plot model

# displacement method of analysis
S_DisplMethod
# display and label bending moment distribution
Create_Window (WinXr,WinYr)    # open figure window
Plot_Model(Model)
Plot_2dMomntDistr(Model,ElemData,Q,[],ScaleM)
Label_2dMoments(Model,Q,[],MDigt)

# plot curvature distribution
Create_Window (WinXr,WinYr)       # open figure window
Plot_Model(Model)       # original configuration
Plot_2dCurvDistr(Model,ElemData,Q,[],1/2)

# plot deformed shape
Plot_DeformedStructure (Model,ElemData,Uf,Ve,PlotOpt)