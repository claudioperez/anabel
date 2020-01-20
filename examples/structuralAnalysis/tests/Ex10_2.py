## Script for Example 10.2 in Structural Analysis
#  displacement method of continuous beam

## clear workspace memory and initialize global variables

## define structural model (coordinates, connectivity, boundary conditions, element types)
# specify node coordinates (could only specify non-zero terms)
mdl.node('1',  0,  0)  # first node
mdl.node('2', 20,  0)  # second node, etc
mdl.node('3', 35,  0)  #
mdl.node('4', 50,  0)

# connectivity array
mdl.beam('1', n['1'], n['2'])
mdl.beam('2', n['2'], n['3'])
mdl.beam('3', n['3'], n['4'])

# boundary conditions (1 = restrained,  0 = free)
mdl.boun(n['1'], [1, 1, 1])
mdl.boun(n['2'], [0, 1, 0])
mdl.boun(n['3'], [0, 1, 0])
mdl.boun(n['4'], [0, 1, 0])


Plot_Model (Model,[],PlotOpt)


for el in mdl.elems:
   el.E = 1000       # elastic modulus
   el.A = 1e6        # large area but irrelevant for transverse loading
   el.I = 60         # moment of inertia


## 1. Load case: uniformly distributed load in elements a and c
# no nodal forces
# specify uniformly distributed load value in ElemData
e['1'].w['y'] = -6
e['3'].w['y'] = -6

# plot and label distributed element load
Plot_ElemLoading (Model,ElemData,PlotOpt)

# displacement method of analysis
S_DisplMethod
# display and label bending moment distribution
# -------------------------------------------------------------------------

Plot_Model(Model)
Plot_2dMomntDistr(Model,ElemData,Q,[],ScaleM)
Label_2dMoments(Model,Q,[],MDigt)

# plot deformed shape of structural model

# display model

Plot_Model(Model,[],PlotOpt)
PlotOpt.NodSF = NodSF
Plot_DeformedStructure (Model,ElemData,Uf,[],PlotOpt)

## 2. Load case: differential heating of elements a and c
# no nodal forces
# clear equivalent nodal force vector of previous load case

# clear uniformly distributed load value in ElemData
e['1'].w['y'] = 0
e['3'].w['y'] = 0
# specify initial deformation due to heating in ElemData
e['1'].e0['2'] = 2e-3
e['3'].e0['2'] = 2e-3

# displacement method of analysis
S_DisplMethod
# display and label bending moment distribution

Plot_2dMomntDistr(Model,[],Q,[],ScaleM)
Label_2dMoments(Model,Q,[],MDigt)

# plot deformed shape
Create_Window (WinXr,WinYr)                  # open figure window
Plot_Model(Model)                            # plot Model
# plot curvature distribution
Plot_2dCurvDistr(Model,ElemData,Q,[],0.70)


Plot_Model(Model,[],PlotOpt)
PlotOpt.NodSF = NodSF
Plot_DeformedStructure (Model,ElemData,Uf,[],PlotOpt)