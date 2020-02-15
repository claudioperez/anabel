## Script for Example 12.2 in Structural Analysis
#  plastic analysis of column-girder assembly


## define structural model (coordinates, connectivity, boundary conditions, element types)
mdl.node('1',  0,  0)  # first node
mdl.node('2',  0,  6)  # second node, etc
mdl.node('3',  4,  6)  #
mdl.node('4',  8,  6)  # 
  
# element connectivity array
mdl.beam('1', n['1'], n['2'])
mdl.beam('2', n['2'], n['3'])
mdl.beam('3', n['3'], n['4'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
mdl.boun(1, [1, 1, 1])
mdl.boun(4, [0, 1, 0])



## static matrix Bf for column-girder (Ex 3.2)
Bf = [[1/6,  1/6,    0 ,    0 ,    0 ],
      [ 0 ,   1 ,    1 ,    0 ,    0 ],
      [ 0 ,   0 ,  -1/4,  -1/4,   1/4],
      [ 0 ,   0 ,    0 ,    1 ,    1 ],]  
# specify plastic capacities in vector Qpl
Qpl  = [160 160 120 120 120]
# specify reference load in vector Pref
Pref = [20 0 -20 0]

# [lambdac,Qc]  = PlasticAnalysis_wLBT (Bf,Qpl,Pref)
# disp('the basic forces Qc at collapse are')
# disp(Qc')

lambdac,Qc  = PlasticAnalysis_wLBT(Bf,Qpl,Pref)
disp('the basic forces Qc at collapse are')
disp(Qc)

# open new window
Create_Window (WinXr,WinYr)       # open figure window
# plot model 
Plot_Model (Model,[],PlotOpt)
# display and label bending moment distribution with plastic hinges
# -------------------------------------------------------------------------

Model.Qmis{1} = 1
Model.Qmis{2} = 1
Model.Qmis{3} = [1 3]
Plot_2dMomntDistr (Model,[],Qc,[],ScaleM)
# display plastic hinge locations

Plot_PlasticHinges (Model,Qpl,[],Qc,PlotOpt)
Label_2dMoments (Model,Qc,[],MDigt)

## braced frame
mdl.truss('4',  n['1']   n['4']) # 2d truss element


# plot and label model for checking (optional)
Create_Window (WinXr,WinYr)       # open figure window
Plot_Model  (Model,[],PlotOpt)
Label_Model (Model,PlotOpt)

# structure equilibrium matrix without trivial and w/o axial forces in a-c 
Bf = [[1/6, 1/6,   0 ,   0 ,  0 , 0.8],
      [ 0 ,  1 ,   1 ,   0 ,  0 ,  0 ],
      [ 0 ,  0 , -1/4, -1/4, 1/4,  0 ],
      [ 0 ,  0 ,   0 ,   1 ,  1 ,  0 ],]
  
# specify plastic capacities in vector Qpl
Qpl  = [160, 160, 120, 120, 120, 50]

Pref = [20, 0, -20, 0]

lambdac, Qc = PlasticAnalysis_wLBT (Bf,Qpl,Pref)

# -------------------------------------------------------------------------
Model.Qmis{1} = 1
Model.Qmis{2} = 1
Model.Qmis{3} = [1 3]
# open new window and plot moment diagram and plastic hinge locations label moment values
Create_Window (WinXr,WinYr)       # open figure window

Plot_2dMomntDistr (Model,[],Qc,[],ScaleM)
PlotOpt.HngSF = HngSF
PlotOpt.HOfSF = HOfSF
Plot_PlasticHinges (Model,Qpl,[],Qc,PlotOpt)
Label_2dMoments (Model,Qc,[],MDigt)