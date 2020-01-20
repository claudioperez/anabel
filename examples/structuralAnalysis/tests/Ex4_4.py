## Script for Example 4.4 in Structural Analysis
#  static solution for determinate beam with overhang under distributed loading

#  =========================================================================================
#  FEDEASLab - Release 5.0, July 2018
#  Matlab Finite Elements for Design, Evaluation and Analysis of Structures
#  Professor Filip C. Filippou (filippou@berkeley.edu)
#  Department of Civil and Environmental Engineering, UC Berkeley
#  Copyright(c) 1998-2018. The Regents of the University of California. All Rights Reserved.
#  =========================================================================================

## clear memory and define global variables
CleanStart

## define structural model (coordinates, connectivity, boundary conditions, element types)
# define model geometry
mdl.node('1',  0,  0)  # first node
mdl.node('2', 10,  0)  # second node, etc
mdl.node('3', 20,  0)  #
mdl.node('4', 25,  0)  # 
   
# element connectivity array
CON(1,  1   2]
CON(2,  2   3]
CON(3,  3   4]

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
BOUN(1, 1 1]
BOUN(3, 0 1]

# specify element type
ne = length(CON)                       # number of elements
[ElemName{1:ne}] = deal('2dFrm')       # linear 2d frame element

## create Model
Model = Create_SimpleModel (XYZ,CON,BOUN,ElemName)

# plot parameters
# -------------------------------------------------------------------------
WinXr = 0.40  # X-ratio of plot window to screen size 
WinYr = 0.80  # Y-ratio of plot window to screen size
# -------------------------------------------------------------------------
NodSF = 0.60  # relative size for node symbol
# -------------------------------------------------------------------------
# plot and label model for checking (optional)
Create_Window (WinXr,WinYr)            # open figure window
PlotOpt.PlNod = 'yes'
PlotOpt.PlBnd = 'yes'
PlotOpt.NodSF = NodSF
Plot_Model  (Model,[],PlotOpt)         # plot model
PlotOpt.LOfSF = 1.8.*NodSF
Label_Model (Model,PlotOpt)            # label model

## form static (equilibrium) matrix B
B  = B_matrix(Model)
# extract submatrix for free dofs
Bf = B(1:Model.nf,:)

## generate equivalent nodal forces Pwf
ElemData{2}.w = [0-6]
ElemData{3}.w = [0-6]

# plot and label distributed element load
Create_Window (WinXr,WinYr)       # open figure window
PlotOpt.FrcSF = 5
PlotOpt.TipSF = 1.5
Plot_ElemLoading (Model,ElemData,PlotOpt)
PlotOpt.PlNod = 'no'
PlotOpt.PlBnd = 'yes'
Plot_Model  (Model,[],PlotOpt)    # plot model

# generate applied force vector Pw
Pw  = Create_PwForces (Model,ElemData)
Pwf = Pw(1:Model.nf)

# solve for basic forces and display the result
Q  = Bf\(-Pwf)
disp('the basic forces are')
disp(Q)

# determine support reactions: the product B*Q delivers all forces at the global dofs
# the upper 6 should be equal to the applied forces, the lower 3 are the support reactions
disp('B*Q gives')
disp(B*Q)

## plotting results
ScaleM = 1/2    # scale factor for bending moments
MDigt  = 2      # number of significant digits for bending moments

# open window and plot moment diagram for particular solution w/o effect of element load w
Create_Window (WinXr,WinYr)       # open figure window
Plot_Model (Model)
Plot_2dMomntDistr (Model,[],Q,[],ScaleM)
Label_2dMoments (Model,Q,[],MDigt)

# include effect of element load w by including ElemData in Plot_2dMomntDistr
Create_Window (WinXr,WinYr)       # open figure window
Plot_Model (Model)
Plot_2dMomntDistr (Model,ElemData,Q,[],ScaleM)
Label_2dMoments (Model,Q,[],MDigt)