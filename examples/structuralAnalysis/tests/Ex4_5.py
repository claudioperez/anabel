## Script for Example 4.5 additional in Structural Analysis
#  static solution for determinate frame with inclined element




## define structural model (coordinates, connectivity, boundary conditions, element types)
# define model geometry
mdl.node(1,:) = [  0   0]  # first node
mdl.node(2,:) = [  0   4]  # second node, etc
mdl.node(3,:) = [  4   7]  #
   
# element connectivity array
CON(1,:) = [  1   2]
CON(2,:) = [  2   3]

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
BOUN(1,:) = [ 1 1]
BOUN(3,:) = [ 0 1]

# specify element type
ne = length(CON)                       # number of elements
[ElemName{1:ne}] = deal('2dFrm')       # linear 2d frame element

## create Model
Model = Create_SimpleModel (XYZ,CON,BOUN,ElemName)

# plot parameters
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# plot and label model for checking (optional)
Create_Window (WinXr,WinYr)          # open figure window
PlotOpt.PlNod = 'yes'
PlotOpt.PlBnd = 'yes'
PlotOpt.NodSF = NodSF
Plot_Model  (Model,[],PlotOpt)       # plot model
PlotOpt.AxsSF = 0.5                  # scale factor for axis arrow
Label_Model (Model,PlotOpt)          # label model

## form static (equilibrium) matrix B
B  = B_matrix(Model)
# extract submatrix for free dofs
Bf = B(1:Model.nf,:)

## generate equivalent nodal forces Pwf
ElemData{2}.w = [0-10]

# plot and label distributed element load
Create_Window (WinXr,WinYr)       # open figure window
PlotOpt.FrcSF = 3
PlotOpt.TipSF = 1.25
Plot_ElemLoading (Model,ElemData,PlotOpt)
Plot_Model  (Model,[],PlotOpt)    # plot model

# generate applied force vector Pw
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
ScaleM = 1/4    # scale factor for bending moments
MDigt  = 2      # number of significant digits for bending moments

# include effect of element load w by including ElemData in Plot_2dMomntDistr
Create_Window (WinXr,WinYr)       # open figure window
Plot_Model (Model)
Plot_2dMomntDistr (Model,ElemData,Q,[],ScaleM)
Label_2dMoments (Model,Q,[],MDigt)