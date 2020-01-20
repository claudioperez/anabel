## Script for Example 4.6 in Structural Analysis
#  static solution for determinate three hinge portal frame under distributed loading


import ema as em 

mdl = em.Model(2,3)
n = mdl.dnodes 
e = mdl.delems 

## define structural model (coordinates, connectivity, boundary conditions, element types)
mdl.node('1',  0,   0)  # first node
mdl.node('2',  0,  10)  # second node, etc
mdl.node('3',  8,  10)  #
mdl.node('4', 16,  10)  # 
mdl.node('5', 16, 2.5)  # 
  
# element connectivity array
mdl.beam('1', n['1'] , n['2'])
mdl.beam('2', n['2'] , n['3'])
mdl.beam('3', n['3'] , n['4'])
mdl.beam('4', n['4'] , n['5'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
mdl.boun(n['1'], [1, 1, 0])
mdl.boun(n['5'], [1, 1, 0])


# plot parameters
# -------------------------------------------------------------------------
Plot_Model  (Model,[],PlotOpt)    # plot model
PlotOpt.AxsSF = 0.5
Label_Model (Model,PlotOpt)    # label model

ElemData{3}.Release = [010]
PlotOpt.HngSF = 0.6
PlotOpt.HOfSF = 1
Plot_Releases (Model,ElemData,[],PlotOpt)

## form static (equilibrium) matrix B
B  = em.B_matrix(Model)
# extract submatrix for free dofs
Bf = B(1:Model.nf,:)

# insert hinge at the left end of element c by removing corresponding column of Bf matrix
iq = setdiff(1:sum(Model.nq),8)
Bf = Bf(:,iq)

## specify nodal forces at free dofs
# nodal forces
Pe(2,1) = 5        # force at node 2 in direction X
# plot nodal forces for checking
Create_Window (WinXr,WinYr)
Plot_Model (Model,[],PlotOpt)

Plot_NodalForces (Model,Pe,PlotOpt)

# generate applied force vector Pf
Pf = Create_NodalForces(Model,Pe)

## generate equivalent nodal forces Pwf
e['3'].w['y'] = -5

# plot and label distributed element load in the window of the nodal force(s)

Plot_ElemLoading (Model,ElemData,PlotOpt)
Plot_Releases (Model,ElemData,[],PlotOpt)

# generate applied force vector Pw
Pwf = Pw(1:Model.nf)

## solve for basic forces
# make sure to re-insert zero at moment release
Q     = np.zeros(12,1)
Q(iq) = Bf\(Pf-Pwf)
# display the result for the basic forces
disp('the basic forces are')
disp(Q)

## determine support reactions
# the product B*Q delivers all forces at the global dofs
# the upper 11 should be equal to the applied forces, the lower 4 are the support reactions
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