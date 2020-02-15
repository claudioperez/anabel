## Script for Example 4.3 in Structural Analysis
#  static solution for determinate beam with overhang

import ema as em 

mdl = em.Model(2,3)
n = mdl.dnodes
e = mdl.delems 

## define structural model (coordinates, connectivity, boundary conditions, element types)
# define model geometry
mdl.node('1',  0,  0)  # first node
mdl.node('2', 10,  0)  # second node, etc
mdl.node('3', 20,  0)  #
mdl.node('4', 25,  0)  # 
   
# element connectivity array
mdl.beam('1', n['1'],  n['2'])
mdl.beam('2', n['2'],  n['3'])
mdl.beam('3', n['3'],  n['4'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
mdl.boun(n['1'], [1, 1])
mdl.boun(n['3'], [0, 1])


# plot parameters
# plot and label model for checking (optional)
Create_Window (WinXr,WinYr)       # open figure window
PlotOpt.PlNod = 'yes'
PlotOpt.PlBnd = 'yes'
PlotOpt.NodSF = NodSF
Plot_Model  (Model,[],PlotOpt)    # plot model
PlotOpt.LOfSF = 1.8.*NodSF
Label_Model (Model,PlotOpt)       # label model

## form static (equilibrium) matrix B
B  = B_matrix(Model)
# extract submatrix for free dofs
Bf = B(1:Model.nf,:)

## specify loading
Pe(2,2) = -15        # force at node 2 in direction Y
Pe(4,2) =  -5        # force at node 4 in direction Y
# plot nodal forces for checking

Plot_Model (Model,[],PlotOpt)

Plot_NodalForces (Model,Pe,PlotOpt)

# generate applied force vector Pf
Pf = Create_NodalForces (Model,Pe)

# solve for basic forces and display the result
Q  = Bf\Pf
print('the basic forces are')
print(Q)

# determine support reactions: the product B*Q delivers all forces at the global dofs
# the upper 6 should be equal to the applied forces, the lower 3 are the support reactions
print('B*Q gives')
print(B*Q)


# open window and plot moment diagram M(x)
Create_Window (WinXr,WinYr)       # open figure window
Plot_Model (Model)
Plot_2dMomntDistr (Model,[],Q,[],ScaleM)
Label_2dMoments (Model,Q,[],MDigt)