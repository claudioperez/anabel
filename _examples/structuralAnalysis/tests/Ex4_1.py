## Script for Example 4.1 in Structural Analysis
#  static solution for determinate plane truss


## the following example shows the determination of the truss response first without and then
#  with the use of FEDEASLab functions


mdl = em.Model(2,2)
e = mdl.delems 
n = mdl.dnodes

# define model geometry
mdl.node('1',  0.0,  0.0)  # first node
mdl.node('2',  8.0,  0.0)  # second node, etc
mdl.node('3', 16.0,  0.0)  #
mdl.node('4',  8.0,  6.0)  # 
   
# element connectivity array
mdl.truss('1',  n['1'],  n['2']]
mdl.truss('2',  n['2'],  n['3']]
mdl.truss('3',  n['1'],  n['4']]
mdl.truss('4',  n['2'],  n['4']]
mdl.truss('5',  n['3'],  n['4']]

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
BOUN(1, 1 1]
BOUN(3, 0 1]

## form static (equilibrium) matrix B
B  = em.B_matrix(mdl)
# extract submatrix for free dofs
Bf = B(0:mdl.nf,:)

## specify loading
Pe[2,2] = -5    # force at node 2 in direction Y
Pe[4,1] = 10    # force at node 4 in direction X

# plot nodal forces for checking
Create_Window (WinXr,WinYr)
PlotOpt.PlNod = 'no'
PlotOpt.PlBnd = 'yes'
Plot_Model (Model,[],PlotOpt)
PlotOpt.Label ='yes'
PlotOpt.FrcSF = 4
PlotOpt.TipSF = 1.25
Plot_NodalForces (Model,Pe,PlotOpt)

# generate applied force vector Pf
Pf = Create_NodalForces (Model,Pe)

## solution for basic forces and support reactions
# solve for basic forces and display the result
Q  = Bf\Pf
disp('the basic forces are')
disp(Q)

# determine support reactions: the product B*Q delivers all forces at the global dofs
# the upper 5 should be equal to the applied forces, the lower 3 are the support reactions
disp('B*Q gives')
disp(B*Q)


# plot axial force distribution
Create_Window (WinXr,WinYr)       # open figure window
Plot_Model(Model)
Plot_AxialForces(Model,Q,[],ScaleN)
AxialForcDg = 1
Label_AxialForces(Model,Q,[],NDigt)