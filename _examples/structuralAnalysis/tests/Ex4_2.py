## Script for Example 4.2 in Structural Analysis
#  static solution for determinate space truss


## define structural model (coordinates, connectivity, boundary conditions, element types)
# specify node coordinates (XYZ is a 6 by 3 array for this problem)
mdl = em.Model(3,3)
e = mdl.delems 
n = mdl.dnodes

mdl.node('1',-10, -5.77,  0)   # X, Y and Z coordinate of node 1
mdl.node('2', 10, -5.77,  0)   # X, Y and Z coordinate of node 2, etc
mdl.node('3',  0, 11.55,  0)
mdl.node('4',  0, -3.47, 16)
mdl.node('5',  3,  1.73, 16)
mdl.node('6', -3,  1.73, 16)
   
# connectivity array  (this is a cell array with row vectors for each cell)
# note the syntax of braces for the contents of a particular cell array element
mdl.truss3d('1', n['1'], n['6'])      # element 1 connects nodes 1 and 6
mdl.truss3d('2', n['1'], n['4'])      # element 2 connects nodes 1 and 4, etc
mdl.truss3d('3', n['2'], n['4'])
mdl.truss3d('4', n['2'], n['5'])
mdl.truss3d('5', n['3'], n['5'])
mdl.truss3d('6', n['3'], n['6'])
mdl.truss3d('7', n['4'], n['5'])
mdl.truss3d('8', n['5'], n['6'])
mdl.truss3d('9', n['4'], n['6'])

# boundary conditions (BOUN is a 6 by 3 array, but only the restrained nodes need
# to be specified (1 = restrained,  0 = free)
mdl.boun(n['1'], [1, 1, 1])
mdl.boun(n['2'], [1, 1, 1])
mdl.boun(n['3'], [1, 1, 1])


Plot_Model  (Model,[],PlotOpt);    # plot model
Label_Model (Model);               # label model with default properties

## form static (equilibrium) matrix B
B  = em.B_matrix(Model)
# extract submatrix for free dofs
Bf = B[1:Model.nf,:]

disp(['the size of Bf is ' num2str(size(Bf,1)) ' x ' num2str(size(Bf,2))])
disp(['the rank of Bf is ' num2str(rank(Bf))])

## define loading
# specify nodal forces
Pe(4,5 0 -20];
Pe(5,5 0 -20];
Pe(6,5 0 -20];

# plot nodal forces for checking
Create_Window (WinXr,WinYr);
Plot_Model (Model,[],PlotOpt);
Plot_NodalForces (Model,Pe,PlotOpt);

# generate applied force vector Pf
Pf = Create_NodalForces (Model,Pe);

# solve for basic forces and display the result
Q  = Bf\Pf;


# determine support reactions: the product B*Q delivers all forces at the global dofs
# the upper nf should be equal to the applied forces, the lower nt-nf are the support reactions
disp('B*Q gives');
disp(B*Q);

## plotting results

# plot axial force distribution
Create_Window (WinXr,WinYr);       # open figure window
Plot_Model(Model);
Plot_AxialForces(Model,Q,[],ScaleN);
AxialForcDg = 1;
Label_AxialForces(Model,Q,[],NDigt);