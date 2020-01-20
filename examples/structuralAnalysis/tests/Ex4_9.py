## Script for Example 4.9 in Structural Analysis
import ema as em
import numpy as np 

#  static solution for indeterminate plane truss with NOS=1

## specify each step of the process with Matlab functions


# structure equilibrium matrix for free dofs (consult equation 2.52) 
Bf=[[1,  0,  0,    0,    0.8,  0 ],
    [0,  0,  0,   -1,   -0.8,  0 ],
    [0,  1,  0,    0,    0.6,  0 ],
    [0,  0,  0,    1,    0  , 0.8],
    [0,  0,  1,    0,    0  , 0.6]]

# select the redundant basic force Qx (index ix)
ix=6

# define basic forces of primary structure Qi (index ip)
ip=1:5

# extract matrix Bi (static matrix of primary structure)
Bi = Bf(:,ip)

# check if primary structure is stable
disp(['the rank of the equilibrium matrix of the primary structure is ' num2str(rank(Bi))])

# define loading
Pf = [0 ; 0 ; 0 ; 15 ; 0]

# determine basic element forces of primary structure under the applied loading
Qp(ip,1) = Bi\Pf
Qp(ix,1) = 0

# extract matrix Bx
Bx = Bf(:,ix)

# determine force influence matrix for redundant basic forces
# set up only column 
Bbarx(ip,1) = -Bi\Bx(:,1)
Bbarx(ix,1) = 1

# display Qp, Bxbar
Qp
Bbarx

## use FEDEASLab functions

# after defining equilibrium matrix Bf use it as first argument to the function BiBxbar_matrix
# to obtain the force influence matrices of the primary structure for the applied forces and redundants
Bibar,Bbarx = BbariBbarx_matrix (Bf)
Qp = Bibar*Pf
Bbarx

## use FEDEASLab functions to also set up Bf matrix and Pf vector


## define structural model (coordinates, connectivity, boundary conditions, element types)
mdl = em.Model(2,2)
e = mdl.delems 
n = mdl.dnodes

# specify node coordinates
mdl.node('1',  0,   0)  # first node
mdl.node('2',  8,   0)  # second node, etc
mdl.node('3',  0,   6)  #
mdl.node('4',  8,   6)  # 
   
# connectivity array
mdl.truss('1', n['1'], n['2'])
mdl.truss('2', n['1'], n['3'])
mdl.truss('3', n['2'], n['4'])
mdl.truss('4', n['3'], n['4'])
mdl.truss('5', n['2'], n['3'])
mdl.truss('6', n['1'], n['4'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
mdl.boun(n['1'], [1, 1])
mdl.boun(n['2'], [0, 1])


## form static (equilibrium) matrix B
B  = em.B_matrix(Model)
# extract submatrix for free dofs
Bf = B(1:Model.nf,:)
# determine particular and homogeneous solution
Bbari,Bbarx = BbariBbarx_matrix (Bf)

## specify applied forces at free dofs
Pe(4,1) = 15

# plot nodal forces for checking
Plot_Model(Model,[],PlotOpt);
Plot_NodalForces(Model,Pe,PlotOpt);

# assign nodal forces to nodal force vector
# particular solution

Qp = Bbari*Pf
# homogeneous solution
Bbarx