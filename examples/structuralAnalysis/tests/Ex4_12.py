## Script for Example 4.12 in Structural Analysis
#  static solution for cable stayed bridge under construction frame with NOS=2

#  =========================================================================================
#  FEDEASLab - Release 5.0, July 2018
#  Matlab Finite Elements for Design, Evaluation and Analysis of Structures
#  Professor Filip C. Filippou (filippou@berkeley.edu)
#  Department of Civil and Environmental Engineering, UC Berkeley
#  Copyright(c) 1998-2018. The Regents of the University of California. All Rights Reserved.
#  =========================================================================================

## specify each step of the process with Matlab functions


# structure equilibrium matrix for non-trivial free dof's w/o axial forces in a-d 
Bf=[[1/6, 1/6,  0 ,   0 , -1/8 , -0.6,  0.6 ],
    [ 0 ,  0 , 1/6,   0 ,    0 , -0.8,   0  ],
    [ 0 ,  1 ,  1 ,   1 ,    1 ,   0 ,   0  ],
    [ 0 ,  0 ,  0 , -1/6,    0 ,   0 , -0.8 ],
    [ 0 ,  0 ,  0 ,   0 ,   1/8,  0.6, -0.6 ]]

# select redundant basic forces Qx (index ix)
ix = [3, 5]

# define basic forces of primary structure Qi (index ip)
ip = setdiff(1:7,ix)

# extract matrix Bi (equilibrium matrix of primary structure)
Bi = Bf(:,ip)

# check if primary structure is stable
disp(['the rank of the equilibrium matrix of the primary structure is ' num2str(rank(Bi))])

## define loading
Pf = [0,  -20,   0,  0,  0]

# determine basic element forces under the applied loading: particular solution with subscript p
Qp[ip,1] = Bi\Pf
Qp[ix,1] = [0, 0]

# extract matrix Bx
Bx = Bf(:,ix)

# determine force influence matrix for redundant basic forces
# set up 1st column 
Bbarx[ip,1] = -Bi\Bx(:,1)
Bbarx[ix,1] = [10]
# set up 2nd column
Bbarx(ip,2) =-Bi\Bx(:,2)
Bbarx(ix,2) = [0,1]


# display Qp, Bbarx
Qp
Bbarx

## use FEDEASLab functions

# after defining equilibrium matrix Bf use it as first argument to the function BiBxbar_matrix
# to obtain the force influence matrices of the primary structure for the applied forces and redundants
# in order to use specific basic forces as redundants these need to be last among the columns
# of the equilibrium matrix the order is the second argument of BiBxbar_matrix (optional)  
Bbari, Bbarx = em.BbariBbarx_matrix(Bf,ix)
Qp = Bbari*Pf
Bbarx