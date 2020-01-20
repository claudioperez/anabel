## Script for Example 10.1 in Structural Analysis
#  displacement method of analysis for plane truss

#  =========================================================================================
#  FEDEASLab - Release 5.0, July 2018
#  Matlab Finite Elements for Design, Evaluation and Analysis of Structures
#  Professor Filip C. Filippou (filippou@berkeley.edu)
#  Department of Civil and Environmental Engineering, UC Berkeley
#  Copyright(c) 1998-2018. The Regents of the University of California. All Rights Reserved.
#  =========================================================================================

import ema as em 

mdl = em.Model(2,2)
e = mdl.delems
n = mdl.dnodes

## define structural model (coordinates, connectivity, boundary conditions, element types)
# specify node coordinates (could only specify non-zero terms)
mdl.node('1',  0,  0)  # first node
mdl.node('2',  8,  0)  # second node, etc
mdl.node('3', 11,  4)  #
mdl.node('4',  8,  8)
mdl.node('5',  2,  8)

# connectivity array
mdl.truss('1', n['1'], n['2'])
mdl.truss('2', n['2'], n['3'])
mdl.truss('3', n['3'], n['4'])
mdl.truss('4', n['2'], n['5'])

# boundary conditions (1 = restrained,  0 = free)
mdl.boun(n['1'], [1, 1, 0])
mdl.boun(n['2'], [0, 1, 0])
mdl.boun(n['3'], [1, 0, 0])
mdl.boun(n['4'], [1, 1, 0])
mdl.boun(n['5'], [1, 1, 0])

# display model
Plot_Model (Model,[],PlotOpt)


for el in mdl.elems:
   el.E = 1000     # elastic modulus
   el.A = 10       # area

e['1'].A = 20            
e['4'].A = 30

## 1. Load case: applied nodal forces
Pf = [85]
Loading.Pref = Pf
# plot nodal force for checking
Create_Window (WinXr,WinYr)
Plot_Model (Model,[],PlotOpt)
PlotOpt.Label='yes'
PlotOpt.FrcSF = 4
PlotOpt.TipSF = 1.25
Plot_NodalForces (Model,Loading,PlotOpt)

# displacement method of analysis
S_DisplMethod
# display displacements
disp('the free dof displacements Uf under applied nodal forces are')
disp(Uf)
# display basic forces
disp('the basic forces under the applied nodal forces are')
disp(Q)
# determine support reactions
Pr  = A'*Q     # resisting force vector
# check equilibrium at free dofs
Pu  = Pf - Pr(1:Model.nf)     # unbalance force vector
Res = sqrt(Pu'*Pu)
disp(['the error of the equilibrium equations is ' num2str(Res)])
# support reactions
R   = Pr(Model.nf+1:end)
disp('the support reactions under the applied nodal forces are')
disp(R)
 
## 2. Load case: thermal heating of elements c and d
Pf = [00]
ElemData{3}.e0 = 0.001
ElemData{4}.e0 = 0.001
# displacement method of analysis
S_DisplMethod
# display displacements
disp('the free dof displacements Uf under thermal heating are')
disp(Uf)
# display basic forces
disp('the basic forces under thermal heating are')
disp(Q)
# determine support reactions
Pr  = A'*Q     # resisting force vector
# check equilibrium at free dofs
Pu  = Pf - Pr(1:Model.nf)     # unbalance force vector
Res = sqrt(Pu'*Pu)
disp(['the error of the equilibrium equations is ' num2str(Res)])
# support reactions
R   = Pr(Model.nf+1:end)
disp('the support reactions under thermal heating are')
disp(R)