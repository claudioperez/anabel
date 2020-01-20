## Script for Example 9.2 in Structural Analysis
#  force method of analysis for braced frame with NOS=2

#  =========================================================================================
#  FEDEASLab - Release 5.0, July 2018
#  Matlab Finite Elements for Design, Evaluation and Analysis of Structures
#  Professor Filip C. Filippou (filippou@berkeley.edu)
#  Department of Civil and Environmental Engineering, UC Berkeley
#  Copyright(c) 1998-2018. The Regents of the University of California. All Rights Reserved.
#  =========================================================================================

## clear memory and define global variables
import ema as em 
mdl = em.Model(2,3)
e = mdl.delems 
n = mdl.dnodes 


## define structural model (coordinates, connectivity, boundary conditions, element types)
# specify node coordinates (could only specify non-zero terms)
mdl.node('1',  0,  0)  # first node
mdl.node('2',  0,  6)  # second node, etc
mdl.node('3',  4,  6)  #
mdl.node('4',  8,  6)

# connectivity array
mdl.beam('1',  n['1'],  n['2'])
mdl.beam('2',  n['2'],  n['3'])
mdl.beam('3',  n['3'],  n['4'])
mdl.truss('4', n['1'],  n['4'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
mdl.boun('1', [1, 1, 1])
mdl.boun('4', [0, 1, 0])

# display model


for el in mdl.elems:
   el.E = 1000       # elastic modulus
   el.A = 1e6        # large area for "inextensible" elements a-c
   el.I = 50         # moment of inertia

e['4'].A = 10        # correct area for brace element

## First load case: applied nodal forces
# specify nodal forces
Pe(2,1) =  20
Pe(3,2) = -30
# plot nodal forces for checking
Create_Window (WinXr,WinYr)
Plot_Model (Model,[],PlotOpt)

Plot_NodalForces (Model,Pe,PlotOpt)


## force method of analysis
# specify index ind_r for redundant basic forces (optional)
ind_r = [2 6]
S_ForceMethod
# display value for redundant basic forces
disp(['the redundant basic forces Qx are ' num2str(Qx')])

# display displacements
disp('free dof displacements Uf under applied nodal forces')
disp(Uf)

## plotting
# display and label bending moment distribution
# -------------------------------------------------------------------------

Plot_Model(Model)
Plot_2dMomntDistr(Model,[],Q,[],ScaleM)
Label_2dMoments(Model,Q,[],MDigt)


Plot_DeformedStructure (Model,ElemData,Uf,[],PlotOpt)

# store value for horizontal translation dof for later use
U1_1 = Uf(Model.DOF(2,1))

## Second load case: unit prestressing force

# define initial deformation vector for second load case (unit prestress)
e['4'].q0 = 1

## force method of analysis
S_ForceMethod
# display value for redundant basic forces
disp(['the redundant basic forces Qx are ' num2str(Qx')])

# display displacements
disp('free dof displacements Uf under applied nodal forces')
disp(Uf)

## plotting
# display and label bending moment distribution
Create_Window (WinXr,WinYr)       # open figure window
Plot_Model(Model)
Plot_2dMomntDistr(Model,[],Q,[],ScaleM)
Label_2dMoments(Model,Q,[],MDigt)

Plot_DeformedStructure (Model,ElemData,Uf,[],PlotOpt)

# store value for horizontal translation dof for later use
U1_2 = Uf(1)

## Third load case: applied nodal forces and prestressing for optimum moment distribution
e['4'].q0 = 43.729
Pf = Create_NodalForces (Model,Pe)

## force method of analysis
S_ForceMethod
# display value for redundant basic forces
disp(['the redundant basic forces Qx are ' num2str(Qx')])

# display displacements
disp('free dof displacements Uf under applied nodal forces')
disp(Uf)

## plotting
# display and label bending moment distribution

Plot_2dMomntDistr(Model,[],Q,[],ScaleM)
Label_2dMoments(Model,Q,[],MDigt)

Plot_DeformedStructure (Model,ElemData,Uf,[],PlotOpt)

## Fourth load case: applied nodal forces and prestressing
# determine prestressing force so as to cancel horizontal translation under applied nodal forces
e['4'].q0 = -U1_1/U1_2

## force method of analysis
S_ForceMethod
# display value for redundant basic forces
disp(['the redundant basic forces Qx are ' num2str(Qx')])

# display displacements
disp('free dof displacements Uf under applied nodal forces')
disp(Uf)

## plotting
# display and label bending moment distribution
Create_Window (WinXr,WinYr)       # open figure window
Plot_Model(Model)
Plot_2dMomntDistr(Model,[],Q,[],ScaleM)
Label_2dMoments(Model,Q,[],MDigt)

# deformed shape
Create_Window (WinXr,WinYr)       # open figure window
MAGF = 100                     # magnification factor for deformed shape
Plot_Model(Model)
PlotOpt.MAGF  = MAGF
PlotOpt.PlNod = 'no'
Plot_Model(Model,Uf,PlotOpt)
PlotOpt.NodSF = NodSF
Plot_DeformedStructure (Model,ElemData,Uf,[],PlotOpt)