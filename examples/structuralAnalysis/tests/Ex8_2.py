## Script for Example 8.2 in Structural Analysis
#  force-displacement for simply supported girder with overhang


## define structural model (coordinates, connectivity, boundary conditions, element types)
# specify node coordinates
mdl.node('1',  0,   0)
mdl.node('2', 15,   0)
mdl.node('3', 20,   0)

# connectivity array
mdl.beam('1',  n['1'], n['2'])
mdl.beam('2',  n['2'], n['3'])

# boundary conditions
mdl.boun(n['1'], [1, 1])
mdl.boun(n['2'], [0, 1])

# specify element type                       

## Model creation
Model = Create_SimpleModel (XYZ,CON,BOUN,ElemName)

# display model
# -------------------------------------------------------------------------
Plot_Model (Model,[],PlotOpt)

Label_Model (Model,PlotOpt)

## set up static matrix B of structural model
B  = B_matrix(Model)
# extract upper nf rows for free dofs
Bf = B(1:Model.nf,:)

## specify applied forces at free dofs
Pe(3,2) =-15
# plot nodal forces for checking

Plot_Model (Model,[],PlotOpt)

# PlotOpt.TipSF = 1.5
Plot_NodalForces (Model,Pe,PlotOpt)

Pf = Create_NodalForces (Model,Pe)

## solve for basic forces Q
Q  = Bf\Pf

## display results
# -------------------------------------------------------------------------
Plot_Model(Model)
Plot_2dMomntDistr(Model,[],Q,[],ScaleM)
Label_2dMoments(Model,Q,[],MDigt)

## specify element properties
ne = Model.ne   # number of elements in structural model
ElemData = cell(ne,1)
for el in mdl.elems:
   el.E = 1000       # elastic modulus
   el.A = 100        # area does not matter for this problem
   el.I = 20         # moment of inertia


## collection of element flexibility matrices
Fs = em.Fs_matrix(Model)

# force influence matrix
Bbar = inv(Bf)
## determination of free dof displacements
# collection of element deformations
V    = Fs@Q
# free dof displacements under given nodal forces
Uf   = Bbar.T@V

# plot deformed shape of structural model
# display model
Create_Window (WinXr,WinYr)    # open figure window

Plot_Model(Model)
Plot_Model(Model,Uf,PlotOpt)
PlotOpt.NodSF = NodSF
Plot_DeformedStructure (Model,ElemData,Uf,[],PlotOpt)