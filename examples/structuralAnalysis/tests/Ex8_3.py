## Script for Example 8.3 in Structural Analysis
#  force-displacement for simply supported girder with overhang under distributed load



## define structural model (coordinates, connectivity, boundary conditions, element types)
# specify node coordinates
mdl.node('1',  0,  0)
mdl.node('2', 15,  0)
mdl.node('3', 20,  0)
# connectivity array
mdl.beam('1', n['1'],  n['2'])
mdl.beam('2', n['2'],  n['3'])
# boundary conditions
mdl.boun(n['1'], [1, 1])
mdl.boun(n['2'], [0, 1])


# display model
Create_Window (WinXr,WinYr)       # open figure window
PlotOpt.PlNod = 'yes'
PlotOpt.PlBnd = 'yes'
PlotOpt.NodSF = NodSF
Plot_Model (Model,[],PlotOpt)
Plotopt.AxsSF = AxsSF
PlotOpt.LOfSF = 1.8.*NodSF
Label_Model (Model,Plotopt)

## set up static matrix B of structural model
B  = em.B_matrix(Model)
# extract upper nf rows for free dofs
Bf = B(1:Model.nf,:)

## specify element properties
for el in mdl.elems:
   el.E = 1000       # elastic modulus
   el.A = 100        # area does not matter for this problem
   el.I = 20         # moment of inertia
   el.w['y'] = -10    # uniform element load w

# plot and label distributed element load

Plot_ElemLoading(Model,ElemData,PlotOpt)
Plot_Model (Model,[],PlotOpt)    # plot model

# generate applied force vector Pw
Pw  = Create_PwForces (Model,ElemData)
Pwf = Pw(1:Model.nf)

## solve for basic forces Q (homogeneous static solution)
Q  = Bf\(-Pwf)

## display and label bending moment distribution
# -------------------------------------------------------------------------
Create_Window (WinXr,WinYr)    # open figure window
Plot_Model(Model)
Plot_2dMomntDistr(Model,[],Q,[],ScaleM)
Label_2dMoments(Model,Q,[],MDigt)

## collection of element flexibility matrices
Fs = Fs_matrix(Model,ElemData)

## determination of free dof displacements
# initial deformation vector due to element load
V0 = V0_vector(Model,ElemData)
# collection of element deformations
V  = Fs*Q + V0
# free dof displacements under given nodal forces (Af = Bf')
Uf = (Bf')\V

# plot deformed shape of structural model
# display model
Create_Window (WinXr,WinYr)    # open figure window
Plot_Model(Model)

Plot_Model(Model,Uf,PlotOpt)
PlotOpt.NodSF = NodSF
Plot_DeformedStructure (Model,ElemData,Uf,[],PlotOpt)