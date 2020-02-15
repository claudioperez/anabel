## Script for Example 8.1 in Structural Analysis
#  force-displacement for statically determinate truss

## define structural model (coordinates, connectivity, boundary conditions, element types)
# specify node coordinates
mdl.node('1',  0,  0)
mdl.node('2', 12,  0)
mdl.node('3',  6,  8)
# connectivity array
mdl.truss('1',  n['1'],  n['2'])
mdl.truss('2',  n['1'],  n['3'])
mdl.truss('3',  n['2'],  n['3'])
# boundary conditions
mdl.boun(n['1'], [1, 1])
mdl.boun(n['2'], [0, 1])


# display model

Plot_Model (Model,[],PlotOpt)

Label_Model (Model,PlotOpt)

PlotOpt.HngSF = 1/3
PlotOpt.HOfSF = 0.6
Plot_Releases (Model,[],[],PlotOpt)

## set up static matrix B of structural model
B  = em.B_matrix(Model)
# extract upper nf rows for free dofs
Bf = B(1:Model.nf,:)

## specify applied forces at free dofs
Pe(3,1) = 20
Pe(3,2) =-10
# plot nodal forces for checking
Create_Window (WinXr,WinYr)

Plot_Model (Model,[],PlotOpt)
PlotOpt.Label='yes'
Plot_NodalForces (Model,Pe,PlotOpt)

Pf = Create_NodalForces (Model,Pe)

## solve for basic forces Q
Q  = Bf\Pf

## display results and label values
ScaleN = 1/5                        # scale factor for axial forces
Create_Window (WinXr,WinYr)         # open figure window
Plot_Model(Model)
Plot_AxialForces(Model,Q,[],ScaleN)
Label_AxialForces(Model,Q)

## specify element properties
for el in mdl.elems:
   el.E = 1000   # elastic modulus
   el.A = 10     # area

e['2'].A = 20    # correct area of element b

## collection of element flexibility matrices
Fs = em.Fs_matrix(Model)

# collection of element deformations
V  = Fs@Q
# free dof displacements under given nodal forces (Af = Bf')
Uf = (Bf.T)\V

# plot deformed shape of structural model
MAGF  = 40                     # magnification factor for deformed shape
# display model
Create_Window (WinXr,WinYr)    # open figure window
PlotOpt.MAGF = MAGF
Plot_Model(Model)

Plot_DeformedStructure(Model,ElemData,Uf,[],PlotOpt)