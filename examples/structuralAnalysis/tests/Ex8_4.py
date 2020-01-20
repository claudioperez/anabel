## Script for Example 8.4 in Structural Analysis
#  force-displacement for determinate braced frame


mdl = em.Model(2,3)

## define structural model (coordinates, connectivity, boundary conditions, element types)
# specify node coordinates (could only specify non-zero terms)
mdl.node('1',  0,  0)  # first node
mdl.node('2',  0,  6)  # second node, etc
mdl.node('3',  4,  6)  #
mdl.node('4',  8,  6)

# connectivity array
mdl.beam('1',  n['1'], n['2'])
mdl.beam('2',  n['2'], n['3'])
mdl.beam('3',  n['3'], n['4'])
mdl.truss('4', n['1'], n['4'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
mdl.boun(n['1'],[1, 1, 1])
mdl.boun(n['4'],[0, 1, 0])

## form static matrix B
B  = em.B_matrix (Model)
# extract submatrix for free dofs Bf
Bf = B(1:Model.nf,:)
# select release locations (index ih) including axial deformations
ih = [2 6]
# continuous deformation locations ic
ic = setdiff(1:sum(Model.nq),ih)

## specify applied forces at free dofs
Pe(2,1) = 20
Pe(3,2) =-30
# plot nodal forces for checking


Pf = Create_NodalForces (Model,Pe)
# solution under applied forces
Q = zeros(sum(Model.nq),1)
Q(ic) = Bf(:,ic)\Pf

## specify element properties
for el in mdl.elems:
   el.E = 1000       # elastic modulus
   el.A = 1e6        # large area for "inextensible" elements a-c
   el.I = 50         # moment of inertia

e['4'].A = 10             # correct area for brace element

## collection of element flexibility matrices
Fs = em.Fs_matrix(mdl)

## determination of free dof displacements
# collection of element deformations for particular solution
Veps = Fs*Q
# determine free dof displacements based on continuous deformations
Af = Bf
Uf = Af(ic,:)\Veps(ic)

## determine release deformations Vh_p
Vh = Af*Uf-Veps
disp('the fictitious release deformations are')
format short e
disp(Vh(ih))

Plot_Model (Model) 
# plot element chords in deformed geometry
PlotOpt.MAGF  = MAGF
PlotOpt.PlNod = 'no'
Plot_Model (Model,Uf,PlotOpt)
# plot deformed shape with fictitious release deformations
ElemData{1}.Release = [010]
ElemData{2}.Release = [001]
PlotOpt.HngSF = 1/3
PlotOpt.HOfSF = 0.6
PlotOpt.NodSF = NodSF
Plot_DeformedStructure (Model,ElemData,Uf,Veps,PlotOpt)