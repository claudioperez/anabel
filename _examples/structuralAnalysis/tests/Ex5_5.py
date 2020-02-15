## Script for Example 5.5 in Structural Analysis
#  kinematic solution for indeterminate braced frame with NOS=2


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
mdl.boun(n['1'], [1, 1, 1])
mdl.boun(n['4'], [0, 1, 0])



## form kinematic matrix A
A  = em.A_matrix (Model)
# extract submatrix for free dofs Af
Af = A(:,1:Model.nf)

## specify element deformations
Veps = np.zeros(sum(Model.nq),1)
L    = np.zeros(Model.ne,1)
for el in mdl.elems:
   [xyz] = Localize (Model,el)
   L(el) = ElmLenOr (xyz)

Veps(2:3) =  0.002 .*[-11].*L(1)/2
Veps(5:6) =  0.002 .*[-11].*L(2)/2
Veps(8:9) =  0     .*[-11].*L(3)/2
Veps(10)  =  0.01

## determine free dof displacements based on continuous deformations
# select fictitious release locations (index ih) including axial deformations
ih = [2, 6]
# continuous deformation locations ic
ic = setdiff(1:sum(Model.nq),ih)
Uf = Af(ic,:)\Veps(ic)

## determine fictitious release deformations Vh
Vh = Af@Uf - Veps
print('the fictitious release deformations are')

print(Vh[ih])

## plotting
# display model

# plot original geometry
Plot_Model (Model,[],PlotOpt) 
# # plot element chord in deformed geometry
PlotOpt.PlBnd = 'yes'
PlotOpt.NodSF = 1/3
Plot_Model (Model,Uf,PlotOpt)
# plot deformed shape with fictitious release deformations
ElemData{1}.Release = [0,1,0]
ElemData{2}.Release = [0,0,1]
PlotOpt.HngSF = 1/3
PlotOpt.HOfSF = 0.6
PlotOpt.NodSF = 1/3
Plot_DeformedStructure (Model,ElemData,Uf,Veps,PlotOpt)