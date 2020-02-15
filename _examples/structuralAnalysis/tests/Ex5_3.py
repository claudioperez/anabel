## Script for Example 5.3 in Structural Analysis
#  kinematic solution for statically determinate three hinge portal frame


## define structural model (coordinates, connectivity, boundary conditions, element types)
# specify node coordinates
mdl.node('1',  0,  0  )  # first node
mdl.node('2',  0, 10  )  # second node, etc
mdl.node('3',  8, 10  )  #
mdl.node('4', 16, 10  )  # 
mdl.node('5', 16,  2.5)  # 
   
# connectivity array
mdl.beam('1',  n['1'], n['2'])
mdl.beam('2',  n['2'], n['3'])
mdl.beam('3',  n['3'], n['4'])
mdl.beam('4',  n['4'], n['5'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
mdl.boun(n['1'], [1, 1, 1])
mdl.boun(n['5'], [1, 1, 1])

# display model

PlotOpt.NodSF = NodSF
Plot_Model  (Model,[],PlotOpt)    # plot model

## specify element deformations
Veps = zeros(sum(Model.nq),1)
L    = zeros(Model.ne,1)
for el=1:Model.ne
   [xyz] = Localize (Model,el)
   L(el) = ElmLenOr (xyz)

Veps[2:3] = -0.00072.*[-11].*L(1)/2
Veps[5:6] = -0.0006 .*[-11].*L(2)/2
Veps[8:9] = -0.0006 .*[-11].*L(3)/2
# ih = index of release, ic = index for continuous element deformations
ih = [2, 8, 12]  
ic = setdiff(1:sum(Model.nq),ih)

## form kinematic matrix A
A  = em.A_matrix(Model)
# extract submatrix for free dofs and continuous element deformations
Af = A(ic,1:Model.nf)

## solve for free dof displacements
Uf = Af\Veps(ic)

# determine release deformation(s)

Vh = A(:,1:Model.nf)*Uf-Veps
print('the release deformations are')
print(Vh(ih))

## plot deformed shape of structural model

Plot_Model(Model,[],PlotOpt)
# plot element chords in deformed configuration

Plot_Model(Model,Uf,PlotOpt)
PlotOpt.HngSF = HngSF
PlotOpt.HOfSF = HOfSF
PlotOpt.NodSF = NodSF
ElemData{1}.Release = [0,1,0]
ElemData{3}.Release = [0,1,0]
ElemData{4}.Release = [0,0,1]
Plot_DeformedStructure (Model,ElemData,Uf,Veps,PlotOpt)