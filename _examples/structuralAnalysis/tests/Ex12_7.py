## Script for Example 12.7 in Structural Analysis
#  upper bound theorem of plastic analysis of one+two story portal frame

## define structural model (coordinates, connectivity, boundary conditions, element types)
mdl = em.Model(2,3)
e = mdl.delems 
n = mdl.dnodes

mdl.node('1',  0.0,   0.0) # first node
mdl.node('2',  0.0,   5.0) # second node, etc
mdl.node('3',  4.0,   5.0) #
mdl.node('4',  8.0,   5.0) # 
mdl.node('5',  8.0,   0.0) # 
mdl.node('6',  8.0,  10.0)
  
# element connectivity array
mdl.beam('1', n['1'], n['2'])
mdl.beam('2', n['2'], n['3'])
mdl.beam('3', n['3'], n['4'])
mdl.beam('4', n['4'], n['5'])
mdl.beam('5', n['4'], n['6'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
mdl.boun(n['1'], [1, 1, 0])
mdl.boun(n['5'], [1, 1, 1])
mdl.boun(n['6'], [1, 1, 1])



## define plastic flexural capacity of elements in a column vector (3 entries per element)
# axial force capacity is "very large" for all elements
# avoid "double hinge" in girder by specifying 120.001 for end j of element b 
Qpl = [ 1e5, 150, 150, 1e5, 120, 120.001, 1e5, 120, 120, 1e5, 150, 150, 1e5, 150, 150]

## form kinematic (compatibility) matrix A
A  = em.A_matrix(Model)
# extract submatrix for free dofs
Af = A[:,1:Model.nf]

## define loading
Pe[2,1] =  50
Pe[3,2] = -60


## call function for upper bound plastic analysis in FEDEASLab
lambdac,DUf,DVhp = PlasticAnalysis_wUBT(Af,Qpl,Pref)

# -------------------------------------------------------------------------
Plot_Model (Model,[],PlotOpt)

Plot_Model (Model,DUf,PlotOpt)

Plot_PlasticHinges (Model,[],DUf,DVhp,PlotOpt)
# print -dpdf -r600 Ex12P7F1