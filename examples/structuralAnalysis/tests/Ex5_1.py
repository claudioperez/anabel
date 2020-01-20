## Script for Example 5.1 in Structural Analysis
#  kinematic solution for statically determinate truss
import ema as em 

mdl = em.Model(2,2)
e = mdl.delems
n = mdl.dnodes

## define structural model (coordinates, connectivity, boundary conditions, element types)
# specify node coordinates
mdl.node('1',  0,  0)  # first node
mdl.node('2',  8,  0)  # second node, etc
mdl.node('3',  0,  6)  #
mdl.node('4',  8,  6)  # 
   
# connectivity array
mdl.truss('1', n['1'], n['2'])
mdl.truss('2', n['1'], n['3'])
mdl.truss('3', n['2'], n['4'])
mdl.truss('4', n['3'], n['4'])
mdl.truss('5', n['1'], n['4'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
mdl.boun(n['1'], [1, 1])
mdl.boun(n['2'], [0, 1])


Plot_Model (Model,[],PlotOpt)     # plot model (optional)

Label_Model (Model,PlotOpt)        # label model (optional)

Plot_Releases(Model,[],[],PlotOpt)

## specify initial deformations
V0 = np.zeros(sum(Model.nq),1)
V0[1] = -0.03
V0[2] =  0.01
V0[3] =  0.02
V0[4] = -0.03
V0[5] =  0.02

## form kinematic matrix A
A  = em.A_matrix(Model)
# extract submatrix for free dofs
Af = A(:,1:Model.nf)

## solve for free dof displacements
Uf = Af\V0

## plot deformed shape of structural model
PlotOpt.MAGF = 30                # magnification factor for deformed configuration
Create_Window (WinXr,WinYr)      # open figure window
Plot_Model(Model)
PlotOpt.PlRel = 'no'
PlotOpt.NodSF = NodSF
Plot_DeformedStructure (Model,[],Uf,[],PlotOpt)