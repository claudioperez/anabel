## Script for Example 5.2 in Structural Analysis
#  kinematic solution for statically determinate beam with overhang



## define structural model (coordinates, connectivity, boundary conditions, element types)
# define model geometry
mdl.node('1',  0,  0)  # first node
mdl.node('2', 10,  0)  # second node, etc
mdl.node('3', 20,  0)  #
mdl.node('4', 25,  0)  # 
   
# element connectivity array
mdl.beam('1', n['1'], n['2'])
mdl.beam('2', n['2'], n['3'])
mdl.beam('3', n['3'], n['4'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
BOUN('1', [1, 1])
BOUN('3', [0, 1])


# -------------------------------------------------------------------------
# display model

Plot_Model (Model,[],PlotOpt)     # plot model (optional)
PlotOpt.LOfSF = 2.5.*NodSF
Label_Model (Model,PlotOpt)       # label model (optional)

## specify initial deformations
Veps = np.zeros(sum(Model.nq),1)
La = 10
Lb = 10
Lc = 5
# flexural deformations for element a
Veps(2) = -0.0012*La/2
Veps(3) =  0.0012*La/2
# flexural deformations for element b
Veps(5) =  0.0006*Lb/2
Veps(6) = -0.0006*Lb/2
# flexural deformations for element c
Veps(8) =  0.0008*Lc/2
Veps(9) = -0.0008*Lc/2

## form kinematic matrix A
A  = em.A_matrix(Model)
# extract submatrix for free dofs
Af = A(:,1:Model.nf)

## solve for free dof displacements
Uf = Af\Veps

## plot deformed shape of structural model
MAGF  = 100                       # magnification factor for deformed shape
Create_Window (WinXr,WinYr)       # open figure window

Plot_Model(Model,[],PlotOpt)
# to plot element chords use Plot_Model with free dof displacements Uf

Plot_Model(Model,Uf,PlotOpt)
Plot_DeformedStructure (Model,[],Uf,Veps,PlotOpt)