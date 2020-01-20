## Script for Example 5.4 in Structural Analysis
#  kinematic solution for indeterminate plane truss with NOS=1



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
mdl.truss('5', n['2'], n['3'])
mdl.truss('6', n['1'], n['4'])

# boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
mdl.boun('1', [1, 1])
mdl.boun('2', [0, 1])


# plot parameters
# -------------------------------------------------------------------------
# plot and label model for checking (optional)

Plot_Model  (Model,[],PlotOpt)    # plot model

Label_Model (Model,PlotOpt)       # label model



## form kinematic matrix A
A  = em.A_matrix (Model)
# extract submatrix for free dofs Af
Af = A(:,1:Model.nf)

# specify the element deformations V
V  = [0.02, 0.05, -0.02, 0.05, 0.02, 0.02]

# extract nf rows of kinematic matrix
Ai = Af[1:5,:]
# extract corresponding deformations
Vi = V[1:5]

## solution for free global dof displacements
Uf = Ai\Vi
# display result

print('the free global dof displacements are')
print(Uf)

## determine fictitious release deformation at redundant basic force
Vh = Af@Uf - V
disp(['the fictitious release deformation is ' num2str(Vh(6))])

## plot deformed shape of structural model
MAGF  = 15                        # magnification factor for deformed shape
Create_Window (WinXr,WinYr)       # open figure window
# plot original geometry
Plot_Model (Model)
PlotOpt.MAGF  = MAGF
PlotOpt.LnStl = '-'
PlotOpt.LnClr = 'r'
PlotOpt.NodSF = NodSF
PlotOpt.PlBnd = 'yes'
Plot_Model (Model,Uf,PlotOpt)