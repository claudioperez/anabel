# ===========================================================================
# FEDEASLab - Release 5.2, July 2021
# Matlab Finite Elements for Design, Evaluation and Analysis of Structures
# Professor Filip C. Filippou (filippou@berkeley.edu)
# Department of Civil and Environmental Engineering, UC Berkeley
# Copyright(c) 1998-2021. The Regents of the University of California. 
# All Rights Reserved.
# ===========================================================================
# function added by Jade Cohen                                        06-2021                                                       
from anabel.template import template
import jax.numpy as jnp


class DamageWrapper:
    schema = {
    }

@template(dim="shape")
def DmgEvowNpnd(
  n,
  # damage evolution rule
  dFun,
  # damage threshold tolerance
  gtol = 1e-8,
  dtol = 1e-4,
  **DmgData
):
#    DmgData = Check_DmgParam (DmgData);
  shape = ((n,1),(n,1))

  # damage parameters
  dp   = DmgData["dp"]
  # repeat cycle damage parameters
  Cwc  = DmgData["Cwc"]
  # +ve-ve coupling damage parameters
  Ccd  = DmgData["Ccd"]
  # interaction matrices (nxn, with zeros on the diagonal)
  Cin_pp  = jnp.asarray(DmgData["Cin_pp"])  # Cin_pp(i,j) : effect of sj+ over si+
  Cin_pn  = jnp.asarray(DmgData["Cin_pn"])  # Cin_pn(i,j) : effect of sj+ over si-
  Cin_np  = jnp.asarray(DmgData["Cin_np"])  # Cin_np(i,j) : effect of sj- over si+
  Cin_nn  = jnp.asarray(DmgData["Cin_nn"])  # Cin_nn(i,j) : effect of sj- over si-

  # dissipated energy at d=0 and d=1
  psi_d0 = jnp.asarray(DmgData["psi_d0"])
  psi_d1 = jnp.asarray(DmgData["psi_d1"])
  # fracture energy
  Frac = DmgData["Frac"]
  
  state = {'d'    : jnp.zeros((n,2)), # damage indices (+ve, -ve)
           'se'   : jnp.zeros((n,1)), # effective forces
           'eEx'  : jnp.zeros((n,2)), # extreme values of total deformation 'eEx'
           'psi'  : jnp.zeros((n,2)), # energy dissipation
           'psiEx': psi_d0}           # extreme values of energy dissipation 
  
  origin = state["se"], state["se"], state

  def main(e, s, state, **kwds):
      ## extract past history variables
      # all history variables are stored as follows: +ve(N;M) -ve(N;M)
      d     = state["d"]       # nx2
      eEx   = state["eEx"]     # nx2
      psiP  = state["psi"]     # nx2
      psiEx = state["psiEx"]   # nx2
      seP   = state["se"]      # effective force at last convergence

      ## extract effective state variables
      # effective forces and tangent stiffness matrix
      se = DmgState["se"]                # effective force
      kt = DmgState["kt"]                # tangent stiffness
      #DmgState.Pres.se = se           # store effective force in history

      # deformations
      if strcmpi(DmgData["DOpt"],'total'):
          e  = DmgState["e"]             # nx1 : total deformation
          De = DmgState["De"]            # nx1 : deformation increment from last convergence
          eP = e - De                 # nx1 : total deformation at last convergence
          EF = jnp.eye(n)             # nxn : stiffness factor
      else:
          fe = DmgState.fe            # elastic flexibility
          e  = DmgState.ep            # nx1 : plastic deformation
          eP = DmgState.epP           # nx1 : plastic deformation at last convergence
          De = e - eP                 # nx1 : plastic deformation increments
          EF = jnp.eye(n) - fe@kt     # nxn : stiffness factor


      ## splitting of effective hinge forces into +ve and -ve components (N, Mi, Mj)
      se_pos  = (se + abs(se))/2      # nx1 : current positive hinge forces
      seP_pos = (seP + abs(seP))/2    # nx1 : past    positive hinge forces
      se_neg  = se - se_pos           # nx1 : current negative hinge forces
      seP_neg = seP - seP_pos         # nx1 : past    negative hinge forces

      se_pn  = jnp.array([  se_pos,  se_neg ])   # nx2
      seP_pn = jnp.array([ seP_pos, seP_neg ])   # nx2

      ## CwcIF and Ccd coefficients
      CwcIF = jnp.zeros((n,2))        # initialize
      for k in range(n):

        # positive loading
        if (e[k] - eEx[k,pos] > gtol) and De[k]>0:
          CwcIF[k,pos] = 1.0
          if eP[k]<=eEx[k,pos]:
            De[k] = e[k] - eEx[k,pos] + Cwc[k,pos]*(eEx[k,pos] - eP[k])
          eEx[k,pos] = e[k];
        else:
          CwcIF[k,pos] = Cwc[k,pos];

        # negative loading
        if (e[k] - eEx[k,neg] < -gtol) and De[k]<0:
          CwcIF[k,neg] = 1.0;
          if eP[k]>=eEx[k,neg]:
            De[k] = e[k] - eEx[k,neg] + Cwc[k,neg]*(eEx[k,neg] - eP[k])
          eEx[k,neg] = e[k];
        else:
          CwcIF[k,neg] = Cwc[k,neg];
      ## energy dissipation
      # energy increment Dpsi with the following structure: +ve -ve 
      Dpsi = (se_pn + seP_pn) /2.*De;    # nx2

      # update +ve and -ve energy
      # pointwise multiplication (nx2).*(nx2).*(nx2)
      DPsi = CwcIF*Dpsi + CwcIF[:,[2,1]]*Ccd*Dpsi[:,[2,1]]
      # interaction between variables
      # effect of +ve and -ve on +ve variables
      DPsi[:,1] = DPsi[:,1] + CwcIF[:,1]*(Cin_pp@Dpsi[:,pos] + Cin_np@Dpsi[:,2])
      # effect of +ve and -ve on -ve variables
      DPsi[:,2] = DPsi[:,2] + CwcIF[:,2]*(Cin_pn@Dpsi[:,pos] + Cin_nn@Dpsi[:,2])

      psi = psiP + DPsi

      ## damage evolution
      # damage function
      g = psi - psiEx   # nx2

      # damage indices
      DdDpsi = jnp.zeros((n,2))
      for k in range(n):        # loop over variables to be damaged 
        for m in range(2):      # m=1: +ve, m=2: -ve
          if g[k,m] > gtol:
            psiEx[k,m] = psi[k,m]
            psi_tild   = (psi[k,m] - psi_d0[k,m])/(psi_d1[k,m] - psi_d0[k,m])
            # normalized damage rules
            d[k,m], DdDx = dFun(dp[k][:,m],psi_tild,Frac[k,m])
            DdDpsi[k,m]  = DdDx/(psi_d1[k,m]-psi_d0[k,m])
            # upper bound of damage for stability
            d[k,m] = min(d[k,m], 1-dtol);


      ## true forces
      dpos = 1 - d[:,1]               # nx1 : positive damage index
      dneg = 1 - d[:,2]               # nx1 : negative damage index
      s = dpos *se_pos + dneg *se_neg      # nx1

      ## Update current damage history variables
      state = dict(
                    d     = d,
                    se    = se,
                    eEx   = eEx,
                    psi   = psi,
                    psiEx = psiEx)

      return s, state
  return locals()


############################################################################################
## ------ function Check_DmgParam ----------------------------------------------------------
#def Check_DmgParam(DmgData)->DmgData:
#    """
#    check damage parameters and supply default values, if necessary
#
#    dFun = damage evolution function ( default = MBeta )
#    DOpt = 'total' (for total energy) or 'plastic' for plastic energy dissipation
#    dp   = damage rule parameters (nx2, first column for +ve, second for -ve;
#           n depends on the damage rule, for statistical functions n=2, default ones(2) )
#    Cd0  = scale factor of yield energy for damage threshold ( 1x2, default [ 1 1 ] ) 
#    Cd1  = scale factor of yield energy for complete damage  ( 1x2, default [ 100 100 ] )
#    Cwc  = influence factor for repeat cycles  (0=none to 1=full)( 1x2, default [ 0 0 ] )
#    Ccd  = damage coupling for opposite stress (0=none to 1=full)( 1x2, default [ 1 1 ] )
#    Frac = false or true for fracture inclusion ( 1x2 ) (default =false)
#    psiF = energy at fracture initiation        ( 1x2 ) (default = Cd1 )
#    psiU = energy at fracture completion        ( 1x2 ) (default = Cd1 )
#    gtol = tolerance value for damage growth check      (default = 1e-8)
#    dtol = tolerance value for damage index limit       (default = 1e-4 for dmax = 1-1e-4) 
#
#    function added                                                                    06-2021                                                       
#    """
#
#    n = DmgData.n;
#    if not hasattr(DmgData,'dFun'): DmgData.dFun = 'MBeta';
#    if not hasattr(DmgData,'DOpt'): DmgData.DOpt = 'total';
#    if not hasattr(DmgData,'Cd0') : DmgData.Cd0  = jnp.zeros((n,2))
#    if not hasattr(DmgData,'Cd1') : DmgData.Cd1  = 100.*jnp.ones((n,2));
#    if not hasattr(DmgData,'Cwc') : DmgData.Cwc  = jnp.zeros((n,2))
#    if not hasattr(DmgData,'Ccd') : DmgData.Ccd  = jnp.zeros((n,2))
#    if not hasattr(DmgData,'Frac'): DmgData.Frac = jnp.zeros((n,2))
#    if not hasattr(DmgData,'gtol'): DmgData.gtol = 1e-8;
#    if not hasattr(DmgData,'dtol'): DmgData.dtol = 1e-4;
#    if not hasattr(DmgData,'dp'):
#      for i = 1:n :  DmgData.dp{i} = ones(2);
#
#    ## adjust size of damage function parameters dp  
#    dp = DmgData.dp;
#    if not iscell(dp):  dp = repmat({dp},n,1);
#    for i in range(len(dp)):
#      dpi = dp[i];
#      if numel(dpi) == 2
#        if iscolumn(dpi): dp[i] = repmat(dpi ,1,2);
#        else            : dp[i] = repmat(dpi.T,1,2);
#
#    DmgData.dp = dp;
#
#    ## adjust size of damage evolution parameters Cd0, Cd1, Cwc, Ccd
#    Cd0 = DmgData.Cd0;
#    Cd1 = DmgData.Cd1;
#    Cwc = DmgData.Cwc;
#    Ccd = DmgData.Ccd;
#    DmgData.Cd0 = Adjust_DmgParam(n,Cd0);
#    DmgData.Cd1 = Adjust_DmgParam(n,Cd1);
#    DmgData.Cwc = Adjust_DmgParam(n,Cwc);
#    DmgData.Ccd = Adjust_DmgParam(n,Ccd);
#
#    ## adjust damage interaction parameters Cin
#    if not hasattr(DmgData,'Cin_pp'):
#      if hasattr(DmgData,'Cin')   : DmgData.Cin_pp = DmgData.Cin;
#      else                        : DmgData.Cin_pp = jnp.zeros((n))
#
#    if not hasattr(DmgData,'Cin_pn'):
#      if hasattr(DmgData,'Cin')   : DmgData.Cin_pn = DmgData.Cin;
#      else                        : DmgData.Cin_pn = jnp.zeros((n))
#
#    if not hasattr(DmgData,'Cin_np'):
#      if hasattr(DmgData,'Cin')   : DmgData.Cin_np = DmgData.Cin;
#      else                        : DmgData.Cin_np = jnp.zeros((n))
#
#    if not hasattr(DmgData,'Cin_nn'):
#      if hasattr(DmgData,'Cin')   : DmgData.Cin_nn = DmgData.Cin;
#      else                        : DmgData.Cin_nn = jnp.zeros((n))
#
#    Cin_pp = DmgData.Cin_pp;
#    Cin_pn = DmgData.Cin_pn;
#    Cin_np = DmgData.Cin_np;
#    Cin_nn = DmgData.Cin_nn;
#    if numel(Cin_pp) == 1:
#      Cin_pp = Cin_pp*(jnp.ones(n) - jnp.eye(n))
#    if numel(Cin_pn) == 1:
#      Cin_pn = Cin_pn*(jnp.ones(n) - jnp.eye(n))
#    if numel(Cin_np) == 1:
#      Cin_np = Cin_np*(jnp.ones(n) - jnp.eye(n))
#    if numel(Cin_nn) == 1:
#      Cin_nn = Cin_nn*(jnp.ones(n) - jnp.eye(n))
#
#    DmgData.Cin_pp = Cin_pp;
#    DmgData.Cin_pn = Cin_pn;
#    DmgData.Cin_np = Cin_np;
#    DmgData.Cin_nn = Cin_nn;
#
#    ## reorganize fracture energy field Frac
#    # convert fracture switch to logical
#    DmgData.Frac = bool(DmgData.Frac);
#    if not hasattr(DmgData,'psiF'): DmgData.psiF = DmgData.Cd1;
#    if not hasattr(DmgData,'psiU'): DmgData.psiU = DmgData.Cd1;
#    Frac = struct('Activ', num2cell(DmgData.Frac),\
#                  'psiF' , num2cell(DmgData.psiF),\
#                  'psiU' , num2cell(DmgData.psiU));
#    DmgData.Frac = Frac;
#
#   #end     # end function Check_DmgParam
#
#############################################################################################
### ------ function  Adjust_DmgParam --------------------------------------------------------
#def OutParam = Adjust_DmgParam (n,InParam):
#
#    if numel(InParam) == 1:
#      OutParam = InParam.*ones(n,2);
#    else:
#      if   isrow   (InParam) : OutParam = repmat(InParam ,n,1)
#      elif iscolumn(InParam) : OutParam = repmat(InParam, 1,2)
#      else:
#        # leave as is after checking size of specified array
#        [nr,nc] = size(InParam);
#        if nr==n and nc ==2:
#          OutParam = InParam;
#        else:
#          error (['size of Cd1 array needs to be nd x 2, ' \
#                  'where nd = number of force-deformation pairs for damage']);
#
#    # function Adjust_Param
#
