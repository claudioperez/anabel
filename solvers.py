import numpy as np 
import scipy.linalg
import pandas as pd

import ema.matrices as em


class PlasticAnalysis():
    pass 

class Event2Event():
    def __init__ (self, Model, Pref=None):
        if Pref is None:
            self.Pref = em.P_vector(Model) 
        else:
            self.Pref = Pref
        
        self.Model = Model
        self.init()

    def init(self):
        Model = self.Model
        self.k =  0
        self.Af = em.A_matrix(Model).f
        self.nf = Model.nf
        # set up element diagonal flexibility matrix, Fs
        self.Fs  = em.Fs_matrix(Model, Roption=False)
        # extract plastic capacity data from to set up Qpl array
        self.Qpl = em.Qpl_vector(Model)
        # initializations
        self.lamdah   = np.zeros(1)    # load factor
        self.nq       = len(Model.basic_forces)  # number of basic forces in model
        self.Qh  = np.zeros((self.nq,1))    # basic element forces
        self.Ufh = np.zeros((self.nf,1))    # displacements at free dofs
        self.Vph = np.zeros((self.nq,1))    # plastic deformations
        # initialize indices of basic forces, plastic hinge locations and non-trivial equilibrium equations
        self.ind_e = Model.idx_c        # indices of continuous/elastic basic forces
        self.ind_n = np.arange(self.nf)      # indices of non-trivial equilibrium equations     
        self.Iph   = np.array([])       # indices of plastic hinge locations
        # self.Af0 = self.Af[self.ind_e, :].del_zeros()
        self.Af0 = self.Af[self.ind_e, :]


        # form kinematic matrix A for all dofs of structural model and extract Af matrix from free dofs
    def next(self):
        """Increment to next event"""
        
        if np.linalg.matrix_rank(self.Af0)==len(self.ind_n):
            # set up stiffness matrix and solve for free dof displacements
    #         print(Af0)
            self.KsAf = (self.Fs[self.ind_e,:][:,self.ind_e]).inv @ self.Af0
            self.Kf   = self.Af0.T @ self.KsAf

            self.Ufe  = np.zeros((self.nf,1))
            self.Ufe[self.ind_n,0]  = self.Kf.inv @ self.Pref[self.ind_n]
            
            # determine basic element forces Qe (elastic increment)
            # self.Qe = np.zeros((self.nq,1))
            self.Qe = em.Q_vector(self.Model)
            self.Qe[self.ind_e,:] = self.KsAf @ self.Ufe[self.ind_n]
            
            # distinguish positive and negative basic force values
            self.ineg = np.where(self.Qe[self.ind_e,:]<0)[0]
            self.ne   = len(self.ind_e)
            self.ipos = np.where(self.Qe[self.ind_e,:]>=0.0)[0]

            self.ipe  = self.ind_e[self.ipos]
            self.ine  = self.ind_e[self.ineg]

            # set up residual plastic capacity Qplres for elastic locations
            self.Qplres = np.zeros((self.ne,1))
            self.Qplres[self.ipos,0] =  self.Qpl[self.ipe, 0] - self.Qh[self.ipe, self.k]    
            self.Qplres[self.ineg,0] = -self.Qpl[self.ine, 1] - self.Qh[self.ine, self.k] 
            
            # form demand capacity ratio DC for elastic basic forces
            self.DC        = np.zeros((self.nq,1))
            self.DC[self.ind_e] = self.Qe[self.ind_e]/self.Qplres

            
            # determine location of maximum DC
            self.new_hinge = np.where(self.DC == np.amax(self.DC))[0]
            
            # determine load factor increment for next hinge formation (use one hinge only)
            self.Dlamda     = 1/self.DC[self.new_hinge[0]]
            
            # update load factor
            self.lamdah = np.append(self.lamdah, self.lamdah[self.k] + self.Dlamda)

            # update global dof displacements and basic element forces with elastic increment
            self.Ufh = np.append(self.Ufh, np.array([self.Ufh[:,self.k] + self.Ufe@self.Dlamda]).T, axis=1)
            self.Qh = np.append(self.Qh, np.array([self.Qh[:,self.k] + self.Qe@self.Dlamda]).T, axis=1)


            # update list of plastic hinge locations (several hinges can form simultaneously)
            self.Iph   = np.append(self.Iph, self.new_hinge.T)

            # locate trivial equilibrium equations and set displacement equal to previous step
            self.ind_t = np.setdiff1d(list(range(self.nf)), self.ind_n)
            self.Ufh[self.ind_t, self.k+1] = self.Ufh[self.ind_t,self.k]

            # determine plastic deformations by Vpl = Af*Uf - Fs*Q
            self.Vph = np.append(self.Vph, np.array([self.Af@self.Ufh[:,self.k+1] - self.Fs@self.Qh[:,self.k+1]]).T, axis=1)

            # update list of elastic basic forces by removing new plastic hinge locations
            self.ind_e = np.setdiff1d(self.ind_e, self.new_hinge.T)

            # remove any columns of Af with all zero entries (trivial equilibrium equations)
            
            self.Af0 = self.Af[self.ind_e, :]
            # increment step counter
            self.k += 1
            
        return self.lamdah, self.Qh
        
        
    def run(self):  
        Pref = self.Pref
        Model = self.Model 
            # form kinematic matrix A for all dofs of structural model and extract Af matrix from free dofs
        Af = em.A_matrix(Model).f

        nf = Model.nf

        # set up element diagonal flexibility matrix, Fs
        Fs  = em.Fs_matrix(Model)

        # extract plastic capacity data from to set up Qpl array
        Qpl = em.Qpl_vector(Model)

        # initializations
        k        = 0                        # load step counter
        lamdah   = np.zeros(1)              # load factor
        nq       = len(Model.basic_forces)  # number of basic forces in model
        Qh  = np.zeros((nq,1))              # basic element forces
        Ufh = np.zeros((nf,1))              # displacements at free dofs
        Vph = np.zeros((nq,1))              # plastic deformations
        Acp = np.zeros((nq,1))              # plastic deformations
        Amp = np.zeros((nf,1))
        # initialize indices of basic forces, plastic hinge locations and non-trivial equilibrium equations
        ind_e = Model.idx_c        # indices of continuous/elastic basic forces
        ind_n = np.arange(nf)      # indices of non-trivial equilibrium equations     
        Iph   = np.array([])       # indices of plastic hinge locations
        
        # Loop until the structure is unstable
        # Af0 = Af[ind_e, :].del_zeros()
        Af0 = Af[ind_e, :]

        while np.linalg.matrix_rank(Af0)==len(ind_n):
            # set up stiffness matrix and solve for free dof displacements

            KsAf = (Fs[ind_e,:][:,ind_e]).inv @ Af0
            Kf   = Af0.T @ KsAf

            Ufe  = np.zeros((nf,1))
            Ufe[ind_n,0]  = Kf.inv @ Pref[ind_n]
            
            # determine basic element forces Qe (elastic increment)
            # Qe = np.zeros((nq,1))
            Qe = em.Q_vector(Model)
            Qe[ind_e,:] = KsAf @ Ufe[ind_n]
            
            # distinguish positive and negative basic force values
            ineg = np.where(Qe[ind_e,:]<0)[0]
            ne   = len(ind_e)
            ipos = np.where(Qe[ind_e,:]>=0.0)[0]

            ipe  = ind_e[ipos]
            ine  = ind_e[ineg]

            # set up residual plastic capacity Qplres for elastic locations
            Qplres = np.zeros((ne,1))
            Qplres[ipos,0] =  Qpl[ipe, 0] - Qh[ipe, k]    
            Qplres[ineg,0] = -Qpl[ine, 1] - Qh[ine, k] 
            
            # form demand capacity ratio DC for elastic basic forces
            DC        = np.zeros((nq,1))
            DC[ind_e] = Qe[ind_e]/Qplres

            
            # determine location of maximum DC
            new_hinge = np.where(DC == np.amax(DC))[0]
            
            # determine load factor increment for next hinge formation (use one hinge only)
            Dlamda     = 1/DC[new_hinge[0]]
            
            # update load factor
            lamdah = np.append(lamdah, lamdah[k] + Dlamda)

            # update global dof displacements and basic element forces with elastic increment
            Ufh = np.append(Ufh, np.array([Ufh[:,k] + Ufe@Dlamda]).T, axis=1)
            Qh = np.append(Qh, np.array([Qh[:,k] + Qe@Dlamda]).T, axis=1)


            # update list of plastic hinge locations (several hinges can form simultaneously)
            Iph   = np.append(Iph, new_hinge.T)

            # locate trivial equilibrium equations and set displacement equal to previous step
            ind_t = np.setdiff1d(list(range(nf)), ind_n)
            Ufh[ind_t, k+1] = Ufh[ind_t,k]

            # determine plastic deformations by Vpl = Af*Uf - Fs*Q
            Vph = np.append(Vph, np.array([Af@Ufh[:,k+1] - Fs@Qh[:,k+1]]).T, axis=1)
            
            # Ampi = scipy.linalg.null_space(Af[ind_e, :])
            # Amp = np.append(Amp, Ampi, axis=1)
            # Acp = np.append(Acp,  Af @ Ampi, axis=0)
            

            # update list of elastic basic forces by removing new plastic hinge locations
            ind_e = np.setdiff1d(ind_e, new_hinge.T)

            # remove any columns of Af with all zero entries (trivial equilibrium equations)
            
            Af0 = Af[ind_e, :]
            # increment step counter
            k += 1
        
        output = (lamdah, Qh, Ufh, Vph, Iph)
        self.lamda = lamdah 
        self.Q  = [em.Q_vector(self.Model, Qh[:,K].T) for K in range(k+1)]
        # self.U = [em.U_vector(self.Model, Ufh[:,K].T) for K in range(k+1)]  
        self.U = pd.DataFrame(Ufh)
        self.V = [em.V_vector(self.Model, Vph[:,K].T) for K in range(k+1)]  
        self.Iph = Iph
        self.Acp = Acp 
        self.Amp = Amp 
        return output[0:2]