"""
# Matrix API

This module provides functions and classes for constructing various structural analysis matrices. 

"""

import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib.pyplot as plt

try:
    import emme
except:
    import anabel as emme
# from emme.utilities import Structural_Matrix, Structural_Vector, transfer_vars


settings = {
    "DATAFRAME_LATEX": True,
}

subscripts = {
    'part': 'subscript',
     None: '',
    'initial': '0',
    'continuous':'c',
    'primary':'i',
    'reactions':'d',
    'elem-load':'w',
    'free': 'f',
}

def del_zeros(mat):
    delrows = np.where(~mat.any(axis=1))[0]
    delcols = np.where(~mat.any(axis=0))[0]

    newM = np.delete(mat, delcols, axis=1)
    newM = np.delete(newM, delrows, axis=0)
    return newM

def elem_dofs(Elem):
    dofs = []
    for node in Elem.nodes:
        dofs.extend(node.dofs)
    return dofs

def transfer_vars(item1, item2):
    for key, value in item1.__dict__.items():
        item2.__dict__[key] = value

def Localize(U_vector, P_vector, model=None):
    if model is None: model = U_vector.model
    A =  A_matrix(model)
    Q0 = Q0_vector(model)
    Ks = Ks_matrix(model)

    V = A.f @ U_vector.f
    Q  = Ks@V + Q0
    return V, Q



class Structural_Vector(np.ndarray):
    column_data = ["vector"]
    row_data = None
    subs = [None]
    tag = 'Vector'
    def __new__(cls, mat):
        mat = mat
        return np.asarray(mat).view(cls)

    def __init__(self, mat):
        self.tag = None

    def _repr_html_(self):
        try:
            out = self.df.to_html()
            return out
        except: 
            return self

    def __add__(self, other):
        if isinstance(other, type(self)):
            out = np.add(self, other).view(type(self))
            transfer_vars(self, out)
        else:
            out = super().__add__(other)
        return out
    
    def get(self, key):
        idx = np.array([i for i,j in enumerate(self.row_data) if str(j) == key], dtype=int)
        # row = self.row_data.index(component)
        return self[idx]

    def set_item(self, key, value):
        idx = np.array([i for i,j in enumerate(self.row_data) if str(j) == key], dtype=int)
        self[idx] = value

    def rows(self, component):
        idx = np.where(np.isin(self.row_data,component))[0]
        newV = self[idx]
        newV.row_data = np.array(self.row_data)[idx]
        newV.model = self.model
        return newV

    @property
    def df(self):
        row_data = ['$'+str(tag)+'$' for tag in self.row_data]
        header = '$'+self.tag+'_{{'
        try:
            for sub in self.subs:
                header += subscripts[sub]
        except: pass
        print
        header += '}}$'

        return pd.DataFrame(np.around(self,14), index=row_data, columns=[header])

    @property
    def symb(self):
        var = []
        for eid in self.row_data:
            var.append(sp.symbols(self.tag+eid))
        return sp.Matrix(var)
    
    @property
    def disp(self):
        return sp.Matrix(self)
     
class Structural_Matrix(np.ndarray):
    column_data = None
    row_data = None
    c_ridx = None # row indexes for use in .c method
    c_cidx = None # column indexes for use in .c method
    tag = None

    def __new__(cls, mat):
        mat = mat
        return np.asarray(mat).view(cls)

    def __init__(self, mat):
        self.tag = None

    def __matmul__(self, other):
        if isinstance(other, Structural_Matrix):
            out = np.matmul(self,other).view(Structural_Matrix)
            transfer_vars(self,out)
            out.column_data = other.column_data
        
        elif isinstance(other, Structural_Vector):
            out = np.matmul(self,other).view(Structural_Vector)
            out.row_data = self.row_data
            # out.column_data = ['.']
        else:
            out = np.matmul(self,other).view(Structural_Vector)
            out.row_data = self.row_data
            # out.column_data = ['.']
        return out

    def __add__(self, other):
        out = np.add(self,other)
        if (isinstance(other, Structural_Matrix)
            or isinstance(other, Structural_Vector)):
            out.row_data = self.row_data
        return out

    
    def __truediv__(self, other):
        out = np.ndarray.__truediv__(self, other) 
        if (isinstance(other, float) 
            or isinstance(other, Structural_Matrix)
            or isinstance(other, Structural_Vector)):
            out = np.ndarray.__truediv__(self, other).view(Structural_Matrix)
            transfer_vars(self, out)
        else:
            out = np.ndarray.__truediv__(self, other)
        return out
        
    def _repr_html_(self):
        try: 
            df = self.df
            return df.to_html()
        except:
            try:
                return pd.DataFrame(self).to_html()
            except:
                pass

    @property
    def disp(self):
        return sp.Matrix(self)
    
    @property
    def df(self):
        if settings['DATAFRAME_LATEX']:
            row_data = ['$'+str(tag)+'$' for tag in self.row_data]
            column_data = ['$'+str(tag)+'$' for tag in self.column_data]
        else:
            row_data = [str(i) for i in self.row_data]
            column_data = [str(i) for i in self.column_data]
        return pd.DataFrame(np.around(self,5), index=row_data, columns=column_data)

    @property
    def inv(self):
        mat = np.linalg.inv(self)
        transfer_vars(self, mat)
        mat.row_data = self.column_data
        mat.column_data = self.row_data
        return mat
    
    @property
    def rank(self):
        """Return the rank of a matrix"""
        return np.linalg.matrix_rank(self)
    
    @property
    def lns(self):
        """Return a basis for the left nullspace of a matrix."""
        return scipy.linalg.null_space(self.T)
    
    @property
    def nls(self):
        """return a basis for the nullspace of matrix."""
        return scipy.linalg.null_space(self)
    
    @property
    def ker(self):
        "Return a basis for the kernel (nullspace) of a matrix."
        kernel = scipy.linalg.null_space(self) 
        ker = Structural_Matrix(kernel)
        transfer_vars(self,ker)
        ker.row_data = self.column_data 
        ker.column_data = [str(i+1) for i in range(len(ker[0]))]
        return ker

    def lu(self):
        
        return scipy.linalg.lu(self)

    @property
    def c(self):
        delcols = self.c_cidx
        delrows = self.c_ridx

        newM = np.delete(self, delrows, axis=0).view(type(self))
        if delcols: newM = np.delete(newM, delcols, axis=1).view(type(self))
        
        transfer_vars(self, newM)

        if delcols: newM.column_data = list(np.delete(self.column_data, delcols))

        newM.row_data = list(np.delete(self.row_data, delrows))
        return newM

    def round(self, num):
        newM = np.around(self, num).view(Structural_Matrix)
        transfer_vars(self,newM)
        return newM

    def remove(self, component):
        """Remove items by looking up column_data/row_data"""
        if type(component) is list:
            for item in component:
                if item in self.column_data: 
                    delcol = self.column_data.index(item)
                    newM = np.delete(self, delcol, axis=1).view(type(self))
                    transfer_vars(self,newM)
                    newM.column_data = list(np.delete(self.column_data, delcol))
                else:
                    delrow = self.row_data.index(item)
                    newM = np.delete(self, delrow, axis=0).view(type(self))
                    transfer_vars(self, newM)
                    # newM.column_data = self.column_data
                    # newM.model = self.model
                    newM.row_data = list(np.delete(self.row_data, delrow))

        else:
            item = component
            if item in self.column_data: 
                delcol = self.column_data.index(item)
                newM = np.delete(self, delcol, axis=1).view(type(self))
                newM.row_data = self.row_data
                newM.model = self.model
                newM.column_data = list(np.delete(self.column_data, delcol))
                # try: newM.rel = self.rel
                # except: pass
            else:
                delrow = self.row_data.index(item)
                newM = np.delete(self, delrow, axis=0).view(type(self))
                newM.column_data = self.column_data
                newM.model = self.model
                newM.row_data = list(np.delete(self.row_data, delrow))
                # try: newM.rel = self.rel
                # except: pass
        return newM

    def get(self, row_name, col_name):
        idxr = np.where(self.row_data == row_name)
        idxc = np.where(self.column_data == col_name)
        return self[idxr, idxc]


    def rows(self, component):
        rows = [self.row_data.index(item) for item in component]
        newM = self[rows,:]
        newM.model = self.model
        newM.column_data = self.column_data
        newM.row_data = list(np.array(self.row_data)[rows])
        return newM

    def del_zeros(self):
        """Delete rows and columns of a matrix with all zeros"""
        delrows = np.where(~self.any(axis=1))[0]
        delcols = np.where(~self.any(axis=0))[0]

        newM = np.delete(self, delcols, axis=1).view(type(self))
        newM = np.delete(newM, delrows, axis=0).view(type(self))
        transfer_vars(self, newM)

        newM.column_data = list(np.delete(self.column_data, delcols))
        newM.row_data = list(np.delete(self.row_data, delrows))
        return newM
        
    def add_cols(self, component):
        if "colinear" in component:
            vertical = [elem for elem in self.model.elems if elem.Dx==0.0]
            other = set({})
            for elem in self.model.elems:
                try: other.add(elem.Dy/elem.Dx)
                except ZeroDivisionError: pass
            return vertical, other
        if type(component) is list:
            #ASSUMES component IS A LIST OF COLUMN INDICES
            delcols = [self.column_data.index(item) for item in component[1:len(component)]]
            i0 = self.column_data.index(component[0])
            newM = np.delete(self, delcols, axis=1).view(type(self))
            for col in delcols:
                newM[:,i0] += self[:,col]
            newM.row_data = self.row_data
            newM.model = self.model
            newM.column_data = list(np.delete(self.column_data, delcols))
            # try: newM.rel = self.rel
            # except: pass
            return newM

    def add_rows(self, component):
        if type(component) is list:
            delrows = [self.row_data.index(item) for item in component[1:len(component)]]
            i0 = self.row_data.index(component[0])
            newA = np.delete(self, delrows, axis=0).view(type(self))
            for row in delrows:
                newA[i0,:] += self[row,:]
            newA.column_data = self.column_data
            newA.model = self.model
            newA.row_data = list(np.delete(self.row_data, delrows))
            return newA

class row_vector (Structural_Vector):
    def __new__(cls, Matrix):
        V = np.zeros((len(Matrix.row_data)))
        return np.asarray(V).view(cls)

    def __init__(self, Matrix):
        self.tag = "Y"
        self.matrix = Matrix 
        self.row_data = Matrix.row_data
    
class column_vector (Structural_Vector):

    def __new__(cls, Matrix, Vector=None):
        V = np.zeros((len(Matrix.column_data)))
        return np.asarray(V).view(cls)

    def __init__(self, Matrix, Vector=None):
        self.tag = "X"
        self.matrix = Matrix 
        self.row_data = Matrix.column_data
        if Vector is not None:
            for key in Vector.row_data:
                self.set_item(key, Vector.rows([key]))


class Static_matrix (Structural_Matrix):
    """B_MATRIX static matrix of structural model with 2d/3d truss and 2d frame elements
    the function forms the static matrix B for all degrees of freedom and
    all basic forces of the structural model specified in data structure MODEL;
    the function is currently limited to 2d/3d truss and 2d frame elements
    
    Parameters
    ---------------

    model: emme.Model object
    
    Partitions
    =========================================================================================
 
    - B.f  : nf x ntq

    - B.c  : nf x nq

    - B.fc : nf x nq

    - B.i  : ni x nq

    - B.x  : nx x nq

    where:

    - ni: number of primary (non-redundant) forces.
    - nq: number of total, continuous forces.
    - nx: number of redundant forces.

    """

    ranges = {
        'f': 'free dof rows',
        'i': 'primary force columns',
        'x': 'redundant force columns',
        'c': 'continuous force columns (removes releases)',
        'd': 'reaction force columns',
    }

    def __new__(cls, model, matrix=None, rng=None):
        fullnq = sum([len(elem.rel) for elem in model.elems])
        B = np.zeros((model.nt,fullnq))
        ci = 0
        for elem in model.elems:
            dofs = elem.dofs
            bg = elem.bg_matrix()
            nq = np.size(bg,1)
            for j,dof in enumerate(dofs):
                B[int(dof)-1,ci:ci+nq] = bg[j,:]
            ci = ci+nq
        input_array = B
        return np.asarray(input_array).view(cls)

    def __init__(self, model, matrix=None, rng=None):
        if rng is None:
            self.rng = None
        self.model = model
        self.row_data = np.array([str(dof) for dof in range(1, model.nt+1)])
        self.column_data = np.array([elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()])
        if matrix is not None:
            fullnq = sum([len(elem.rel) for elem in model.elems])
            self[:,:] = np.zeros((model.nt,fullnq))
            for idxr, rw in enumerate(self.row_data):
                if rw in matrix.row_data:
                    for idxc, cl in enumerate(self.column_data):
                        if cl in matrix.column_data:
                            self[idxr,idxc] = matrix.get(rw,cl)

    def __matmul__(self, Vector):
        if type(Vector) is nForce_vector:
            vect = np.matmul(self, Vector).view(iForce_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        elif type(Vector) is iForce_vector:
            vect = np.matmul(self, Vector).view(nForce_vector) 
            vect.row_data = self.row_data
            vect.matrix = self
        else:
            vect = Structural_Matrix.__matmul__(self, Vector)
        return vect
  #<# 
    @property
    def f(self):
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) > self.model.nf]
        
        newB = np.delete(self, delrows, axis=0).view(type(self))
        transfer_vars(self, newB)
        # newB.model = self.model
        newB.row_data = list(np.delete(self.row_data, delrows))
        # newB.column_data = self.column_data
        return newB

    @property
    def i(self):
        """Removes rows of B_matrix corresponding to primary (non-redundant) forces"""
        Bf = self.f
        idx_i = self.model.idx_i
        newB = Bf[:,idx_i]
        # newB = Bf[idx_i,:]
        transfer_vars(Bf, newB)
        newB.column_data = Bf.column_data[idx_i]
        return newB
    
    @property
    def d(self):
        """Removes rows corresponding to free dofs"""
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) <= self.model.nf]
        newB = np.delete(self, delrows, axis=0).view(type(self))
        transfer_vars(self,newB)
        newB.row_data = list(np.delete(self.row_data, delrows))
        return newB

    @property
    def c(self):

        Bf = self.f
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        delcols = [Bf.column_data.index(tag) for tag in tags]
        newB = np.delete(Bf, delcols, axis=1).view(type(self))
        transfer_vars(Bf, newB)
        newB.column_data = list(np.delete(Bf.column_data, delcols))
        return newB

    @property
    def o(self):
        """Remove columns corresponding to element force releases, then delete zeros"""
        Bf = self.f
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        # delcols = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        delcols = [Bf.column_data.index(tag) for tag in tags]
        newB = np.delete(Bf, delcols, axis=1).view(type(self))
        transfer_vars(Bf, newB)
        newB.column_data = list(np.delete(Bf.column_data, delcols))
        newB = newB.del_zeros()
        return newB

    @property
    def fc(self):
        return self.f.c

    @property
    def x(self):
        """Removes rows of B_matrix corresponding to primary (non-redundant) forces"""
        idx_x = self.model.idx_x
        newB = self[:,idx_x]
        transfer_vars(self, newB)
        newB.column_data = self.column_data[idx_x]
        return newB

    @property
    def barxi(self):
        Bx = self.f.x
        Bbarxi = self.bari @ -Bx
        Bbarxi.column_data = Bx.column_data
        return Bbarxi

    @property
    def barx(self):
        nQ = len(self.column_data)
        nx = len(self.model.redundants)

        Bbarxi = self.barxi

        Bbarx = Structural_Matrix(np.zeros((nQ,nx)))
        transfer_vars(self, Bbarx)
        Bbarx.column_data = Bbarxi.column_data
        Bbarx.row_data = self.column_data

        for idxc, cl in enumerate(Bbarx.column_data):
            for idxr, rw in enumerate(Bbarx.row_data):
                if rw in Bbarxi.row_data:
                    Bbarx[idxr,idxc] = Bbarxi.get(rw,cl)
                elif cl==rw:
                        Bbarx[idxr, idxc] = 1.

        return Bbarx

    @property
    def bari(self):
        return self.i.del_zeros().inv

    @property
    def c0(self):
        Bf = self.f
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        delcols = [Bf.column_data.index(tag) for tag in tags]
        newB = Bf 
        for col in delcols:
            newB[:,col] = [0.]*len(Bf[:,0])
        transfer_vars(Bf, newB)
        newB.column_data = list(np.delete(Bf.column_data, delcols))
        return newB

class nStatic_matrix(Structural_Matrix):
    ranges = {
        'f': 'free dof rows',
        'i': 'primary force columns',
        'x': 'redundant force columns',
        'c': 'continuous force columns (removes releases)',
        'd': 'reaction force columns',}
    def __new__(cls, arry, model, rcdata):
        return np.asarray(arry).view(cls)

    def __init__(self, arry, model, rcdata):
        self.model = model
        self.row_data = rcdata[0]
        self.column_data = rcdata[1]
        
    def __matmul__(self, Vector):
        if type(Vector) is nForce_vector:
            vect = np.matmul(self, Vector).view(iForce_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        elif isinstance(Vector, iForce_vector):
            vect = np.matmul(self, Vector).view(nForce_vector) 
            vect.row_data = self.row_data
            vect.matrix = self
        else:
            vect = Structural_Matrix.__matmul__(self, Vector)
        return vect

    @property
    def f(self):
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) > self.model.nf]
        
        newB = np.delete(self, delrows, axis=0).view(type(self))
        transfer_vars(self, newB)
        newB.row_data = np.delete(self.row_data, delrows)
        return newB
        
    @property
    def i(self):
        """Removes rows of B_matrix corresponding to redundant forces"""
        # reducedB = self.f.c.del_zeros()
        reducedB = self.f.c
        rdts = reducedB.model.redundants
        tags = [q.elem.tag + '_'+str(q.nature) for q in rdts]
        delcols = [reducedB.column_data.index(tag) for tag in tags]
        newB = np.delete(reducedB, delcols, axis=1).view(Static_matrix)
        transfer_vars(reducedB, newB)
        newB.column_data = list(np.delete(reducedB.column_data, delcols))
        return newB
    
    @property
    def d(self):
        """Removes rows corresponding to free dofs"""
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) <= self.model.nf]
        newB = np.delete(self, delrows, axis=0).view(type(self))
        transfer_vars(self,newB)
        newB.row_data = list(np.delete(self.row_data, delrows))
        return newB

    @property
    def c(self):
        """Removes columns corresponding to element hinges/releases"""
        Af = self.f
        idx_c = self.model.idx_c
        newA = Af[:,idx_c]
        transfer_vars(Af, newA)
        newA.column_data = Af.column_data[idx_c]
        return newA

    @property
    def o(self):
        """Remove columns corresponding to element force releases, then delete zeros"""
        Bf = self.f
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        # delcols = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        delcols = [Bf.column_data.index(tag) for tag in tags]
        newB = np.delete(Bf, delcols, axis=1).view(type(self))
        transfer_vars(Bf, newB)
        newB.column_data = list(np.delete(Bf.column_data, delcols))
        newB = newB.del_zeros()
        return newB

    @property
    def fc(self):
        return self.f.c
    
    @property
    def x(self):
        """Removes rows of B_matrix corresponding to primary (non-redundant) forces
        
        """
        rdts = self.model.redundants
        tags = [q.elem.tag + '_'+str(q.nature) for q in rdts]
        cols = [self.column_data.index(tag) for tag in tags]
        newB = self[:,cols]
        transfer_vars(self, newB)
        newB.column_data = np.array([self.column_data[col] for col in cols])
        # newB.row_data = self.row_data
        # newB.model = self.model
        return newB

    @property
    def barxi(self):
        Bx = self.f.x
        Bbarxi = self.bari @ -Bx
        Bbarxi.column_data = Bx.column_data
        return Bbarxi

    @property
    def barx(self):
        nQ = len(self.column_data)
        nx = len(self.model.redundants)

        Bbarxi = self.barxi

        Bbarx = Structural_Matrix(np.zeros((nQ,nx)))
        transfer_vars(self, Bbarx)
        Bbarx.column_data = Bbarxi.column_data
        Bbarx.row_data = self.column_data
        for idxc, cl in enumerate(Bbarx.column_data):
            for idxr, rw in enumerate(Bbarx.row_data):
                if rw in Bbarxi.row_data:
                    Bbarx[idxr,idxc] = Bbarxi.get(rw,cl)
                elif cl==rw:
                        Bbarx[idxr, idxc] = 1.

        return Bbarx

    @property
    def bari(self):
        return self.i.del_zeros().inv

    @property
    def c0(self):
        Bf = self.f
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        # delcols = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        delcols = [Bf.column_data.index(tag) for tag in tags]
        newB = Bf
        for col in delcols:
            newB[:,col] = [0.]*len(Bf[:,0])
        transfer_vars(Bf, newB)
        newB.column_data = list(np.delete(Bf.column_data, delcols))
        # newB.row_data = self.row_data
        # newB.rel = self.rel
        return newB

def B_matrix(model, matrix=None, rng=None):
    """Returns a Static_matrix object"""
    return Static_matrix(model, matrix, rng)

def nB_matrix(model, rng=None):
    """Returns a Static_matrix object"""
    fullnq = sum([len(elem.rel) for elem in model.elems])
    B = np.zeros((model.nt, fullnq))
    ci = 0
    for elem in model.elems:
        eid = elem.dofs
        bg = elem.bg_matrix()
        nq = np.size(bg,1)
        for j,eidi in enumerate(eid):
            B[int(eidi)-1,ci:ci+nq] = bg[j,:]
        ci = ci+nq
    input_array = B
    return np.asarray(input_array).view(nStatic_matrix)
    # matrix =  np.asarray(input_array)
    # return nStatic_matrix(model, matrix, rng)

def Bh_matrix(model):
    """Returns a Static_matrix object"""
    fullnq = sum([len(elem.rel) for elem in model.elems])
    B = np.zeros((model.nt, fullnq))
    ci = 0
    for elem in model.elems:
        eid = elem.dofs
        bg = elem.bg_matrix(Roption=True)
        nq = np.size(bg,1)
        for j,eidi in enumerate(eid):
            B[int(eidi)-1,ci:ci+nq] = bg[j,:]
        ci = ci+nq
    row_data = np.array([str(dof) for dof in range(1, model.nt+1)])
    column_data = np.array([elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()])
    rcdata = (row_data, column_data)
    return nStatic_matrix(B, model, rcdata)

class Kinematic_matrix (Structural_Matrix):
    """Class for the kinematic matrix of a structural model with 2d/3d truss and 2d frame elements
    the function forms the kinematic matrix A for all degrees of freedom and
    all element deformations of the structural model specified in data structure MODEL
    the function is currently limited to 2d/3d truss and 2d frame elements
    
    Returns
    ---------

    Kinematic matrix

    """

    ranges = {
        'f': 'free dof columns',
        'i': 'primary force/deformation rows',
        'x': 'redundant force/deformation rows',
        'd': 'reaction force/deformation rows',
        'c': 'continuous (hingeless) force/deformation rows'
    }

    def __new__(cls, model, matrix=None,rng=None):
        A  = np.zeros((sum(model.nv),model.nt))
        ri = 0
        for elem in model.elems:
            eid = elem.dofs
            ag = elem.ag()
            nv = len(elem.v)
            for j, eidi in enumerate(eid):
                A[ri:ri+nv, int(eidi)-1] = ag[:,j]
            ri = ri+nv
        input_array = A
        return np.asarray(input_array).view(cls)

    def __init__(self, model, matrix=None,rng=None):
        if rng is None:
            self.rng = None
        self.model = model
        self.column_data = np.array([str(dof) for dof in range(1,model.nt+1)])
        self.row_data = np.array([elem.tag+'_'+key for elem in model.elems for key in elem.v.keys()])
        self.basic_deformations = np.array([v for elem in model.elems for v in elem.basic_deformations])
        
        self.idx_h = []
        if matrix is not None:
            fullnq = sum([len(elem.rel) for elem in model.elems])
            self[:,:] = np.zeros((model.nt,fullnq))
            for idxr, rw in enumerate(self.row_data):
                if rw in matrix.row_data:
                    for idxc, cl in enumerate(self.column_data):
                        if cl in matrix.column_data:
                            self[idxr,idxc] = matrix.get(rw,cl)

    def __matmul__(self, Vector):
        if isinstance(Vector, Deformation_vector):
            # print('a')
            vect = np.matmul(self, Vector).view(Displacement_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        elif isinstance(Vector, Displacement_vector):
            vect = np.matmul(self, Vector).view(Deformation_vector) 
            vect.row_data = self.row_data
            vect.matrix = self
        else:
            vect = Structural_Matrix.__matmul__(self, Vector)

        return vect


    def combine(self, component):
        if "colinear" in component:
            vertical = [elem for elem in self.model.elems if elem.Dx==0.0]
            other = set({})
            for elem in self.model.elems:
                try: other.add(elem.Dy/elem.Dx)
                except ZeroDivisionError: pass
            return vertical, other

        if type(component) is list:
            ## TO BE DEPRECATED
            #ASSUMES component IS A LIST OF COLUMN INDICES
            delcols = [self.column_data.index(item) for item in component[1:len(component)]]
            i0 = self.column_data.index(component[0])
            newA = np.delete(self, delcols, axis=1).view(Kinematic_matrix)
            for col in delcols:
                newA[:,i0] += self[:,col]
            newA.row_data = self.row_data
            newA.model = self.model
            newA.column_data = list(np.delete(self.column_data, delcols))
            # newA.rel = self.rel
            return newA
    
    @property
    def f(self):
        """Removes columns corresponding to fixed dofs"""
        delcols = [idx for idx, dof in enumerate(self.column_data) if int(dof) > self.model.nf]
        newA = np.delete(self, delcols, axis=1).view(Kinematic_matrix)
        transfer_vars(self, newA)
        newA.column_data = list(np.delete(self.column_data, delcols))
        return newA

    @property
    def i(self):
        """Removes rows corresponding to redundant forces"""
        Afc = self.f.c
        rdts = self.model.redundants
        tags = [q.elem.tag +'_'+ str(q.nature) for q in rdts]
        delrows = [Afc.row_data.index(tag) for tag in tags]
        newA = np.delete(Afc, delrows, axis=0).view(Kinematic_matrix)
        transfer_vars(Afc,newA)
        newA.row_data = list(np.delete(Afc.row_data, delrows))
        return newA

    @property
    def d(self):
        """Removes columns corresponding to free dofs"""
        delcols = [idx for idx, dof in enumerate(self.column_data) if int(dof) <= self.model.nf]
        newA = np.delete(self, delcols, axis=1).view(Kinematic_matrix)
        transfer_vars(self,newA)
        newA.column_data = list(np.delete(self.column_data, delcols))
        return newA


    @property
    def c(self):
        """Removes rows corresponding to element hinges/releases"""
        Af = self.f
        idx_c = self.model.idx_c
        newA = Af[idx_c,:]
        transfer_vars(Af, newA)
        newA.row_data = Af.row_data[idx_c]
        return newA
    
    @property
    def c0(self):
        Af = self.f
        n_col = len(Af.T)
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        # delcols = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        delrows = np.where(np.isin(Af.row_data,tags))
        # delrows = [Af.row_data.index(tag) for tag in tags]
        newA = Af 
        for rw in delrows:
            newA[rw,:] = [0.]*n_col
        transfer_vars(Af, newA)
        # newA.row_data = list(np.delete(Af.row_data, delrows))
        return newA
    
    @property
    def o(self):
        Af = self.f
        n_col = len(Af.T)
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        # delcols = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        delrows = [Af.row_data.index(tag) for tag in tags]
        newA = Af 
        for rw in delrows:
            newA[rw,:] = [0.]*n_col
        newA = newA.del_zeros()
        transfer_vars(Af, newA)
        # newA.row_data = list(np.delete(Af.row_data, delrows))
        return newA
    
    @property
    def e(self):
        Af = self.f
        n_col = len(Af.T)
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        # delcols = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        delrows = [Af.row_data.index(tag) for tag in tags]
        newA = Af 
        for rw in delrows:
            newA[rw,:] = [0.]*n_col
        newA = newA.del_zeros()
        transfer_vars(Af, newA)
        # newA.row_data = list(np.delete(Af.row_data, delrows))
        return newA


class nKinematic_matrix (Structural_Matrix):
    """Class for the kinematic matrix of a structural model with 2d/3d truss and 2d frame elements
    the function forms the kinematic matrix A for all degrees of freedom and
    all element deformations of the structural model specified in data structure MODEL 
    the function is currently limited to 2d/3d truss and 2d frame elements
    
    Returns
    ---------
    Kinematic matrix

    """

    ranges = {
        'f': 'free dof columns',
        'i': 'primary force/deformation rows',
        'x': 'redundant force/deformation rows',
        'd': 'reaction force/deformation rows',
        'c': 'continuous (hingeless) force/deformation rows'
    }

    def __new__(cls, model, matrix=None,rng=None):
        A  = np.zeros((sum(model.nv),model.nt))
        ri = 0
        for elem in model.elems:
            eid = elem.dofs
            ag = elem.ag()
            nv = len(elem.v)
            for j, eidi in enumerate(eid):
                A[ri:ri+nv, int(eidi)-1] = ag[:,j]
            ri = ri+nv
        input_array = A
        return np.asarray(input_array).view(cls)

    def __new__(cls, arry, model, rcdata):
        return np.asarray(arry).view(cls)

    def __init__(self, arry, model, rcdata):
        self.model = model
        self.row_data = rcdata[0]
        self.column_data = rcdata[1]
        self.basic_deformations = np.array([v for elem in model.elems for v in elem.basic_deformations])
        
        self.idx_h = []


    def __matmul__(self, Vector):
        if isinstance(Vector, Deformation_vector):
            # print('a')
            vect = np.matmul(self, Vector).view(Displacement_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        elif isinstance(Vector, Displacement_vector):
            # print('b')
            vect = np.matmul(self, Vector).view(Deformation_vector)
            # vect = np.matmul(self, Vector).view(Structural_Vector)  
            vect.row_data = self.row_data
            vect.matrix = self
        else:
            # print('else')
            vect = Structural_Matrix.__matmul__(self, Vector)
        return vect

    def combine(self, component):
        if "colinear" in component:
            vertical = [elem for elem in self.model.elems if elem.Dx==0.0]
            other = set({})
            for elem in self.model.elems:
                try: other.add(elem.Dy/elem.Dx)
                except ZeroDivisionError: pass
            return vertical, other

        if type(component) is list:
            ## TO BE DEPRECATED
            #ASSUMES component IS A LIST OF COLUMN INDICES
            delcols = [self.column_data.index(item) for item in component[1:len(component)]]
            i0 = self.column_data.index(component[0])
            newA = np.delete(self, delcols, axis=1).view(Kinematic_matrix)
            for col in delcols:
                newA[:,i0] += self[:,col]
            newA.row_data = self.row_data
            newA.model = self.model
            newA.column_data = list(np.delete(self.column_data, delcols))
            # newA.rel = self.rel
            return newA
    
    @property
    def f(self):
        """Removes columns corresponding to fixed dofs"""
        delcols = [idx for idx, dof in enumerate(self.column_data) if int(dof) > self.model.nf]
        newA = np.delete(self, delcols, axis=1).view(Kinematic_matrix)
        transfer_vars(self, newA)
        newA.column_data = list(np.delete(self.column_data, delcols))
        return newA

    @property
    def i(self):
        """Removes rows corresponding to redundant forces"""
        Afc = self.f.c
        rdts = self.model.redundants
        tags = [q.elem.tag +'_'+ str(q.nature) for q in rdts]
        delrows = [Afc.row_data.index(tag) for tag in tags]
        newA = np.delete(Afc, delrows, axis=0).view(Kinematic_matrix)
        transfer_vars(Afc,newA)
        newA.row_data = list(np.delete(Afc.row_data, delrows))
        return newA

    @property
    def d(self):
        """Removes columns corresponding to free dofs"""
        delcols = [idx for idx, dof in enumerate(self.column_data) if int(dof) <= self.model.nf]
        newA = np.delete(self, delcols, axis=1).view(Kinematic_matrix)
        transfer_vars(self,newA)
        newA.column_data = list(np.delete(self.column_data, delcols))
        return newA


    @property
    def c(self):
        """Removes rows corresponding to element hinges/releases"""
        Af = self.f
        idx_c = self.model.idx_c
        newA = Af[idx_c,:]
        transfer_vars(Af, newA)
        newA.row_data = Af.row_data[idx_c]
        return newA
    
    @property
    def c0(self):
        Af = self.f
        n_col = len(Af.T)
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        # delcols = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        delrows = [Af.row_data.index(tag) for tag in tags]
        newA = Af 
        for rw in delrows:
            newA[rw,:] = [0.]*n_col
        transfer_vars(Af, newA)
        # newA.row_data = list(np.delete(Af.row_data, delrows))
        return newA
    
    @property
    def o(self):
        Af = self.f
        n_col = len(Af.T)
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        # delcols = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        delrows = [Af.row_data.index(tag) for tag in tags]
        newA = Af 
        for rw in delrows:
            newA[rw,:] = [0.]*n_col
        newA = newA.del_zeros()
        transfer_vars(Af, newA)
        # newA.row_data = list(np.delete(Af.row_data, delrows))
        return newA
    
    @property
    def e(self):
        Af = self.f
        n_col = len(Af.T)
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        # delcols = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        delrows = [Af.row_data.index(tag) for tag in tags]
        newA = Af 
        for rw in delrows:
            newA[rw,:] = [0.]*n_col
        newA = newA.del_zeros()
        transfer_vars(Af, newA)
        # newA.row_data = list(np.delete(Af.row_data, delrows))
        return newA

def A_matrix(Domain, matrix=None):
    """Returns a Kinematic_matrix object"""
    return Kinematic_matrix(Domain,matrix)

class Flexibility_matrix (Structural_Matrix):
    """
    Parameters
    =========================================================================================
    model
    
    -----------------------------------------------------------------------------------------
    """
    tag = 'F'
    def __new__(cls, model, Roption=True):
        f  = np.array([elem.f_matrix() for elem in model.elems])
        Fs = scipy.linalg.block_diag(*f)


        B = Static_matrix(model)
        Fsr = emme.utilities.del_zeros(Fs)

        F = B.bari.T@(Fsr@B.bari)
        
        input_array = np.asarray(F).view(cls)
        input_array.model = model
        input_array.column_data = np.array([elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()])
        input_array.row_data = np.array([elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()])
        # input_array.rel = [rel for elem in model.elems for rel in elem.rel.values()]

        # # input_array.xx = np.asarray(Fxx).view(cls)
        # input_array.xx.model = model
        # input_array.xx.row_data = B.x.column_data
        # input_array.xx.column_data = B.x.column_data
        # # input_array.xx.rel = [rel for elem in model.elems for rel in elem.rel.values()]


        input_array.s = np.asarray(Fs).view(cls)
        input_array.s.model = model
        input_array.s.column_data = np.array([elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()])
        input_array.s.row_data =    np.array([elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()])
        # input_array.s.rel = [rel for elem in model.elems for rel in elem.rel.values()]

        return input_array

    def __init__(self, model, Roption=True):
        # self.model = model 
        # self.column_data = [elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()]
        # self.row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()]
        # self.rel = [rel for elem in model.elems for rel in elem.rel.values()]
        pass

    def __matmul__(self, Vector):
        if type(Vector) is Deformation_vector:
            vect = np.matmul(self, Vector).view(iForce_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        elif type(Vector) is iForce_vector:
            vect = np.matmul(self, Vector).view(Deformation_vector)
            # vect.part = 'continuous'
            vect.row_data = self.row_data
            vect.matrix = self

        else:
            vect = Structural_Matrix.__matmul__(self, Vector)

        return vect

    @property 
    def c(self):
        """Removes rows corresponding to element hinges/releases"""
        rows = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        newF = np.delete(self, rows, axis=0).view(type(self))
        newF = np.delete(newF, rows, axis=1).view(type(self))
        newF.row_data = list(np.delete(self.row_data, rows))
        newF.column_data = list(np.delete(self.column_data, rows))
        newF.model = self.model
        newF.tag = self.tag + "_c"
        return newF

def F_matrix(Domain):
    """Returns a Flexibility_matrix object"""    
    return Flexibility_matrix(Domain, Roption=True)

class Diag_matrix (Structural_Matrix):
    """Block diagonal matrix of element flexibility/stiffness matrices for structural model
    

    this class represents the block diagonal matrix of element flexibility or stiffness matrices
    for a structural model.
    
    """
    tag = 'F'
    def __new__(cls, arry, rc_data, model):
        basic_forces = rc_data
        arry = np.asarray(arry).view(cls)
        arry.basic_forces = basic_forces
        arry.model = model
        # arry.column_data = [elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()]
        arry.column_data = arry.row_data = np.array([q.elem.tag+'_'+str(q.number) for q in basic_forces])
        # arry.row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()]
        return arry

    def __init__(self, arry, rc_data, model):
        """Parameters
        =========================================================================================
        model
        
        """
        self.model = model 
        pass

    def __matmul__(self, Vector):
        if type(Vector) is Deformation_vector:
            vect = np.matmul(self, Vector).view(iForce_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        elif isinstance(Vector, iForce_vector):
            vect = np.matmul(self, Vector).view(Deformation_vector)
            # vect.part = 'continuous'
            vect.row_data = self.row_data
            vect.matrix = self

        else:
            vect = Structural_Matrix.__matmul__(self, Vector)
        return vect

    @property
    def c(self):
        """Removes columns corresponding to element hinges/releases"""
        idx_c = self.model.idx_c
        newA = self[:,idx_c]
        newA = newA[idx_c,:]
        transfer_vars(self, newA)
        newA.basic_forces = self.basic_forces[idx_c]
        newA.column_data = self.column_data[idx_c]
        newA.row_data = self.row_data[idx_c]
        return newA

def Fs_matrix(model, Roption=True):
    """Returns a Flexibility_matrix object"""  
    if Roption:
        f  = np.array([elem.f_matrix(Roption) for elem in model.elems])
        basic_forces = np.array([q for elem in model.elems for q in elem.basic_forces if not q.rel]) 
    else:
        f  = np.array([elem.f_matrix(Roption) for elem in model.elems])
        basic_forces = np.array([q for elem in model.elems for q in elem.basic_forces]) 
    Fs = scipy.linalg.block_diag(*f) 
    
    basic_forces = basic_forces
    Fs = Diag_matrix(Fs, basic_forces, model)
    return Fs

def Ks_matrix(model):
    """Returns a Flexibility_matrix object"""  
    k  = np.array([elem.k_matrix() for elem in model.elems])
    Ks = scipy.linalg.block_diag(*k)  
    basic_forces = np.array([q for elem in model.elems for q in elem.basic_forces])
    Ks = Diag_matrix(Ks, basic_forces, model)
    return Ks


class Stiffness_matrix (Structural_Matrix):
    """...
    Parameters
    ============
    model
    
    -----------------------------------------------------------------------------------------
    """
    tag = 'K'
    def __new__(cls, arry, model, Roption=None):

        input_array = np.asarray(arry).view(cls)
        input_array.model = model
        input_array.column_data = [str(dof) for dof in range(1, model.nt+1)]
        input_array.row_data = ['P_{'+str(dof)+'}' for dof in range(1, model.nt+1)]
        
        return input_array

    def __init__(self, arry, model, Roption=None):
        self.subs = [None]
        pass
    

    def __matmul__(self, Vector):
        if type(Vector) is Displacement_vector:
            vect = np.matmul(self, Vector).view(nForce_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        elif type(Vector) is nForce_vector:
            vect = np.matmul(self, Vector).view(Displacement_vector)
            vect.row_data = self.row_data
            vect.model = self.model
        else:
            vect = Structural_Matrix.__matmul__(self, Vector)
        return vect

    @property
    def f(self):
        delrows = [idx for idx, dof in enumerate([str(dof) for dof in range(1, self.model.nt+1)]) if int(dof) > self.model.nf]
        
        newK = np.delete(self, delrows, axis=0)
        newK = np.delete(newK, delrows, axis=1).view(type(self))
        transfer_vars(self, newK)
        
        newK.row_data = list(np.delete(self.row_data, delrows))
        newK.column_data = list(np.delete(self.column_data, delrows))
        return newK

def K_matrix(Model):
    """Returns a Stiffness_matrix object"""
    K = np.zeros((Model.nt, Model.nt))
    for elem in Model.elems:
        ke = elem.ke_matrix()
        for i, dof in enumerate(elem.dofs):
            for j, doff in enumerate(elem.dofs):
                K[int(dof)-1, int(doff)-1] += ke[i, j]

    return Stiffness_matrix(K, Model, Roption=None)

def K_tensor(Model,U=None):
    """Returns a Stiffness_matrix object"""
    K = np.zeros((Model.nt, Model.nt))
    for elem in Model.elems:
        ke = elem.ke_matrix(U)
        for i, dof in enumerate(elem.dofs):
            for j, doff in enumerate(elem.dofs):
                K[int(dof)-1, int(doff)-1] += ke[i, j]
    return K

def AssemblyTensor(Model):
    pass

def Kt_matrix(Model, State):
    """Returns a Stiffness_matrix object"""
    K = np.zeros((Model.nt, Model.nt))
    for elem in Model.elems:
        kt = elem.kt_matrix(State)
        for i, dof in enumerate(elem.dofs):
            for j, doff in enumerate(elem.dofs):
                K[int(dof)-1, int(doff)-1] += kt[i, j]

    return Stiffness_matrix(K, Model, Roption=None)


class Mass_matrix (Structural_Matrix):
    tag = 'M'
    def __new__(cls, Model):

        M = np.zeros((Model.nt, Model.nt))
        ddof = Model.ddof
        mass_dofs = []
        for node in Model.nodes:
            for i,dof in enumerate(node.dofs):
                if 'r' not in list(ddof.keys())[i]:
                    if node.mass != 0.0: mass_dofs.append(str(dof))
                    M[int(dof)-1, int(dof)-1] += node.mass


        input_array = np.asarray(M).view(cls)
        input_array.model = Model
        input_array.mass_dofs = mass_dofs
        input_array.column_data = ['u_{{'+str(dof)+'}}' for dof in range(1, Model.nt+1)]
        input_array.row_data = ['u_{{'+str(dof)+'}}' for dof in range(1, Model.nt+1)]
        return input_array

    @property
    def f(self):
        delrows = [idx for idx, dof in enumerate([str(dof) for dof in range(1, self.model.nt+1)]) if int(dof) > self.model.nf]
        
        newK = np.delete(self, delrows, axis=0)
        newK = np.delete(newK, delrows, axis=1).view(type(self))
        transfer_vars(self, newK)
        
        newK.row_data = list(np.delete(self.row_data, delrows))
        newK.column_data = list(np.delete(self.column_data, delrows))
        return newK
    
    @property
    def m(self):
        delrows = [idx for idx, dof in enumerate([str(dof) for dof in range(1, self.model.nt+1)]) if int(dof) > self.model.nf]
        
        newK = np.delete(self, delrows, axis=0)
        newK = np.delete(newK, delrows, axis=1).view(type(self))
        transfer_vars(self, newK)
        
        newK.row_data = list(np.delete(self.row_data, delrows))
        newK.column_data = list(np.delete(self.column_data, delrows))
        return newK


class Displacement_vector(column_vector):
    tag = 'U'
    def __new__(cls, Kinematic_matrix, Vector=None):
        U = np.zeros((len(Kinematic_matrix.column_data)))
        return np.asarray(U).view(cls)

    def __init__(self, Kinematic_matrix, Vector=None):
        self.matrix = Kinematic_matrix 
        self.row_data = Kinematic_matrix.column_data
        self.subs = [None]
        if Vector is not None:
            for key in Vector.row_data:
                if key in self.row_data:
                    self.set_item(key, Vector.rows([key]))

    @property
    def f(self):
        """Removes rows corresponding to fixed dofs"""
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) > self.model.nf]
        newU = np.delete(self, delrows, axis=0).view(Displacement_vector)
        newU.row_data = list(np.delete(self.row_data, delrows))
        # newU.matrix = self.matrix
        newU.model = self.model
        return newU

class nDisplacement_vector(Structural_Vector):
    tag = 'U'
    def __new__(cls, arry, model, row_data, Vector=None):
        input_array = np.asarray(arry).view(cls)
        return input_array

    def __init__(self, arry, model, row_data, Vector=None):
        self.subs = [None]
        self.model = model
        self.row_data = row_data

        if Vector is not None:
            for key in Vector.row_data:
                if key in self.row_data:
                    self.set_item(key, Vector.rows([key]))

    @property
    def f(self):
        """Removes rows corresponding to fixed dofs"""
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) > self.model.nf]
        newU = np.delete(self, delrows, axis=0).view(nDisplacement_vector)
        newU.row_data = list(np.delete(self.row_data, delrows))
        newU.model = self.model
        return newU

def U_vector(model, vector=None):
    """Returns a Displacement_vector object"""
    U = np.zeros(model.nt)
    row_data = [str(dof) for dof in range(1,model.nt+1)]
    U = nDisplacement_vector(U, model, row_data)

    if vector is not None:
        if len(vector)==len(U):
            U[:,0] = vector[:]
        elif isinstance(vector,Structural_Vector):
            for key in vector.row_data:
                if key in U.row_data:
                    U.set_item(key, vector.rows([key]))
        else:
            U[:len(vector)] = vector.flatten()
    return U


class iForce_vector(Structural_Vector):
    tag = 'Q'
    def __new__(cls, arry, model, row_data, Vector=None):
        return np.asarray(arry).view(cls)

    def __init__(self, arry, model, row_data, Vector=None):
        self.model = model
        self.subs = [None]
        self.row_data = row_data
        if Vector is not None:
            for key in Vector.row_data:
                if key in self.row_data:
                    self.set_item(key, Vector.get(key))
  #<
    @property
    def i(self):
        """Removes rows corresponding to redundant forces"""
        rdts = self.model.redundants

        tags = [q.elem.tag + str(q.nature) for q in rdts]
        delrows = [self.row_data.index(tag) for tag in tags]
        newQ = np.delete(self, delrows, axis=0).view(iForce_Vector)
        transfer_vars(self, newQ)
        newQ.subs.append('primary')
        newQ.row_data = list(np.delete(self.row_data, delrows))
        return newQ
    
    @property
    def c(self):
        """Remove rows corresponding to element hinges/releases"""
        idx_c = self.model.idx_c
        newQ = self[idx_c]
        transfer_vars(self, newQ)
        newQ.row_data = self.row_data[idx_c]
        return newQ
    
    @property
    def x(self):
        """Remove rows of corresponding to primary forces"""
        rdts = self.model.redundants
        tags = [q.elem.tag + '_'+str(q.nature) for q in rdts]
        rows = [self.row_data.index(tag) for tag in tags]
        newV = self[rows]
        newV.row_data = [self.row_data[row] for row in rows]
        transfer_vars(self, newV)
        return newV

def Q_vector(model, vector=None):
    """Returns a iForce_vector object"""   
    
    arry = np.zeros((model.nQ,1))
    row_data = np.array([elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()])
    Q = iForce_vector(arry, model, row_data)
    if vector is not None:
        if len(vector)==len(Q):
            Q[:,0] = vector[:]
        else:
            for key in vector.row_data:
                if key in Q.row_data:
                    Q.set_item(key, vector.rows([key]))
    return Q

def Q0_vector(model):
    """Returns a vector of initial element forces"""   
    arry = np.concatenate([elem.q0_vector() for elem in model.elems])
    row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.q.keys()] 
    return iForce_vector(arry, model, row_data)

def Qpl_vector(model):
    """Returns a vector of element plastic capacities""" 

    Qp_pos = [elem.Qp['+'][key] for elem in model.elems for key in elem.Qp['+']]
    Qp_neg = [elem.Qp['-'][key] for elem in model.elems for key in elem.Qp['-']]
    row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.Qp['-']]

    # del_idx = np.where(~Bf.any(axis=0))[0]
    # Qp_pos = np.delete(Qp_pos, del_idx)
    # Qp_neg = np.delete(Qp_neg, del_idx)
    # row_data = np.delete(row_data, del_idx)
    column_data = ['Q_{pl}^+', 'Q_{pl}^-']

    Qpl = nKinematic_matrix(np.array([Qp_pos, Qp_neg]).T, model, (row_data, column_data))
    return Qpl

def Qp_vector(model):
    """Returns a vector of element plastic capacities""" 
    B = B_matrix(model)
    Bf = B.f
    Qp_pos = [elem.Qp['+'][key] for elem in model.elems for key in elem.Qp['+']]
    Qp_neg = [elem.Qp['-'][key] for elem in model.elems for key in elem.Qp['-']]
    row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.Qp['-']]

    del_idx = np.where(~Bf.any(axis=0))[0]
    Qp_pos = np.delete(Qp_pos, del_idx)
    Qp_neg = np.delete(Qp_neg, del_idx)
    row_data = np.delete(row_data, del_idx)
    column_data = ['Q_{pl}^+', 'Q_{pl}^-']

    Qpl = nStatic_matrix(np.array([Qp_pos, Qp_neg]).T, model, (row_data, column_data))
    return Qpl

class nForce_vector(Structural_Vector):
    tag = 'P'
    def __new__(cls, arry, model, row_data, Vector=None):
        return np.asarray(arry).view(cls)

    def __init__(self, arry, model, row_data, Vector=None):
        self.model = model
        self.subs = [None]
        self.row_data = row_data
        if Vector is not None:
            for key in Vector.row_data:
                if key in self.row_data:
                    self.set_item(key, Vector.get(key))


    @property
    def f(self):
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) > self.model.nf]
        
        newP = np.delete(self, delrows, axis=0).view(type(self))
        transfer_vars(self, newP)
        # newB.model = self.model
        newP.row_data = list(np.delete(self.row_data, delrows))
        # newB.column_data = self.column_data
        return newP

    @property
    def d(self):
        """Removes rows corresponding to free dofs"""
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) <= self.model.nf]
        newP = np.delete(self, delrows, axis=0).view(type(self))
        newP.model = self.model
        newP.row_data = list(np.delete(self.row_data, delrows))
        newP.subs.append('reactions')
        return newP
        
  #<

def P_vector(model, vector=None):
    P = np.zeros(model.nt)
    for node in model.nodes:
        p = node.p_vector()
        for i, dof in enumerate(node.dofs):
            P[int(dof)-1] += p[i]
    row_data = [str(dof) for dof in range(1, model.nt+1)]
    return nForce_vector(P, model, row_data)



# def P0_vector(model):
#     """Returns a _ object"""   
#     arry = np.concatenate([elem.p0_vector() for elem in model.elems]) 
#     row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.v.keys()] 
#     return nForce_vector(arry, model, row_data)

def P0_vector(model):
    P = np.zeros(model.nt)
    for elem in model.elems:
        dofs = elem.dofs
        if hasattr(elem, 'q0_vector'):
            p0 = elem.bg_matrix()@elem.q0_vector()
        else:
            p0 = np.zeros(len(dofs))

        for i,df in enumerate(dofs):
            P[int(df)-1] +=  p0[i]

    row_data = [str(dof) for dof in range(1, model.nt+1)]
    return nForce_vector(P, model, row_data)

def Pw_vector(model):
    P = np.zeros(model.nt)
    for elem in model.elems:
        dofs = elem.dofs
        if len(dofs)==6:
            P[int(dofs[0])-1] +=  elem.w['y']*elem.L/2*elem.sn
            P[int(dofs[1])-1] += -elem.w['y']*elem.L/2*elem.cs
            P[int(dofs[3])-1] +=  elem.w['y']*elem.L/2*elem.sn
            P[int(dofs[4])-1] += -elem.w['y']*elem.L/2*elem.cs
        else:
            pw = elem.pw_vector()
            for i,df in enumerate(dofs):
                P[int(df)-1] +=  pw[i]


    row_data = [str(dof) for dof in range(1, model.nt+1)]
    return nForce_vector(P, model, row_data)


class Deformation_vector(Structural_Vector):
    tag = "V"
    def __new__(cls, arry, model, row_data, Vector=None):
        input_array = np.asarray(arry).view(cls)
        return input_array

    def __init__(self, arry, model, row_data, Vector=None):
        self.model = model
        self.row_data = row_data
    
    @property
    def c(self):
        """Removes rows corresponding to element hinges/releases"""
        idx_c = self.model.idx_c
        newQ = self[idx_c]
        transfer_vars(self, newQ)
        newQ.row_data = self.row_data[idx_c]
        return newQ
    
    @property
    def i(self):
        """Removes rows corresponding to redundant forces"""
        rdts = self.model.redundants
        tags = [q.elem.tag + '_'+str(q.nature) for q in rdts]
        delrows = [self.row_data.index(tag) for tag in tags]
        newV = np.delete(self, delrows, axis=0).view(type(self))
        transfer_vars(self,newV)
        newV.row_data = list(np.delete(self.row_data, delrows))
        newV.subs.append('primary')
        return newV

    @property
    def x(self):
        """Removes rows of corresponding to primary forces
        
        """
        rdts = self.model.redundants
        tags = [q.elem.tag + '_'+str(q.nature) for q in rdts]
        rows = [self.row_data.index(tag) for tag in tags]
        newV = self[rows]
        newV.row_data = [self.row_data[row] for row in rows]
        transfer_vars(self, newV)
        return newV

def V_vector(model, vector=None):
    """Returns a Deformation_vector object"""    
    arry = np.zeros((model.nQ,1))
    row_data = np.array([elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()])
    V = Deformation_vector(arry, model, row_data)
    if vector is not None:
        if len(vector)==len(V):
            V[:,0] = vector[:]
        else:
            for key in vector.row_data:
                if key in V.row_data:
                    V.set_item(key, vector.rows([key]))
    return V

def V0_vector(model):
    """Returns a Deformation_vector object"""   
    arry = np.concatenate([elem.v0_vector() for elem in model.elems]) 
    row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.v.keys()] 
    return Deformation_vector(arry, model, row_data)

def Aub_matrix(model, alpha):
    """Return the interaction upperbound matrix"""  
    aub  = np.array([elem.aub_matrix(alpha) for elem in model.elems])
    A = scipy.linalg.block_diag(*aub)
    return A
