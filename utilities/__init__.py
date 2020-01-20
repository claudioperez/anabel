import numpy as np
import pandas as pd
import sympy as sp
from . import export
import scipy.linalg
from . import ipyutils
from . import numerical

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

matmul = {
    'B':{'Q':'P'}
}



def del_zeros(mat):
    delrows = np.where(~mat.any(axis=1))[0]
    delcols = np.where(~mat.any(axis=0))[0]

    newM = np.delete(mat, delcols, axis=1)
    newM = np.delete(newM, delrows, axis=0)
    return newM

def transfer_vars(item1, item2):
    for key, value in item1.__dict__.items():
        item2.__dict__[key] = value

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
            transfer_vars(self,out)
        else:
            out = super().__add__(other)
        return out
    
    def get(self, key):
        idx = np.array([i for i,j in enumerate(self.row_data) if str(j) == key], dtype=int)
        row = self.row_data.index(component)
        return self[idx]

    def set_item(self, key, value):
        idx = np.array([i for i,j in enumerate(self.row_data) if str(j) == key], dtype=int)
        self[idx] = value

    def rows(self, component):
        rows = [self.row_data.index(item) for item in component]
        idx = np.array([i for i,j in enumerate(self.row_data) if str(j) == key])
        newV = self[idx]
        newV.row_data = list(np.array(self.row_data)[idx])
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
        return pd.DataFrame(np.around(self,14), index=row_data, columns=column_data)

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
        ker.row_data = self.column_data 
        ker.column_data = [str(i+1) for i in range(len(ker[0]))]
        ker.model = self.model
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

