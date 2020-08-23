import numpy as np
import scipy.linalg
# import sympy as sp
import matplotlib.pyplot as plt
# import pandas as pd
import ema.elements 
import sympy as sp


def transfer_vars(item1, item2):
    for key, value in item1.__dict__.items():
        item2.__dict__[key] = value

class Structural_Vector(np.ndarray):
    column_data = ["vector"]
    row_data = None
    subs = [None]
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

    def __add__(self, v1):
        if type(v1) is type(self):
            out = np.add(self, v1).view(type(self))
            transfer_vars(self,out)
        return out
    
    def get(self, component):
        row = self.row_data.index(component)
        return self[row]

    def set_item(self, key, value):
        idx = self.row_data.index(key)
        self[idx] = value

    def rows(self, component):
        rows = [self.row_data.index(item) for item in component]
        newV = self[rows]
        newV.row_data = list(np.array(self.row_data)[rows])
        newV.model = self.model
        return newV

    @property
    def df(self):
        row_data = ['$'+tag+'$' for tag in self.row_data]
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
            out.column_data = ['.']
        else:
            out = np.matmul(self,other)
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
            row_data = ['$'+tag+'$' for tag in self.row_data]
            column_data = ['$'+tag+'$' for tag in self.column_data]
        else:
            row_data = self.row_data
            column_data = self.column_data
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
        return np.linalg.matrix_rank(self)
    
    @property
    def lns(self):
        return scipy.linalg.null_space(self.T)
    
    @property
    def nls(self):
        return scipy.linalg.null_space(self)
    
    @property
    def ker(self):
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
        idxr = self.row_data.index(row_name)
        idxc = self.column_data.index(col_name)
        return self[idxr, idxc]

    def rows(self, component):
        rows = [self.row_data.index(item) for item in component]
        newM = self[rows,:]
        newM.model = self.model
        newM.column_data = self.column_data
        newM.row_data = list(np.array(self.row_data)[rows])
        return newM

    def del_zeros(self):
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
            # try: newA.rel = self.rel
            # except: pass
            return newA


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
        # if matrix is not None:
        #     fullnq = sum([len(elem.rel) for elem in model.elems])
        #     self[:,:] = np.zeros((model.nt,fullnq))
        #     for idxr, rw in enumerate(self.row_data):
        #         if rw in matrix.row_data:
        #             for idxc, cl in enumerate(self.column_data):
        #                 if cl in matrix.column_data:
        #                     self[idxr,idxc] = matrix.get(rw,cl)

    def __matmul__(self, Vector):
        if type(Vector) is P_vector:
            vect = np.matmul(self, Vector).view(Q_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        elif type(Vector) is Q_vector:
            vect = np.matmul(self, Vector).view(P_vector) 
            vect.row_data = self.row_data
            vect.matrix = self
        else:
            vect = Structural_Matrix.__matmul__(self, Vector)
        return vect
  #<
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
        
        Bf = self.f
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        # delcols = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        delcols = [Bf.column_data.index(tag) for tag in tags]
        newB = np.delete(Bf, delcols, axis=1).view(type(self))
        transfer_vars(Bf, newB)
        newB.column_data = list(np.delete(Bf.column_data, delcols))
        # newB.row_data = self.row_data
        # newB.rel = self.rel
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
        
        # newB.row_data = self.row_data
        # newB.rel = self.rel
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
        newB.column_data = [self.column_data[col] for col in cols]
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
        # nQ = self.model.nQ
        nQ = len(self.column_data)
        nx = len(self.model.redundants)

        Bbarxi = self.barxi

        Bbarx = Structural_Matrix(np.zeros((nQ,nx)))
        transfer_vars(self, Bbarx)
        Bbarx.column_data = Bbarxi.column_data
        Bbarx.row_data = self.column_data

        # for idxr, rw in enumerate(Bbarx.row_data):
        #     if rw in Bbarxi.row_data:
        #         for idxc, cl in enumerate(Bbarx.column_data):
        #             print(rw,cl)
        #             if cl in Bbarxi.column_data:
        #                 Bbarx[idxr,idxc] = Bbarxi.get(rw,cl)
        #             elif cl==rw:
        #                 Bbarx[idxr, idxc] = 1.
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

class Kinematic_matrix(Structural_Matrix, np.ndarray):
    ranges = {
        'f': 'free dof columns',
        'i': 'primary force/deformation rows',
        'x': 'redundant force/deformation rows',
        'd': 'reaction force/deformation rows',
        'c': 'continuous (hingeless) force/deformation rows'
    }
    def __new__(cls, arry, model, rcitems):
        return np.asarray(arry).view(cls)

    def __init__(self, arry, model, rcitems):
        if rng is None:
            self.rng = None
        self.model = model
        self.column_data = [str(dof) for dof in range(1,model.nt+1)]
        # self.column_data = ['U_{'+str(dof)+'}' for dof in range(1, model.nt+1)]
        self.row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.v.keys()]
        
        
        # self.rel = [rel for elem in model.elems for rel in elem.rel.values()]
        self.idx_h = []
        if matrix is not None:
            fullnq = sum([len(elem.rel) for elem in model.elems])
            self[:,:] = np.zeros((model.nt,fullnq))
            for idxr, rw in enumerate(self.row_data):
                if rw in matrix.row_data:
                    for idxc, cl in enumerate(self.column_data):
                        if cl in matrix.column_data:
                            self[idxr,idxc] = matrix.get(rw,cl)
  #<
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

    def remove(self, component):
        if "frame-axial" in component:
            """Remove rows corresponding to frame axial deformations"""

            rows=[]
            for elem in self.model.elems:
                if type(elem)== ema.elements.Beam:
                    rows.append(self.row_data.index(elem.tag+'_1'))

            newA = np.delete(self, rows, axis=0).view(Kinematic_matrix)
            newA.row_data = list(np.delete(self.row_data, rows))
            newA.model = self.model
            newA.column_data = self.column_data
            # newA.rel = list(np.delete(self.rel, rows))
            return newA 

        if type(component) is list:
            newA = self
            for item in component:
                if item in self.column_data: 
                    delcol = self.column_data.index(item)
                    newA = np.delete(self, delcol, axis=1).view(Kinematic_matrix)
                    newA.row_data = self.row_data
                    newA.model = self.model
                    newA.column_data = list(np.delete(self.column_data, delcol))
                    # newA.rel = self.rel
                else:
                    delrow = self.row_data.index(item)
                    newA = np.delete(self, delrow, axis=0).view(Kinematic_matrix)
                    newA.column_data = self.column_data
                    newA.model = self.model
                    newA.row_data = list(np.delete(self.row_data, delrow))
                    # newA.rel = self.rel
            return newA

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
        newA.column_data = list(np.delete(self.column_data, delcols))
        newA.row_data = self.row_data
        newA.model = self.model
        # newA.rel = self.rel
        return newA

    @property
    def i(self):
        """Removes rows corresponding to redundant forces"""
        # tags = [q.elem.tag + str(q.nature) for q in rdts]
        # delrows = [self.row_data.index(tag) for tag in tags]
        # newA = np.delete(self, delrows, axis=0).view(Kinematic_matrix)
        # newA.row_data = list(np.delete(self.row_data, delrows))
        # newA.column_data = self.column_data
        # newA.model = self.model
        # newA.rel = list(np.delete(self.rel, delrows))
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
        # delrows = [idx for idx, rel in enumerate(self.model.rel) if rel==1]
        tags = [elem.tag + "_" + rel for elem in self.model.elems for rel in elem.rel if elem.rel[rel]]
        delrows = [Af.row_data.index(tag) for tag in tags]
        newA = np.delete(Af, delrows, axis=0).view(Kinematic_matrix)
        transfer_vars(Af, newA)
        newA.row_data = list(np.delete(Af.row_data, delrows))
        # newA.column_data = Af.column_data
        # newA.model = Af.model
        # newA.rel = list(np.delete(Af.rel, delrows))
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

class Flexibility_matrix (Structural_Matrix):
    tag = 'F'
    def __new__(cls, arry, model, rcitems):
        return np.asarray(arry).view(cls)

    def __init__(self, arry, model, rcitems):
        # self.model = model 
        # self.column_data = [elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()]
        # self.row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()]
        # self.rel = [rel for elem in model.elems for rel in elem.rel.values()]
        pass

    def __matmul__(self, Vector):
        if type(Vector) is Deformation_vector:
            vect = np.matmul(self, Vector).view(Q_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        elif type(Vector) is Q_vector:
            vect = np.matmul(self, Vector).view(Deformation_vector)
            # vect.part = 'continuous'
            vect.row_data = self.row_data
            vect.matrix = self

        else:
            vect = Structural_Matrix.__matmul__(self, Vector)

        return vect
  #<
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

class Stiffness_matrix (Structural_Matrix):
    tag = 'K'
    def __new__(cls, arry, model, rcitems):
        return np.asarray(arry).view(cls)

    def __init__(self, arry, model, rcitems):
        self.subs = [None]
        pass
    
    def __matmul__(self, Vector):
        if type(Vector) is Displacement_vector:
            vect = np.matmul(self, Vector).view(P_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        elif type(Vector) is P_vector:
            vect = np.matmul(self, Vector).view(Displacement_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        else:
            vect = Structural_Matrix.__matmul__(self, Vector)
        return vect
  #<
    @property
    def f(self):
        delrows = [idx for idx, dof in enumerate([str(dof) for dof in range(1, self.model.nt+1)]) if int(dof) > self.model.nf]
        
        newK = np.delete(self, delrows, axis=0)
        newK = np.delete(newK, delrows, axis=1).view(type(self))
        transfer_vars(self, newK)
        
        newK.row_data = list(np.delete(self.row_data, delrows))
        newK.column_data = list(np.delete(self.column_data, delrows))
        return newK

class Mass_matrix(Structural_Matrix):
    tag = 'M'
    def __new__(cls, arry, model, rcitems):
        return np.asarray(arry).view(cls)
  #<
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


class Deformation_vector(Structural_Vector):
    tag = "V"
    def __new__(cls, arry, model, Vector=None):
        if part == 'initial':
            V = np.concatenate([elem.v0_vector() for elem in Matrix.model.elems])
        else:
            V = np.zeros((len(Matrix.row_data)))
        input_array = np.asarray(V).view(cls)
        input_array.matrix = Matrix
        return input_array

    def __init__(self, arry, model, Vector=None):
        self.part = part 
        self.subs = [None]
        if part is None: 
            self.o = Deformation_vector(Matrix, part='initial')
            self.o.subs.append('initial')
            pass
        self.row_data = Matrix.row_data
        
        if Vector is not None:
            for key in Vector.row_data:
                if key in self.row_data:
                    self.set_item(key, Vector.rows([key]))
  #<    
    @property 
    def c(self):
        """Removes rows corresponding to element hinges/releases"""
        # rows = [idx for idx, rel in enumerate(self.matrix.model.rel) if rel]
        tags = [elem.tag + "_" + rel for elem in self.matrix.model.elems for rel in elem.rel if elem.rel[rel]]
        delrows = [self.row_data.index(tag) for tag in tags]
        
        newV = np.delete(self, delrows, axis=0).view(type(self))
        transfer_vars(self,newV)
        newV.row_data = list(np.delete(self.row_data, delrows))
        newV.subs.append('continuous')
        return newV  
    
    @property
    def i(self):
        """Removes rows corresponding to redundant forces"""
        rdts = self.matrix.model.redundants
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
        rdts = self.matrix.model.redundants
        tags = [q.elem.tag + '_'+str(q.nature) for q in rdts]
        rows = [self.row_data.index(tag) for tag in tags]
        newV = self[rows]
        newV.row_data = [self.row_data[row] for row in rows]
        transfer_vars(self, newV)
        return newV
        
class Load_Vector(Structural_Vector):
    tag = "P"
    def __new__(cls, arry, model, Vector=None):
        P = np.zeros(Static_matrix.shape[0])
        if part is None:
            for node in Static_matrix.model.nodes:
                p = node.p_vector()
                for i, dof in enumerate(node.dofs):
                    P[int(dof)-1] += p[i]
            # P = np.concatenate([node.p_vector() for node in Static_matrix.model.nodes])
       
        elif part =='elem-load':
            for elem in Static_matrix.model.elems:
                if type(elem) is ema.Beam:
                    dofs = elem.dofs
                    P[int(dofs[0])-1] +=  elem.w['y']*elem.L/2*elem.sn
                    P[int(dofs[1])-1] += -elem.w['y']*elem.L/2*elem.cs
                    P[int(dofs[3])-1] +=  elem.w['y']*elem.L/2*elem.sn
                    P[int(dofs[4])-1] += -elem.w['y']*elem.L/2*elem.cs
                
        return np.asarray(P).view(cls)

    def __init__(self, arry, model, Vector=None):
        self.model = model
        self.subs = [None]
        self.row_data = Static_matrix.row_data
  #<
    @property
    def f(self):
        """Removes rows corresponding to fixed dofs"""
        
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) > self.matrix.model.nf]
        newP = np.delete(self, delrows, axis=0).view(type(self))
        newP.row_data = list(np.delete(self.row_data, delrows))
        newP.model = self.model
        newP.subs.append('free')
        return newP
    
    @property
    def wf(self):
        return self.w.f

    @property
    def d(self):
        """Removes rows corresponding to free dofs"""
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) <= self.matrix.model.nf]
        newP = np.delete(self, delrows, axis=0).view(type(self))
        newP.model = self.model
        newP.row_data = list(np.delete(self.row_data, delrows))
        newP.subs.append('reactions')
        return newP

class Pw_vector(Structural_Vector):
    tag = "P_w"
    def __new__(cls, model):
        P = np.zeros(model.nt)
        for elem in model.elems:
            if type(elem) is ema.Beam:
                dofs = elem.dofs
                P[int(dofs[0])-1] +=  elem.w['y']*elem.L/2*elem.sn
                P[int(dofs[1])-1] += -elem.w['y']*elem.L/2*elem.cs
                P[int(dofs[3])-1] +=  elem.w['y']*elem.L/2*elem.sn
                P[int(dofs[4])-1] += -elem.w['y']*elem.L/2*elem.cs
        return np.asarray(P).view(cls)

    def __init__(self, model, part=None):
        self.model = model
        self.subs = [None]
        if part == None:
            self.w = P_vector(Static_matrix, part='elem-load')
            self.w.subs.append('elem-load')
        self.row_data = [str(dof) for dof in range(1, model.nt+1)]
        
    @property
    def f(self):
        """Removes rows corresponding to fixed dofs"""
        
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) > self.model.nf]
        newP = np.delete(self, delrows, axis=0).view(type(self))
        newP.row_data = list(np.delete(self.row_data, delrows))
        newP.model = self.model
        newP.subs.append('free')
        return newP

    @property
    def d(self):
        """Removes rows corresponding to free dofs"""
        delrows = [idx for idx, dof in enumerate(self.row_data) if int(dof) <= self.matrix.model.nf]
        newP = np.delete(self, delrows, axis=0).view(type(self))
        newP.model = self.model
        newP.row_data = list(np.delete(self.row_data, delrows))
        newP.subs.append('reactions')
        return newP

class Displacement_vector(Structural_Vector):
    tag = 'U'
    def __new__(cls, arry, model, Vector=None):
        return np.asarray(arry).view(cls)

    def __init__(self, arry, model, Vector=None):
        self.model = model 
        self.row_data = Kinematic_matrix.column_data
        self.subs = [None]
        if Vector is not None:
            for key in Vector.row_data:
                if key in self.row_data:
                    self.set_item(key, Vector.rows([key]))
  #<
    @property
    def f(self):
        """Removes rows corresponding to fixed dofs"""
        delrows = [idx for idx, dof in enumerate(self.matrix.column_data) if int(dof) > self.matrix.model.nf]
        newU = np.delete(self, delrows, axis=0).view(Displacement_vector)
        newU.row_data = list(np.delete(self.row_data, delrows))
        newU.model = self.model
        return newU

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
    def x(self):
        pass

class DiagFlex_matrix (Structural_Matrix):
    """FS_MATRIX block diagonal matrix of element flexibity matrices for structural model
    FS = FS_MATRIX (MODEL,ELEMDATA,ROPTION)
    the function sets up the block diagonal matrix of element flexibility matrices FS
    for the structural model specified in data structure MODEL with element property
    information in cell array ELEMDATA;
    if ROPTION=0, element release information is not accounted for in setting up Fs (default=1)
    
    Parameters
    =========================================================================================
    model
    
    -----------------------------------------------------------------------------------------
    """
    tag = 'F'
    def __new__(cls, model, Roption=None):
        return np.asarray(arry).view(cls)

    def __init__(self, model, Roption=None):
        # self.model = model 
        # self.column_data = [elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()]
        # self.row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()]
        # self.rel = [rel for elem in model.elems for rel in elem.rel.values()]
        pass

    def __matmul__(self, Vector):
        if type(Vector) is Deformation_vector:
            vect = np.matmul(self, Vector).view(Q_vector)
            vect.row_data = self.row_data
            vect.matrix = self
        elif type(Vector) is Q_vector:
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

class DiagStif_matrix (Structural_Matrix):
    """...
    Parameters
    =========================================================================================
    model
    
    -----------------------------------------------------------------------------------------
    """    
    tag = 'K'
    def __new__(cls, arry, model, Roption=None):
        return np.asarray(arry).view(cls)

    def __init__(self, model, Roption=None):
        self.subs = [None]
        pass
