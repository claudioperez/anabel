import numpy as np

from anabel.abstract import ModelComponent
import anabel.backend as anp

class Node(ModelComponent):
    def __init__(self, model, name: str, ndf, xyz, mass=None):
        if mass is None:
            # it would be more efficient to use [0.0]*ndf, 
            # but this would cause unexpected behavior if
            # one attempted to later change the mass at a
            # particular dof.
            mass = [0 for i in range(ndf)]

        self.xyz = self.coords = np.array([xi for xi in xyz if xi is not None])

        self._tag = name if isinstance(name, int) else None
        
        self._name = name 
 
        #self.x: float = xyz[0]
        #self.y: float = xyz[1]
        #self.z: float = xyz[2] if len(xyz) > 2 else None

        
        self.rxns = [0]*ndf
        self.model = self._domain = model
        self.mass = mass
        self.elems = []

        self.p = {dof:0.0 for dof in model.dof_names}

    x, y, z = (
        property(lambda self, i=i: self.coords[i]) for i in range(3)
    )

    @property
    def constraints(self):
        return self.rxns

    @property
    def tag(self):
        if self._tag is not None:
            return self._tag
        else:
            return self.domain.nodes.index(self)

    def __repr__(self):
        return 'nd-{}'.format(self.tag)

    def p_vector(self):
        return np.array(list(self.p.values()))

        
    @property
    def dofs(self):
        """Nodal DOF array"""
        # if self.model.DOF == None: self.model.numDOF()
        idx = self.model.nodes.index(self)
        return np.asarray(self.model.DOF[idx],dtype=int)
    
    def dump_opensees(self):
        coords = " ".join(f"{x:6g}" for x in self.xyz)
        mass = f"-mass {' '.join(f'{i:6g}' for i in self.mass)}" if hasattr(self, "mass") else ""
        return f"node {self.tag} {coords} {mass} ; # {self.name}\n"

