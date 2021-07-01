
class SP_Constraint():
    def __init__(self, node, dirn, dof=None):
        self.node = node
        self.dirn = dirn
        self.dof  = dof

    def __repr__(self):
        return 'rxn-{}'.format(self.dirn)


class Hinge():
    def __init__(self, elem, node):
        self.elem = elem
        self.node = node

