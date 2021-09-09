from .elastic_beam import ElasticBeam
from .zero_length import ZeroLength


class LinearTransform:
    def __init__(self, tag, vecxz=None, joint_offsets=None):
        self.tag = tag
        self.vecxz = vecxz
        self.joint_offsets = joint_offsets
    def dump_opensees(self, *other):
        if self.joint_offsets:
            jntOffset = "-jntOffset " + " ".join(f'{f}' for j in self.joint_offsets for f in j)
        else:
            jntOffset = ""
        return f"geomTransf Linear {self.tag} {' '.join(str(i) for i in self.vecxz)} {jntOffset}\n"

