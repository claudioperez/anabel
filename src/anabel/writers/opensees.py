from .writer import ModelWriter

class OpenSeesWriter(ModelWriter):
    def __init__(self, model):
        self.model = model
        self.comment_char = "#"

    def dump_initialize(self):
        ndm, ndf = self.model.ndm, self.model.ndf
        cmds = ""
        cmds += f"# Create ModelBuilder (with {ndm} dimensions and {ndf} DOF/node)"
        cmds += f"\nmodel BasicBuilder -ndm {ndm} -ndf {ndf}"
        return cmds

    def dump_constraints(self):
        model = self.model
        cmds = ""
        for n in model.nodes:
            cmds += f"\nfix {n.tag} {' '.join(str(i) for i in n.constraints)}"
        return cmds

    def dump_connectivity(self):
        cmds = ""
        for nd in self.model.nodes:
            cmds += "\n" + nd.dump_opensees()

        for el in self.model.elems:
            cmds += "\n" + el.dump_opensees()
        return cmds


