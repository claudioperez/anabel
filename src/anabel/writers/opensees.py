from .writer import ModelWriter

class OpenSeesWriter(ModelWriter):
    def __init__(self, model=None):
        self.model = model
        self.comment_char = "#"

    def dump_initialize(self, definitions={}):
        cmds = "# Parameters\n" + "\n".join(f"set {k} {v};" for k,v in definitions.items()) + "\n\n"
        ndm, ndf = self.model.ndm, self.model.ndf
        dof_keys = "dx dy dz rx ry rz"
        dofs = " ".join( str(i) for i in range(1, ndf + 1))
        cmds += f"# Create ModelBuilder (with {ndm} dimensions and {ndf} DOF/node)\n"
        cmds += f"model BasicBuilder -ndm {ndm} -ndf {ndf}\n"
        cmds += f"lassign {{{dofs}}} {dof_keys}"
        return cmds

    @classmethod 
    def dump_elements(self, *elems, definitions={}):
        transforms = set()
        cmds = "\n".join(f"set {k} {v};" for k,v in definitions.items()) + "\n"
        for el in elems:
            cmds += "\n" + el.dump_opensees()
            if hasattr(el, "_transform") and el._transform:
                transforms.update({el._transform})

        return "".join(t.dump_opensees() for t in transforms) + cmds

    @classmethod 
    def dump_sections(self, *sections, definitions={}):
        cmds = "\n".join(f"set {k} {v};" for k,v in definitions.items()) + "\n"
        for sect in sections:
            cmds += sect.dump_opensees()
        return cmds
    
    def dump_materials(self, *materials, definitions={}):
        cmds = "\n".join(f"set {k} {v};" for k,v in definitions.items()) + "\n"
        for mat in self.model.materials.values():
            try:
                cmds += mat.dump_opensees()
            except:
                pass
        return cmds

    def dump_constraints(self, definitions={}):
        cmds = "\n".join(f"set {k} {v};" for k,v in definitions.items()) + "\n"
        model = self.model
        for n in model.nodes:
            if any(n.constraints):
                cmds += f"fix {n.tag} {' '.join(str(i) for i in n.constraints)}\n"
        return cmds

    def dump_connectivity(self, definitions={}):
        cmds = "\n".join(f"set {k} {v};" for k,v in definitions.items()) + "\n"
        for nd in self.model.nodes:
            cmds +=  nd.dump_opensees()
     
        cmds += self.dump_elements(*self.model.elems)
        return cmds


