

class ModelWriter:
    def __init__(self, model):
        self.model = model
    
    def dump(self, **kwds):
        model = self.model
        c = self.comment_char
        ndm, ndf = model.ndm, model.ndf
        cmds  = model.header.replace("\n", f"\n{c} ") + "\n"


        cmds += "\n" + self.heading(1, "Initializations")
        cmds += self.dump_initialize()
        
        cmds += "\n" + self.heading(1, "Elements")
        cmds += self.dump_elements()

        cmds += "\n" + self.heading(1, "Connectivity")
        cmds += self.dump_connectivity()

        cmds += "\n" + self.heading(1, "Constraints")
        cmds += self.dump_constraints()

        return "\n\n".join([cmds])

    def dump_materials(self)->str:
        return ""
    
    def dump_sections(self)->str:
        pass
    
    def dump_elements(self)->str:
        return ""

    def dump_assembly(self)->str:
        pass

    def heading(self, level, text):
        c = self.comment_char
        if level == 1:
            return f"""{c}{'='*60}\n{c} {text}\n{c}{'='*60}\n"""
        elif level == 2:
            return f"\n{c} {text}\n{c}{'-'*50}"


