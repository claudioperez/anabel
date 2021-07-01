from .writer import ModelWriter
import anabel.elements

# from anabel.matvecs import P_vector
import numpy as np
from datetime import datetime
FEDEASmap = {
    "2D beam": "Lin2dFrm",
    anabel.elements.ElasticBeam: "LEFrame",
    "2D truss": "LETruss",
}

def ModelData(self):
        ModelData = {
            "nn"        :self.nn,     # property      -number of nodes in structural model
            "ndm"       :self.ndm,    # attribute     -dimension of structural model"
            "XYZ"       :self.XYZ,    #               -node coordinates, nodes are stored columnwise"
            "ne"        :self.ne,     # property      -number of elements"
            "CON"       :self.CON,    # attribute     -node connectivity array"
            #"ElemName"  :[elem.type for elem in self.elems],    # property      -cell array of element names"
            "nen"       :[],          #
            "nq(el)"    :[],          # property      -no. of basic forces for element el
            "ndf"       :[],          #
            "nt"        :self.nt,     # property      -total number of degrees of freedom
            "BOUN"      :[],          # attribute     -boundary conditions, nodes are stored columnwise"
            "nf"        :self.nf,     #               -number of free degrees of freedom"
            "DOF"       :self.DOF,    # attribute     -array with degree of freedom numbering, nodes are stored columnwise"
        }
        return ModelData


class FEDEAS_Writer(ModelWriter):
    def __init__(self,model,filename=None,simple=True):
        if filename is None: filename = 'ReturnModel.m'
        self.time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model = model 
        self.filename = filename
        self.simple= simple

        self.comment_char = "%"

    def dump_initialize(self):
        Domain = self.model
        script = ''
        #script += 'function [Model,ElemData,Loading] = {}()'.format(self.filename)
        script += '\n' + 'CleanStart'
        return script

        # Node Definitions
    def dump_connectivity(self):
        Domain = self.model
        script = '% Node Definitions\n'
        for i, node in enumerate(Domain.nodes):
            script += '\n' + "XYZ({},:) = [{}];".format(i+1, " ".join(f"{x:8.8}" for x in node.coords))
        
        # Connectivity
        script += '\n% Connections\n'
        for i, elem in enumerate(Domain.elems):
            ni = Domain.nodes.index(elem.nodes[0])+1
            nj = Domain.nodes.index(elem.nodes[1])+1
            script += '\n' + "CON({},:) = [{} {}];".format(i+1, ni, nj)
        
        script += '\n' + "\n% Create model"
        if self.simple:
            script += '\n' + "Model = Create_SimpleModel (XYZ,CON,BOUN,ElemName);"
        else:
            script += '\n' + "Model = Create_Model (XYZ,CON,BOUN,ElemName);"
        return script

    def dump_constraints(self): 
        # Fixities
        script = ""
        Domain = self.model
        for i, node in enumerate(Domain.nodes):
            if 1 in node.rxns:
                if Domain.ndf ==2:
                    nd = Domain.nodes.index(node)+1
                    rx = node.rxns
                    script += "BOUN({},:) = [{} {}];\n".format(nd, rx[0], rx[1])
                if Domain.ndf==3:
                    nd = Domain.nodes.index(node)+1
                    rx = node.rxns
                    script += "BOUN({},:) = [{} {} {}];\n".format(nd, *rx)
        return script

    def dump_elements(self): 
        # Element types
        Domain = self.model
        script = '% Specify element type'
        for i, elem in enumerate(Domain.elems): 
            script += '\n' + "ElemName{{{}}} = '{}';".format(i+1, FEDEASmap[elem.type if hasattr(elem,"type") else type(elem)])

        # Element properties
        script += '\n' + "\n% Element properties"

        script += '\n' + f"\nElemData = cell({len(Domain.elems)},1);"
    
        for i, elem in enumerate(Domain.elems):
            script += "\n% Element: {}".format(elem.name)
            script += '\n' + 'ElemData{{{}}}.A = {};'.format(i+1, float(elem.A))

            script += '\n' + 'ElemData{{{}}}.E = {};'.format(i+1, elem.E)

            script += '\n' + 'ElemData{{{}}}.Np = {};'.format(i+1, elem.Qpl[0,0])
 
            script += '\n' + "ElemData{{{}}}.Geom = 'GL';".format(i+1, elem.Qpl[0,0])
            
            if hasattr(elem,'I'): 
                try: script += '\n' + '\nElemData{{{}}}.I = {};'.format(i+1, float(elem.I.numpy()))
                except: script += '\nElemData{{{}}}.I = {};'.format(i+1, elem.I)

            if elem.nv > 1:
                script += '\n' + 'ElemData{{{}}}.Mp = {};'.format(i+1, elem.Qpl[1,0])
                rel = [1 if rel else 0 for rel in elem.rel.values()]
                script += '\n' + "ElemData{{{}}}.Release = [{};{};{}];".format(i+1, rel[0], rel[1], rel[2])
        return script


    def dump_loading(self): 
        # Element Loads
        Domain = self.model
        script = '\n' + "\n%% Element loads"
        for i, elem in enumerate(Domain.elems):
            if type(elem) is anabel.elements.Beam:
                script += '\n' + "\n% Element: {}".format(elem.tag)
                script += '\n' + 'ElemData{{{}}}.w = [{}; {}];'.format(i+1, elem.w['x'], elem.w['y'])

        # Nodal Loads
        script += '\n' + "\n%% Nodal loads"
        script += '\n' + 'Pf = zeros({});'.format(Domain.nf)
        for node in Domain.nodes:
            p = node.p_vector()
            for i, dof in enumerate(node.dofs):
                if p[i] != 0.:
                    script += '\n' + 'Pf({}) = {};'.format(dof, p[i])
        script += '\n' + 'Loading.Pref = Pf;'
        return script

    def write(self):
        f = open(self.filename,"w+")
        f.write(self.string())
        f.close()


