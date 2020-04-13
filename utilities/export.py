import ema.elements
# from ema.matvecs import P_vector
import numpy as np
from datetime import datetime
FEDEASmap = {
    "2D beam":"Lin2dFrm",
    "2D truss": "LinTruss",
}

def ModelData(self):
        ModelData = {
            "nn"        :self.nn,     # property      -number of nodes in structural model
            "ndm"       :self.ndm,    # attribute     -dimension of structural model"
            "XYZ"       :self.XYZ,    #               -node coordinates, nodes are stored columnwise"
            "ne"        :self.ne,     # property      -number of elements"
            "CON"       :self.CON,    # attribute     -node connectivity array"
            "ElemName"  :[elem.type for elem in self.elems],    # property      -cell array of element names"
            "nen"       :[],          #
            "nq(el)"    :[],          # property      -no. of basic forces for element el
            "ndf"       :[],          #
            "nt"        :self.nt,     # property      -total number of degrees of freedom
            "BOUN"      :[],          # attribute     -boundary conditions, nodes are stored columnwise"
            "nf"        :self.nf,     #               -number of free degrees of freedom"
            "DOF"       :self.DOF,    # attribute     -array with degree of freedom numbering, nodes are stored columnwise"
        }
        return ModelData

def FEDEAS(Domain):
    print('CleanStart')

    # Node Definitions
    print('\n% Node Definitions')
    for i, node in enumerate(Domain.nodes):
        print("XYZ({},:) = [{} {}];".format(i+1, node.x, node.y))
    
    # Connectivity
    print('\n% Connections')
    for i, elem in enumerate(Domain.elems):
        ni = Domain.nodes.index(elem.nodes[0])+1
        nj = Domain.nodes.index(elem.nodes[1])+1
        print("CON({},:) = [{} {}];".format(i+1, ni, nj))
    
    # Fixities
    print('\n% Boundary Conditions')
    for i, node in enumerate(Domain.nodes):
        if 1 in node.rxns:
            if Domain.ndf ==2:
                nd = Domain.nodes.index(node)+1
                rx = node.rxns
                print("BOUN({},:) = [{} {}];".format(nd, rx[0], rx[1]))
            if Domain.ndf==3:
                nd = Domain.nodes.index(node)+1
                rx = node.rxns
                print("BOUN({},:) = [{} {} {}];".format(nd, rx[0], rx[1], rx[2]))

    # Element types
    print('\n% Specify element type')
    for i, elem in enumerate(Domain.elems): 
        print("ElemName{{{}}} = '{}';".format(i+1, FEDEASmap[elem.type]))


    print("\n% Create model")
    print("Model = Create_SimpleModel (XYZ,CON,BOUN,ElemName);")

    # Element properties
    print("\n% Element properties")

    print("\n ElemData = cell(Model.ne,1);")
 
    for i, elem in enumerate(Domain.elems): 
        print("\n% Element: {}".format(elem.tag))
        print('ElemData{{{}}}.A = {};'.format(i+1, elem.A))
        print('ElemData{{{}}}.E = {};'.format(i+1, elem.E))
        print('ElemData{{{}}}.Np = {};'.format(i+1, elem.Qpl[0,0]))
        if hasattr(elem,'I'): print('\nElemData{{{}}}.I = {};'.format(i+1, elem.I))

        if elem.nv > 1:
            rel = [1 if rel else 0 for rel in elem.rel.values()]
            print("ElemData{{{}}}.Release = [{};{};{}];".format(i+1, rel[0], rel[1], rel[2]))
            print('ElemData{{{}}}.Mp = {};'.format(i+1, elem.Qpl[1,0]))


    # Element Loads
    print("\n%% Element loads")
    for i, elem in enumerate(Domain.elems): 
        if type(elem) is ema.elements.Beam:
            print("\n% Element: {}".format(elem.tag))
            print('ElemData{{{}}}.w = [{}; {}];'.format(i+1, elem.w['x'], elem.w['y']))

    # Nodal Loads
    print("\n%% Nodal loads")
    print('Pf = zeros({});'.format(Domain.nf))
    for node in Domain.nodes:
        p = node.p_vector()
        for i, dof in enumerate(node.dofs):
            if p[i] != 0.:
                print('Pf({}) = {};'.format(dof, p[i]))
    print('Loading.Pref = Pf;')


class FEDEAS_func:
    def __init__(self,model,filename=None,simple=True):
        if filename is None: filename = 'ReturnModel.m'
        self.time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model = model 
        self.filename = filename
        self.simple= simple

    def string(self):
        Domain = self.model
        script = ''
        script += 'function [Model,ElemData,Loading] = {}()'.format(self.filename)


        script += '\n' + 'CleanStart'

        # Node Definitions
        script += '\n' + '\n% Node Definitions'
        for i, node in enumerate(Domain.nodes):
            script += '\n' + "XYZ({},:) = [{} {}];".format(i+1, node.x, node.y)
        
        # Connectivity
        script += '\n' + '\n% Connections'
        for i, elem in enumerate(Domain.elems):
            ni = Domain.nodes.index(elem.nodes[0])+1
            nj = Domain.nodes.index(elem.nodes[1])+1
            script += '\n' + "CON({},:) = [{} {}];".format(i+1, ni, nj)
        
        # Fixities
        script += '\n' + '\n% Boundary Conditions'
        for i, node in enumerate(Domain.nodes):
            if 1 in node.rxns:
                if Domain.ndf ==2:
                    nd = Domain.nodes.index(node)+1
                    rx = node.rxns
                    script += '\n' + "BOUN({},:) = [{} {}];".format(nd, rx[0], rx[1])
                if Domain.ndf==3:
                    nd = Domain.nodes.index(node)+1
                    rx = node.rxns
                    script += '\n' + "BOUN({},:) = [{} {} {}];".format(nd, rx[0], rx[1], rx[2])

        # Element types
        script += '\n' + '\n% Specify element type'
        for i, elem in enumerate(Domain.elems): 
            script += '\n' + "ElemName{{{}}} = '{}';".format(i+1, FEDEASmap[elem.type])


        script += '\n' + "\n% Create model"
        if self.simple:
            script += '\n' + "Model = Create_SimpleModel (XYZ,CON,BOUN,ElemName);"
        else:
            script += '\n' + "Model = Create_Model (XYZ,CON,BOUN,ElemName);"

        # Element properties
        script += '\n' + "\n% Element properties"

        script += '\n' + "\n ElemData = cell(Model.ne,1);"
    
        for i, elem in enumerate(Domain.elems): 
            script += '\n' + "\n% Element: {}".format(elem.tag)
            try: script += '\n' + 'ElemData{{{}}}.A = {};'.format(i+1, float(elem.A.numpy()))
            except: script += '\n' + 'ElemData{{{}}}.A = {};'.format(i+1, float(elem.A))

            try: script += '\n' + 'ElemData{{{}}}.E = {};'.format(i+1, float(elem.E.numpy()))
            except: script += '\n' + 'ElemData{{{}}}.E = {};'.format(i+1, float(elem.E))

            script += '\n' + 'ElemData{{{}}}.Np = {};'.format(i+1, elem.Qpl[0,0])
            
            if hasattr(elem,'I'): 
                try: script += '\n' + '\nElemData{{{}}}.I = {};'.format(i+1, float(elem.I.numpy()))
                except: script += '\nElemData{{{}}}.I = {};'.format(i+1, elem.I)

            if elem.nv > 1:
                script += '\n' + 'ElemData{{{}}}.Mp = {};'.format(i+1, elem.Qpl[1,0])
                rel = [1 if rel else 0 for rel in elem.rel.values()]
                script += '\n' + "ElemData{{{}}}.Release = [{};{};{}];".format(i+1, rel[0], rel[1], rel[2])


        # Element Loads
        script += '\n' + "\n%% Element loads"
        for i, elem in enumerate(Domain.elems): 
            if type(elem) is ema.elements.Beam:
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