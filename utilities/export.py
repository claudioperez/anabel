import ema.elements
# from ema.matvecs import P_vector
import numpy as np

def ModelData(self):
        ModelData = {
            "nn"        :self.nn,     # property      -number of nodes in structural model
            "ndm"       :self.ndm,   # attribute     -dimension of structural model"
            "XYZ"       :self.XYZ,    #               -node coordinates, nodes are stored columnwise"
            "ne"        :self.ne,     # property      -number of elements"
            "CON"       :self.CON,    # attribute     -node connectivity array"
            "ElemName"  :[elem.type for elem in self.elems],    # property      -cell array of element names"
            "nen"       :[],    #
            "nq(el)"    :[],    # property      -no of basic forces for element el
            "ndf"       :[],    #
            "nt"        :self.nt,     # property      -total number of degrees of freedom
            "BOUN"      :[],    # attribute     -boundary conditions, nodes are stored columnwise"
            "nf"        :self.nf,     #               -number of free degrees of freedom"
            "DOF"       :self.DOF,    # attribute     -array with degree of freedom numbering, nodes are stored columnwise"
        }
        return ModelData

def FEDEAS(Domain):
    FEDEASmap = {
        "2D beam":"Lin2dFrm",
        "2D truss": "LinTruss",
    }
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

        if type(elem) is not ema.elements.Truss:
            print('\nElemData{{{}}}.I = {};'.format(i+1, elem.I))
            print('ElemData{{{}}}.Mp = {};'.format(i+1, elem.Qpl[1,0]))
            rel = [1 if rel else 0 for rel in elem.rel.values()]
            print("ElemData{{{}}}.Release = [{};{};{}];".format(i+1, rel[0], rel[1], rel[2]))


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


