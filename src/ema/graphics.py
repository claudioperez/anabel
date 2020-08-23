import matplotlib.pyplot as plt
import numpy as np
from ema.elements import Truss
import ema.matrices as mv
from ema.matrices import U_vector


node_style = [{'color':'black', 'marker':'s', 'markersize':3, 'zorder':2},
                               {'marker':'s', 'markersize':3, 'zorder':2}]

rxn_style = {
    0: {'marker':'>', 'color':'black', 'markeredgewidth':1, 'markersize':9,
            'fillstyle':'none',  'zorder':3,},
    1: {'marker':'^', 'color':'black', 'markeredgewidth':1, 'markersize':9,
            'fillstyle':'none', 'zorder':3,},
    2: {'marker':'s', 'color':'black', 'markeredgewidth':1, 'markersize':9,
            'fillstyle':'none', 'zorder':3,},
    'pin':   {'color':'black', 'marker':'t', 'markersize':5, 'zorder':2},
    'roller':{'color':'black', 'marker':'s', 'markersize':5, 'zorder':2},
    'fixed': {'color':'black', 'marker':'s', 'markersize':5, 'zorder':2}
}

hinge_style = [{'s':20, 'zorder':3, 'facecolors':'white', 'edgecolors':'black'},
               {'s':20, 'zorder':3, 'facecolors':'red', 'edgecolors':'black'}] 

elem_style = [{'color': 'grey', 'linewidth':2, 'linestyle':'-', 'zorder':1},
              {'color':'black', 'linewidth':1, 'linestyle':'-', 'zorder':1}] 

chord_style = [{'color':'black', 'linewidth':1, 'linestyle':':', 'zorder':2}] 

axis_style = {'color':'grey', 'linewidth':1, 'linestyle':'--'}


def plot_U(model, U_vect, ax, scale=None, color=None, chords=False):
    """Only works for 2D"""
    
    if scale is None: scale = 10 # factor to scale up displacements
    if color is None: 
        color='red'
    A = mv.Kinematic_matrix(model)
    # print(A.f)
    U = U_vector(model, vector=U_vect)
    plot_structure(model, ax)
    for node in model.nodes:
        delta = [0.,0.]
        for i,dof in enumerate(node.dofs[0:2]):
            if not node.rxns[i]: 
                try: delta[i] = U[U.row_data.index(str(dof))]
                except: pass
        x = node.x
        y = node.y
        plt.plot( x, y, **node_style[0])
        plt.plot( x+delta[0]*scale, y+delta[1]*scale, color=color, **node_style[1])
        
    ###< Plot Chords
    # if chords:
    for elem in model.elems:
        X = np.array([[elem.nodes[0].x, elem.nodes[0].y], 
                    [elem.nodes[1].x, elem.nodes[1].y]])
        delta = np.array([[0.,0.],[0.,0.]])

        for j, node in enumerate(elem.nodes):
            for i,dof in enumerate(node.dofs[0:2]):
                if not node.rxns[i]: 
                    try:
                        delta[j,i] = U[U.row_data.index(str(dof))]
                    except:
                        pass
        x = np.linspace(X[0,0]+delta[0,0]*scale, X[1,0]+delta[1,0]*scale, 3)
        y = np.linspace(X[0,1]+delta[0,1]*scale, X[1,1]+delta[1,1]*scale, 3)
        plt.plot(x, y, ":", zorder = 1, color=color)
        
    # plot deformed curve
    if model.ndf > 2:
        V = A.c0@U.f # Element deformation vector
        n_curve=20   # number of plotting points along element

        for elem in model.elems:
            delta = np.array([[0.,0.],[0.,0.]])
            if hasattr(elem, "Elastic_curve"):
                X = np.array([[elem.nodes[0].x, elem.nodes[0].y], 
                              [elem.nodes[1].x, elem.nodes[1].y]])
                v_tags = [elem.tag+'_2', elem.tag+'_3']
                v = [V.get(v_tags[0]),V.get(v_tags[1])]
                xl = np.linspace(0, elem.L, n_curve)
                xl, yl = elem.Elastic_curve(xl, v, scale=scale, global_coord=True)
                

                for j, node in enumerate(elem.nodes): # Node displacements
                    for i,dof in enumerate(node.dofs[0:2]):
                        if not node.rxns[i]: # if there is no reaction at dof `i`...
                            try:
                                delta[j,i] = U[U.row_data.index(str(dof))]
                            except:
                                pass

                x0 = np.linspace(delta[0,0]*scale, delta[1,0]*scale, n_curve)
                y0 = np.linspace(delta[0,1]*scale, delta[1,1]*scale, n_curve)
                x = xl + x0
                y = yl + y0
                plt.plot(x, y, zorder = 1, color=color)

    # Plot undeformed chords
    n = 3 
    for elem in model.elems:
        x = np.linspace(elem.nodes[0].x, elem.nodes[1].x, n)
        y = np.linspace(elem.nodes[0].y, elem.nodes[1].y, n)
        plt.plot(x, y, **elem_style[0])

    f = 0.5
    for hinge in model.hinges:
        if hinge.node == hinge.elem.nodes[0]:
            x = hinge.node.x + f*hinge.elem.cs
            y = hinge.node.y + f*hinge.elem.sn
        else:
            x = hinge.node.x - f*hinge.elem.cs
            y = hinge.node.y - f*hinge.elem.sn
        plt.scatter(x, y, **hinge_style[0])

def plot_modes(model, shapes, ax, scale=None, color=None, label=None):
    """Only works for 2D"""
    if scale is None: scale = 5 # factor to scale up displacements
    if color is None: 
        color='red'
    A = mv.Kinematic_matrix(model)
    U = mv.Displacement_vector(A, shapes)
    X = []
    Y = []
    # plot_structure(model, ax)
    for node in model.nodes:
        delta = [0.,0.]
        for i,dof in enumerate(node.dofs[0:2]):
            if not node.rxns[i]: 
                try: delta[i] = U[U.row_data.index(str(dof))]
                except: pass
        x = node.x
        y = node.y
        X.append(x+delta[0]*scale)
        Y.append(y+delta[1]*scale)
        plt.plot( x, y, **node_style[0])
        # plt.plot( x+delta[0]*scale, y+delta[1]*scale, color=color, **node_style[1])

        
    ###< Plot Chords
    ne = len(model.elems)-1
    for i, elem in enumerate(model.elems):
        X = np.array([[elem.nodes[0].x, elem.nodes[0].y], 
                      [elem.nodes[1].x, elem.nodes[1].y]])
        delta = np.array([[0.,0.],[0.,0.]])

        for j, node in enumerate(elem.nodes):
            for i,dof in enumerate(node.dofs[0:2]):
                if not node.rxns[i]:
                    try:
                        delta[j,i] = U[U.row_data.index(str(dof))]
                    except:
                        pass

        x = np.linspace(X[0,0]+delta[0,0]*scale, X[1,0]+delta[1,0]*scale, 3)
        y = np.linspace(X[0,1]+delta[0,1]*scale, X[1,1]+delta[1,1]*scale, 3)

        plt.plot(X[:,0], X[:,1], **elem_style[0])
        if i==ne: 
            plt.plot(x, y, ':', zorder = 1, color=color, label=label)
        else:
            plt.plot(x, y, ':', zorder = 1, color=color)
    
    # plot deformed curve
    V = A.c0@U.f
    n_curve=20
    for elem in model.elems:
        delta = np.array([[0.,0.],[0.,0.]])
        if hasattr(elem, "Elastic_curve"):
            X = np.array([[elem.nodes[0].x, elem.nodes[0].y], 
                          [elem.nodes[1].x, elem.nodes[1].y]])
            v_tags = [elem.tag+'_2', elem.tag+'_3']
            v = [V.get(v_tags[0]),V.get(v_tags[1])]
            xl = np.linspace(0, elem.L, n_curve)
            xl, yl = elem.Elastic_curve(xl, v, scale=scale, global_coord=True)
            
            for j, node in enumerate(elem.nodes):
                for i,dof in enumerate(node.dofs[0:2]):
                    if not node.rxns[i]: 
                        try:
                            delta[j,i] = U[U.row_data.index(str(dof))]
                        except:
                            pass

            x0 = np.linspace(delta[0,0]*scale, delta[1,0]*scale, n_curve)
            y0 = np.linspace(delta[0,1]*scale, delta[1,1]*scale, n_curve)
            x = xl + x0
            y = yl + y0
            plt.plot(x, y, zorder = 1, color=color)
    
def plot_structure(Model, ax=None, label=False):
    if ax is None: _,ax = plt.subplots()
    n =3
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_aspect('equal')

    ###< Plot Elements
    f = 0.5 # parameter controling element label distance from element
    for elem in Model.elems:
        x = np.linspace(elem.nodes[0].x0, elem.nodes[1].x0, n)
        y = np.linspace(elem.nodes[0].y0, elem.nodes[1].y0, n)
        # plt.plot(x, y, zorder = 1, color ='grey')
        ax.plot(x, y, **elem_style[0])
        if label:
            # label element tag
            xl = x[1] - elem.sn*f
            yl = y[1] - elem.cs*f
            ax.annotate(elem.tag, xy=(xl, yl))

    f = 0.4
    if Model.ndf == 3:
        for elem in Model.elems:
            if type(elem) is Truss:
                x = elem.nodes[0].x0 + f*elem.cs
                y = elem.nodes[0].y0 + f*elem.sn
                ax.scatter(x, y, s=20, zorder=2, facecolors='white', edgecolors='black')
                x = elem.nodes[1].x0 - f*elem.cs
                y = elem.nodes[1].y0 - f*elem.sn
                ax.scatter(x, y, s=20, zorder=2, facecolors='white', edgecolors='black')

    ###>
    # show hinges
    f = 0.5
    for hinge in Model.hinges:
        if hinge.node == hinge.elem.nodes[0]:
            x = hinge.node.x + f*hinge.elem.cs
            y = hinge.node.y + f*hinge.elem.sn
        else:
            x = hinge.node.x - f*hinge.elem.cs
            y = hinge.node.y - f*hinge.elem.sn
        ax.scatter(x, y, **hinge_style[0])

    # show reactions
    offset = 0.5
    rx_offset = [offset, 0., 0.]
    ry_offset = [0., offset, 0.]
    for rxn in Model.rxns:
        if rxn.node.rxns == [1, 1, 0]: # pinned reaction
            pass
            #plt.scatter(x, y, s=25, zorder=2, color='black', marker = '^')
        elif rxn.node.rxns == [1, 0, 0]: # x - roller
            pass
            #plt.scatter(x, y, s=25, zorder=2, color='black', marker = 'o')
        elif rxn.node.rxns == [0, 1, 0]: # y - roller
            pass
        elif rxn.node.rxns == [1, 1, 1]: # fixed reaction
            pass
        
    for node in Model.nodes:
        for i, rxn in enumerate(node.rxns):
            if rxn:
                x = node.x0-rx_offset[i]#*np.sign(node.elems[0].cs)
                y = node.y0-ry_offset[i]#*np.sign(node.elems[0].sn)
                ax.plot(x, y, **rxn_style[i])

    # plot nodes
    f = 0.4 # factor to tweak annotation distance
    for node in Model.nodes:
        ax.plot(node.x,node.y, color = 'black', marker = 's')
        if label:
            if sum(node.rxns) == 0:
                ax.annotate(node.tag, xy=(f+node.x, 0.5*f+node.y), zorder = 3, color = 'blue')
            else:
                tag = node.tag + " "+ str(node.rxns)
                ax.annotate(tag, xy=(f+node.x, 0.5*f+node.y), zorder = 3, color = 'blue')
    return ax

def plot_beam(Model, ax,label=False):
    n =3
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.set_aspect('equal')

    ###< Plot Elements
    f = 0.5 # parameter controling element label distance from element
    for elem in Model.elems:
        x = np.linspace(elem.nodes[0].x, elem.nodes[1].x, n)
        y = np.linspace(0.0, 0.0, n)
        plt.plot(x, y, zorder = 1, color ='grey')
        # label element tag
        xl = (x[0] + x[-1])/2
        yl = 0.0002
        if label:
            ax.annotate(elem.tag, xy=(xl, yl))

    # show hinges
    f = 0.1
    for hinge in Model.hinges:
        if hinge.node == hinge.elem.nodes[0]:
            x = hinge.node.x + f*hinge.elem.L
            y = 0.0 
        else:
            x = hinge.node.x - f*hinge.elem.L
            y = 0.0 

        plt.scatter(x, y, s=20, zorder=2, facecolors = 'white', edgecolors='black')

    # plot nodes
    y_off = 0.0004 # factor to tweak annotation distance
    for node in Model.nodes:
        plt.plot(node.x,0.0, color = 'black', marker = 's')
        if sum(node.rxns) == 0:
            ax.annotate(node.tag, xy=(node.x, y_off), zorder = 3, color = 'blue')
        else:
            tag = node.tag + " "+ str(node.rxns)
            if label:
                ax.annotate(tag, xy=(node.x, y_off), zorder = 3, color = 'blue')

def plot_structure3d(Model, ax):
    n =3

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

        ###< Plot Elements
    f = 0.9 # parameter controling element label distance from element
    for elem in Model.elems:
        x = np.linspace(elem.nodes[0].x, elem.nodes[1].x, n)
        y = np.linspace(elem.nodes[0].y, elem.nodes[1].y, n)
        z = np.linspace(elem.nodes[0].z, elem.nodes[1].z, n)
        xl = x[1] - elem.sn*f
        yl = y[1]- elem.cs*f
        zl = z[1] - elem.cz*f
        ax.plot(x, y, z, color ='black')
 #         ax.annotate(elem.tag, xyz = (xl, yl, zl))
        ax.text(xl, yl, zl, elem.tag, color='red')
        # plot nodes
    f = 0.4 # factor to tweak annotation distance
    for node in Model.nodes:
        ax.scatter(node.x,node.y, node.z, color = 'black', marker = 's')
 #             if sum(node.rxns) == 0:

 #                 ax.annotate(node.tag, xy=(f+node.x, 0.5*f+node.y), zorder = 3, color = 'blue')
 #             else:
 #                 tag = node.tag + " "+ str(node.rxns)
 #                 ax.annotate(tag, xy=(f+node.x, 0.5*f+node.y), zorder = 3, color = 'blue')
    return ax

def plot_2dshape(xyz,N,node=1,ax=None):
    if ax is None: _,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})

    xx, yy = np.meshgrid(xyz[:,0],xyz[:,1])
    z = N[(i-1)*2,0](xx,yy)






