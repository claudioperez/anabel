"""
# Section Modeling

High-level section modeling API.
"""
from functools import partial
import jax
import anon
import anon.atom as anp
import scipy.optimize
import numpy as onp
import matplotlib
import matplotlib.pyplot as plt

#import anabel.graphics.get_axes as get_axes
import anon.quad
quad_points = anon.quad.quad_points


def epsi(y,epsa,kappa):  return epsa - y*kappa;

ei = epsi

@anon.dual.generator(2,params="params")
def section2d(yi, dA, nIP, mat, **kwds):
    """Generate a section response function

    Studies
    -------

    """
    resp = mat
    state = {
        "...": [mat.origin[2] for i in range(nIP)],
        "updated": False
    }
    params = {...: [{} for i in range(nIP)]}

    s0 = anp.zeros((3,1))

    def jacx(e,s=s0,state=state,params=params): # e = [ eps_a,  kappa ]
        ϵ = [epsi(yi[i], *e.flatten()) for i in range(nIP)]
        st = [
            resp(ϵ[i], None, states[i], **params[...][i]) 
            for i in range(nIP)
        ]
        state = {"...": [r[2] for r in mat_resp]} 
        return anp.array([
        [ sum(st[i][2]["Et"]*dA[i] for i in range(nIP)),
         -sum(st[i][2]["Et"]*yi[i]*dA[i] for i in range(nIP))],

        [-sum(st[i][2]["Et"]*yi[i]*dA[i] for i in range(nIP)),
            sum(st[i][2]["Et"]*yi[i]**2*dA[i] for i in range(nIP))] ])

    def main(e,s=s0,state=state,params=params): # e = [ eps_a,  kappa ]
        ϵ = [epsi(yi[i], *e.flatten()) for i in range(nIP)]
        mat_resp = [
            resp(ϵ[i], None, state["..."][i], **params[...][i]) 
            for i in range(nIP)
        ]
        s = anp.array([
            [sum([mat_resp[i][1]*dA[i]       for i in range(nIP)])],
            [-sum([mat_resp[i][1]*dA[i]*yi[i] for i in range(nIP)])]
        ])
        state = {"...": [r[2] for r in mat_resp],"updated": True}
        return e, s, state
    return locals()

class Section:
    def assemble(self):
        return section2d(**self.SectData)


class CompositeSection(Section):
    def __init__(self, Y, DY, DZ, quad, y_shift = 0.0, mat=None):
        self.Y = Y
        self.DY = DY
        self.DZ = DZ
        self.quad = quad
        self.nr = nr = len(Y) # number of rectangles
        u,du = [],[]
        for i in range(nr):
            loc, wght = quad_points(**quad[i])
            u.append(loc)
            du.append(wght)

        nIPs = [len(ui) for ui in u]
        nIP = sum(nIPs)
        # Length of integration intervals in reference domain
        DU = [sum(du[i]) for i in range(nr)]

        # Transform integration point locations
        yi = [float( Y[i] + DY[i]/DU[i] * u[i][j] )\
                for i in range(nr) for j in range(nIPs[i])]

        dy = [float( DY[i]/DU[i] * du[i][j] )\
                for i in range(nr) for j in range(nIPs[i])]

        dz = [DZ[i] for i in range(nr) for j in range(nIPs[i])]

        yi, dy, dz = map(list, zip( *sorted(zip(yi, dy, dz) )))

        dA = [ dy[i]*dz[i] for i in range(nIP)]

        Qm = onp.array([[*dA], [-y*da for y,da in zip(yi,dA)]])

        yrc = sum(y*dY*dZ for y,dY,dZ in zip(Y,DY,DZ))/sum(dY*dZ for dY,dZ in zip(DY,DZ))

        Izrq = sum(yi[i]**2*dA[i] for i in range(nIP)) # I w.r.t  z @ yref using quadrature

        Izr =  sum(DZ[i]*DY[i]**3/12 + DZ[i]*DY[i]*(Y[i])**2 for i in range(nr))

        Izc =  sum(DZ[i]*DY[i]**3/12 + DZ[i]*DY[i]*(Y[i]+yrc)**2 for i in range(nr))

        self.SectData = {'nIP': nIP,'dA': dA,'yi': yi,'Qm':Qm, 'yrc':yrc,'Izrq':Izrq,'Izr':Izr,'Izc':Izc,'mat':mat}
        self.__dict__.update(self.SectData)

    def plot(self,
            show_properties=True,
            plain=False,
            show_quad=True,
            show_dims=True,
            annotate=True,
        ):
        """Plot a composite cross section.

        """
        if plain:
            show_properties = show_quad = show_dims = False
        if show_properties:
            fig = plt.figure(constrained_layout=True)
            gs = fig.add_gridspec(1,5)
            axp = fig.add_subplot(gs[0,3:-1])
            label = "\n".join(["${}$:\t\t{:0.4}".format(k,v) for k,v in self.properties().items()])
            axp.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')
            axp.set_autoscale_on(True)
            axp.axis("off")

            ax = fig.add_subplot(gs[0,:3])
        else:
            fig, ax = plt.subplots()

        import matplotlib.patches as patches
        pc = matplotlib.collections.PatchCollection(
            [ matplotlib.patches.Rectangle((-self.DZ[i]/2.0, y-self.DY[i]/2.0), self.DZ[i], self.DY[i]) for i,y in enumerate(self.Y) ],
            facecolor = "grey",
            edgecolor = "grey",
            alpha = 0.3,
        )
        ax.add_collection(pc)
        if show_quad:
            # Plot integration points
            ax.scatter([0.0]*self.nIP,self.yi,s=[da**1.1 for da in self.dA], color="r", alpha=0.4)
        if annotate:
            self.annotate(ax)
        ax.set_autoscale_on(True)
        x_lim = 1.2*max(self.DZ)/2.0
        ax.set_xlim(-x_lim, x_lim)
        y_lim = 1.1*(max(self.DY)/2.0 + max(self.Y))
        ax.set_ylim(-y_lim, y_lim)
        ax.set_aspect(1)
        if not show_dims:
            ax.axis("off")
        return fig, ax


def Composite_Section(Y, DY, DZ, quad, y_shift = 0.0, mat=None):
    nr = len(Y) # number of rectangles
    u,du = [],[]
    for i in range(nr):
        loc, wght = quad_points(**quad[i])
        u.append(loc)
        du.append(wght)

    nIPs = [len(ui) for ui in u]
    nIP = sum(nIPs)

    DU = [sum(du[i]) for i in range(nr)]

    yi = [float( Y[i] + DY[i]/DU[i] * u[i][j] )\
            for i in range(nr) for j in range(nIPs[i])]

    dy = [float( DY[i]/DU[i] * du[i][j] )\
            for i in range(nr) for j in range(nIPs[i])]
    dz = [DZ[i] for i in range(nr) for j in range(nIPs[i])]

    yi, dy, dz = map(list, zip( *sorted(zip(yi, dy, dz) )))

    dA = [ dy[i]*dz[i] for i in range(nIP)]

    Qm = onp.array([[*dA], [-y*da for y,da in zip(yi,dA)]])

    yrc = sum(y*dY*dZ for y,dY,dZ in zip(Y,DY,DZ))/sum(dY*dZ for dY,dZ in zip(DY,DZ))

    Izrq = sum(yi[i]**2*dA[i] for i in range(nIP)) # I   w.r.t  z @ yref using quadrature

    Izr =  sum(DZ[i]*DY[i]**3/12 + DZ[i]*DY[i]*(Y[i])**2 for i in range(nr))

    Izc =  sum(DZ[i]*DY[i]**3/12 + DZ[i]*DY[i]*(Y[i]+yrc)**2 for i in range(nr))

    SectData = {'nIP': nIP,'dA': dA,'yi': yi,'Qm':Qm, 'yrc':yrc,'Izrq':Izrq,'Izr':Izr,'Izc':Izc,'mat':mat}

    return SectData

class VerticalSection:
    def plot_limit(self, fy=1.0, **kwds):
        pass

    def plot_yield(self, fy=1.0, **kwds):
        #fig, ax = get_axes(kwds)
        pass

class Tee(CompositeSection,VerticalSection):
    """
    ![](img/sections/tee-dims.svg)

    ```
          bf
    _____________ 
    |           | tf
    -------------
        |   |
        |   |
        |   |
        -----
         tw
    ```
    """
    def __init__(self, d=None, quad=None, b=None, bf=None,tf=None,tw=None,alpha=None, beta=None, yref=None, mat=None):
        """
        Parameters
        ----------
        tf,tw, bf, d: float
            shape parameters
        """
        if quad is None:
            quad = [
                {"n": 8,"rule": "gauss-legendre"},
                {"n": 4,"rule": "gauss-legendre"}
            ]
        if tf is None:
            tf = (1-alpha)*d 
            bf = b 
            tw = beta*b
        else:
            pass
        self.tf = tf
        self.bf = bf
        self.tw = tw
        self.area = area = bf*tf + (d - tf) *tw
        # distance from top edge to centroid
        self.y_centroid = y_c = 1/(2*area) * (tw*d**2 + (bf-tw)*tf**2)
        self.y_plastic = d - area/(2*tw) if tf < area/(2.0*bf) else area/(2.0*bf)
        # Moment of inertia
        self.moi = bf*tf**3/12.0 + (bf*tf)*(tf*0.5*y_c)**2 + tw*(d-tf)**3/12 + tw*(d-tf)*((d+tf)*0.5-y_c)**2

        if yref is None:
            yref = (d-tf)/2 + tf - self.y_centroid
            #yref = 0.0

        #Yref = yref
        Yref = -yref
        Y  = [ Yref,  (d-tf)/2 + tf/2 + Yref]
        
        DY = (d-tf, tf)
        DZ = [ tw , bf]
        
        super().__init__(Y,DY,DZ,quad,mat=mat)
    
    def properties(self):
        return {
            #"I_{xx}": None,
            "A": self.area,
            "I_{xx}": self.moi,
            "y_c": self.y_centroid,
            "y_{pna}": self.y_plastic
        }
    def annotate(self,ax):
        props = dict(
            linewidth=0.5,
            color="k"
        )
        
        tf = self.tf
        bf = self.DZ[1]
        bw = self.DZ[0]
        d = sum(self.DY)
        Y = self.Y
        mid = sum(Y)/2
        print(mid)
        # bf
        ax.plot([-bf/2, -bf/20], [(Y[1] + tf/2) * 1.2]*2, "-|", markevery=[ 0], **props)
        ax.plot([ bf/2,  bf/20], [(Y[1] + tf/2) * 1.2]*2, "-|", markevery=[ 0], **props)
        ax.annotate("$b_f$",[0.0, (Y[1] + tf/2) * 1.2])
        # d
        ax.plot([(bw+bf)/4]*2, [(Y[1] + tf/2), 0.2*(Y[1] + tf/2)], "-_", markevery=[ 0], **props)
        ax.plot([(bw+bf)/4]*2, [0.2*(Y[0]-bw/2), Y[0]-bw/2], "-_", markevery=[-1], **props)
        ax.annotate("$d$", [(bw+bf)/4, mid])
        # tf
        ax.annotate("$t_f$",[1.1*bf/2, Y[1]])
        # bw
        ax.plot([-bw/2, -bw/10], [(Y[0] - tf/2) * 1.2]*2, "-|", markevery=[ 0], **props)
        ax.plot([ bw/2,  bw/10], [(Y[0] - tf/2) * 1.2]*2, "-|", markevery=[ 0], **props)
        ax.annotate("$b_w$",[-0.05*bw, (Y[0] - tf/2) * 1.2])

class Rectangle(CompositeSection,VerticalSection):
    def __init__(self, b, d, quad=None, yref=0.0,mat=None,**kwds):
        """Rectangular cross section

        """
        if quad is None:
            quad = [{"n": 8,"rule": "mid"}]
        self.quad = quad
        Y = [-yref]
        
        DY = [d]
        DZ = [b]
        
        SectData = Composite_Section(Y,DY,DZ,quad, mat=mat)

        # Properties
        A = b*d
        Z = 1/4*b*d**2
        I = 1/12*b*d**3
        S = I/d*2
        SectData['prop'] = dict(A=A,I=I,S=S,Z=Z)
        SectData["mat"] = mat
            
        self.SectData = SectData



def T_Sect(d, quad, b=None, bf=None,tf=None,tw=None,alpha=None, beta=None, yref=0.0, MatData=None):
    """
    ```
    _____________
    |           |
    -------------
        |   |
        |   |
        |   |
        -----
    ```
    """
    if tf is None:
        tf = (1-alpha)*d 
        bf = b 
        tw = beta*b 

    Yref = -yref
    Y  = [ Yref,  (d-tf)/2 + tf/2+ Yref]
    
    DY = (d-tf, tf)
    DZ = [ tw , bf]
    
    SectData = Composite_Section(Y,DY,DZ,quad, MatData=MatData)
    
    return SectData

def I_Sect(b,d,alpha,beta,quad, yref=0.0, MatData=None):
    tf = (1 - alpha*d)*0.5
    bf = b 
    tw = beta*b
    
    Yref = -yref

    Y  = [Yref, 0.5*(d-tf) + Yref, -0.5*(d-tf) + Yref]
    
    DY = (tf,d-2*tf, tf)
    DZ = [bf, tw, bf]
    
    SectData = Composite_Section(Y,DY,DZ,quad, MatData=MatData)
    
    return SectData

def TC_Sect(d, bf, tw, quad, yref=0.0,tf=None, ymf=None, MatData=None,**kwds):
    """
    ```
    ____________
    |   |  |   |
    ----|  |----
        |  |
        |  |
        |  |
        ----
    ```
    """
    if tf is None:
        tf = 2*(d/2-ymf)
    else:
        ymf = (d-tf)/2
    Yref = -yref
    Y  = [ Yref,  ymf + Yref, ymf + Yref]
    
    DY = [d, tf, tf]
    DZ = [ tw , (bf-tw)/2, (bf-tw)/2]
    
    SectData = Composite_Section(Y,DY,DZ,quad, MatData=MatData)
    
    return SectData

def W_Sect(b, d, alpha, beta, quadf, quadw, yref=0.0,MatData=None):
    nip = [nIPf,nIPw]
    nIP = sum(nip)
    u,du = onp.empty((2,1),dtype = None)
    u, du = quad_points(nIPf)
    DU = sum(du)
    
    Yref = -yref
    Y  = [ d-tf/2 + Yref, 
          (d-tf)/2 + Yref]
    
    DY = (tf, d-2*tf, tf)
    
    yi = [float(   Y[i] + DY[i]/DU * u[j]    )\
            for i in range(2) for j in range(nip[i])]

    dy = [float(    DY[i]/DU * du[j]      )\
            for i in range(2) for j in range(nip[i])]

    dz = [bf] * nip[0] + [tw]*nip[1]

    yi, dy, dz = map(list, zip(*sorted(zip(yi, dy, dz))))

    dA = [ dy[i]*dz[i] for i in range(nIP)]

    SectData = {'nIP': nIP,'dA': dA,'yi': yi}
    
    return SectData

def load_aisc(SectionName, props=''):
    """Load cross section properties from AISC database.
    
    props:
        A list of AISC properties, or one of the following:
        - 'simple': `A`, `Ix`, `Zx`
    
    """
    from . import aisc
    SectData = aisc.imperial[SectionName.upper()]
    if props == 'simple':
        props = ''
        return
    elif props:
        props = props.replace(' ','').split(',')
        sectData = {k:v for k,v in SectData.items() if k in props}
        if 'I' in props:
            sectData.update({'I': SectData['Ix']})
        return sectData

    return SectData
