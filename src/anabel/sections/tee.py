from anabel.abstract import CompositeSection, VerticalSection
from .patches import RectangularPatch

class Tee(PatchedSection, VerticalSection):
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
        #self.area = area = bf*tf + (d - tf) *tw

        # distance from top edge to centroid
        self.y_centroid = y_c = 1/(2*area) * (tw*d**2 + (bf-tw)*tf**2)
        self.y_plastic = d - area/(2*tw) if tf < area/(2.0*bf) else area/(2.0*bf)
        # Moment of inertia
        #self.moi = bf*tf**3/12.0 + (bf*tf)*(tf*0.5*y_c)**2 + tw*(d-tf)**3/12 + tw*(d-tf)*((d+tf)*0.5-y_c)**2
        dw = d - tf
        self.moi =  tw*(dw+tf)**3/3 + (bf - tw)*tf**3/3 - area*y_c**2

        if yref is None:
            yref = (d-tf)/2 + tf - self.y_centroid
            #yref = 0.0

        Yref = -yref
        Y  = [ Yref,  (d-tf)/2 + tf/2 + Yref]

        flange = RectangularPatch([-bf/2.0, (d-tf)/2 + tf/2 + Yref])
        
        
        #super().__init__(Y,DY,DZ,quad,mat=mat)
    
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


