from anabel.abstract import FrameSection
__all__ = ["CompositeBuilder"]

class CompositeSection(FrameSection):
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
        import matplotlib.patches as patches

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

class CompositeBuilder:
    pass

