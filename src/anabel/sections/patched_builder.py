
import anabel.backend as anp
from anabel.abstract import ModelComponent, SectionPatch, SectionFiber, SectionLayer
from .patches import RectangularPatch

__all__ = ["PatchedSection"]


class PatchedSection(SectionPatch):
    def __init__(self, *args, material=None, origin=None, **kwds):
        ModelComponent.__init__(self, **kwds)
        if origin is None:
            origin = anp.zeros(2)
        self.origin = origin
        self._material = material
        self.patches  = list(filter(lambda x: isinstance(x,SectionPatch), args))
        self.fibers   = list(filter(lambda x: isinstance(x,SectionFiber), args))
        self.layers   = list(filter(lambda x: isinstance(x,SectionLayer), args))

    @property
    def area(self):
        return sum(i.area for i in self.patches)
    
    @property
    def centroid(self):
        return sum(i.centroid * i.area for i in self.patches) / self.area

    @property
    def Ix(self):
        centroid = self.centroid[1]
        return sum(p.Ix + p.area * (p.centroid[1] - centroid)**2  for p in self.patches)

    @property
    def Iy(self):
        centroid = self.centroid[0]
        return sum(p.Iy + p.area * (p.centroid[0] - centroid)**2  for p in self.patches)

    def shift(self, shift):
        pass

    def dump_opensees(self, indent=4, depth=0, **kwds):
        tag = self.tag if self.tag else "$secTag"
        tab = ' ' * indent * (depth + 1)
        children = tab + f"\n{tab}".join((
                *map(lambda x: x.dump_opensees(), self.patches),
                *map(lambda x: x.dump_opensees(), self.fibers),
                *map(lambda x: x.dump_opensees(), self.layers)
        ))
        return f"""section fiberSec {tag} {{
{children}
}}
""".replace("\n\n","\n")

    def rectangle(self, coord_ll, coord_ur, **kwds):
        """Add a rectangular patch to the section"""
        patch = RectangularPatch(coord_ll, coord_ur, parent=self, **kwds)
        self.patches.append(patch)
        return patch

    rect = rectangle

    def fiber(self, area):
        pass

    def ring(self):
        pass

    def circ(self):
        pass

    def patch(self, patch):
        self.patches.append(patch)
        return patch



    def to_mpl(self, **kwds):
        import matplotlib.collections
        return matplotlib.collections.PatchCollection(
            [p.to_mpl() for p in self.patches],
            facecolor="grey",
            edgecolor="grey",
            alpha=0.3
        )
    def properties(self):
        return {
            "A": self.area,
            "I_{xx}": self.Ix,
            "I_{yy}": self.Iy,
            "y_c": self.centroid[1]
            #"y_{pna}": self.y_plastic
        }

    def plot(self,
            show_properties=True,
            plain=False,
            show_quad=True,
            show_dims=True,
            annotate=True,
            ax = None,
            fig = None,
            **kwds
        ):
        """Plot a composite cross section.
        """    
        import matplotlib.pyplot as plt
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

        if ax is None:
            fig, ax = plt.subplots()
        ax.set_autoscale_on(True)
        ax.set_aspect(1)

        x_max = 1.01 * max(v.vertices[2][0] for v in self.patches)
        
        y_max = 1.05 * max(v.vertices[2][1] for v in self.patches)
        y_min = 1.05 * min(v.vertices[0][1] for v in self.patches)

        ax.axis("off")
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim( y_min, y_max)
        # add shapes
        ax.add_collection(self.to_mpl(**kwds))
        # show centroid
        ax.scatter(*self.centroid)
        # show origin
        ax.scatter(0, 0)
        
        return fig, ax


