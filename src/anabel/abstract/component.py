from elle.units import UnitHandler

def build_string(strng:str, **kwds)->str:
    return strng

class ModelComponent:
    _color: str

    def __init__(self,
            domain = None,
            notes  = None,
            units  = None,
            author = None,
            color  = None,
            parent = None,
            tag    = None
    ):
        self._domain = domain
        self._parent = parent
        self._color = color
        self._author = author
        self._notes = notes
        self._units = UnitHandler(units) if isinstance(units,str) else units

    def dump_opensees(self, **kwds):
        return build_string("""
        >>> Unimplemented translation.
        """, **kwds)

    @property
    def domain(self):
        if hasattr(self,"_domain"):
            return self._domain
        else:
            raise AttributeError("No attribute domain")
    
    @property
    def parent(self):
        if hasattr(self,"_parent") and self._parent is not None:
            return self._parent
        elif self.domain:
            return self.domain
        else:
            return None

    @property
    def tag(self):
        if hasattr(self,"_tag") and self._tag is not None:
            return self._tag
        elif hasattr(self,"_name") and isinstance(self._name, int):
            return self._name
        elif self.parent:
            return self.parent.index(self)
        else:
            return None

    @property
    def color(self):
        if hasattr(self,"_color"):
            if self._color is not None:
                return self._color
            else:
                return self.parent._color
        else:
            raise AttributeError("No attribute color")
    
    @property
    def units(self)->UnitHandler:
        if hasattr(self,"_units"):
            if isinstance(self._units,(int,str)):
                return self.parent.units[self._units]
            else:
                return self._units
        else:
            raise AttributeError("No attribute units")

    @property
    def material(self):
        if hasattr(self,"_material"):
            if isinstance(self._material,(int,str)):
                return self.domain.materials[self._material]
            else:
                return self._material
        else:
            raise AttributeError("No attribute material")


