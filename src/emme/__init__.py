
__version__ = "0.0.0"



import emme.analysis

from emme.objects import *
from emme.matrices import *

from emme.graphics import *
from emme.solvers import *

import emme.utilities
import emme.matlib

__all__ = emme.objects.__all__ + ["elements", "objects"]

settings = {
    "DATAFRAME_LATEX": True,
}


