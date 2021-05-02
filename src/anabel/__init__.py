__version__ = "0.0.0"


import anabel.aisc
#import anabel.analysis

from anabel.assemblers import *
from anabel.matrices import *

from anabel.graphics import *
#from anabel.solvers import *

#import anabel.utilities
#import anabel.matlib

__all__ = anabel.objects.__all__ + ["elements", "objects"]

settings = {
    "DATAFRAME_LATEX": True,
}
