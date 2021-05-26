__version__ = "0.0.10"

try:
    import anabel.aisc
    import anabel.analysis
except:
    pass
import anabel.sections

from . import transient as transient
from . import autodiff as autodiff
from anabel.matrices import *
from anabel.assemble import *

from anabel.graphics import *
#from anabel.solvers import *

#import anabel.utilities
#import anabel.matlib

#__all__ = anabel.assemblers.__all__ + ["elements", "assemblers", "graphics"]

def load(filename):
    pass

settings = {
    "DATAFRAME_LATEX": True,
}
