__version__ = "0.0.8"

try:
    import anabel.aisc
    import anabel.sections
    import anabel.analysis
except:
    pass

from anabel.matrices import *
from anabel.assemble import *

from anabel.graphics import *
#from anabel.solvers import *

#import anabel.utilities
#import anabel.matlib

#__all__ = anabel.assemblers.__all__ + ["elements", "assemblers", "graphics"]

settings = {
    "DATAFRAME_LATEX": True,
}
