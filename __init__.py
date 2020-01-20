"""
Earthquake Matrix Analysis. (:mod:`ema`)
=====================================================

.. currentmodule:: ema


"""


# __all__ = ['a', 'b', 'c']
__version__ = '0.0.2'
__author__ = None


import ema.analysis

from ema.objects import *
# from ema.matvecs import *
from ema.matrices import *

from ema.graphics import *
from ema.solvers import *

import ema.utilities
# from ema.utilities.fedeas.syntax import *
from ema.tests import *


settings = {
    "DATAFRAME_LATEX": True,

}


