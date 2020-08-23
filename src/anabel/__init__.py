"""
=====================================================

.. currentmodule:: anabel

"""
# from .version import __version__
__version__ = '0.0.0'



try:
    from ._anabel import execute_model  # noqa
except ImportError:
    def execute_model(args):
        return max(args, key=len)

try:
    import jaxlib
except ImportError:
    COMPILER = None
else:
    COMPILER = jaxlib


from .core import *
# from anabel import models
import anabel.io
import anabel.autodiff
import anabel.numeric
import anabel.roots 

# import anabel.ops
