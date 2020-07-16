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



# tensorlib = tensor.numpy_backend()
# default_backend = jax.numpy


# def get_backend():
#     """
#     Get the current backend and the associated optimizer
#     Example:
#         >>> import pyhf
#         >>> pyhf.get_backend()
#         (<pyhf.tensor.numpy_backend.numpy_backend object at 0x...>, <pyhf.optimize.opt_scipy.scipy_optimizer object at 0x...>)
#     Returns:
#         backend, optimizer
#     """
#     global tensorlib
#     global optimizer
#     return tensorlib


# @events.register('change_backend')
# def set_backend(tensor_library, compiler='jaxlib'):
#     """
#     Set the backend and the associated optimizer
#     Args:
#         tensor_library (`str`): One of the supported pyhf backends: numpy, jax
#         compiler       (`str`): Optional compiler 
#     Returns:
#         None
#     """
#     global tensor_library
#     global compiler

#     if isinstance(tensor_library, (str, bytes)):
#         if isinstance(tensor_library, bytes):
#             tensor_library = tensor_library.decode("utf-8")
#         tensor_library = tensor_library.lower()

#     # need to determine if the tensor_library changed 
#     tensorlib_changed = bool(
#         (tensor_library.name != tensorlib.name) | (tensor_library.precision != tensorlib.precision)
#     )
#     # set new backend
#     tensorlib = tensor_library
#     # trigger events
#     if tensorlib_changed:
#         events.trigger("tensorlib_changed")()
#     # set up any other globals for backend
#     tensorlib._setup()


from .core import *
from anabel import models
import io
# from anabel import numeric
# from emx.graphics import *
# from .fem import *
# import emx.analysis
# import emx.utilities
# from emx.tests import *
# from emx.tensorelements import *
