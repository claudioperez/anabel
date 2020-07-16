
import sys, os, json
# print('\n')
# print(os.path.dirname(sys.executable))

sys.path.append('/mnt/c/Users/claud/depot/servir/')
from ModelServer import ModelServer
from build import *
from params import *
from jaxlib import xla_client
# from ctypes import c_float
from array import array
import jax.numpy as np

with open('model.pb','rb') as file: c = xla_client.XlaComputation(file.read())

backend=xla_client.get_local_backend( )
C = backend.compile(c)

# param = np.array([E, P, Ao, Au], dtype='float32'   )
# out = xla_client.execute_with_python_values( C, ( param, ), backend=backend)


def response(args):
#     args = np.array( args, dtype='float32' )

    return xla_client.execute_with_python_values( C, ( args, ), backend=backend)[0].tolist()

if __name__ == "__main__":
    ModelServer(response, port=9999, size=64).serve()
    
