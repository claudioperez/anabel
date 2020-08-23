
from anabel.config import _BACKEND 

backend = _BACKEND['ops']

if 'jax' in backend:
    from jax.numpy import *

elif 'numpy' in backend:
    from numpy import *

