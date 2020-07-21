import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp

from jaxlib import xla_client

import anabel
from anabel import models

nvars = 4
nf = 9
nr = 3

model_tree = anabel.load('input.yml')

elements = {
    'model' :  models.Model['basic-linear'], 
    'truss' :  models.DiffTruss}

model = anabel.compose(elements, model_tree)

# params = model_dict['params']
u = onp.zeros(nf+nr,dtype='float32')

Kf = emx.util.autodiff.fwd_stiff(model.f, nf)

@jax.jit
def f(params):
    E, P, Ao, Au = params[[0,1,2,3]]
    kwds = {
    "params":{
      "e1": {"A" : Ao, "E": E},
      "e2": {"A" : Ao, "E": E},
      "e3": {"A" : Ao, "E": E},
      "e4": {"A" : Au, "E": E},
      "e5": {"A" : Au, "E": E},
      "e6": {"A" : Au, "E": E},
      "e7": {"A" : Ao, "E": E},
      "e8": {"A" : Ao, "E": E},
      "e9": {"A" : Ao, "E": E}}}

    kf = Kf(u,**kwds)
    p = jnp.array([0., P, 0., P, 0., 0., 0., 0., 0.], dtype='float32')[:,None] 
    U = jnp.linalg.solve(kf, p)
    return U[[1,3], [0,0]]

param_init = jnp.zeros(nvars, dtype='float32')

f_xla = jax.xla_computation(f)

fxla = xla_client.XlaComputation(f_xla(param_init).as_serialized_hlo_module_proto())

with open('model.pb','wb') as f:
    f.write( fxla.as_serialized_hlo_module_proto ( ) )
