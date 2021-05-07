# Claudio Perez
# University of California, Berkeley
"""
# Templating

Wrappers, decorators and utilities for constructing expression templates.
"""
# Standard library imports
import inspect
from functools import wraps, partial
from collections import namedtuple,UserDict
from typing import Union, Tuple, Callable

# Optional imports
try:
    import jax
    from jax import jit
    from jaxlib import xla_client
except:
    jax = None
    jit=lambda x: x
    xla_client = None

import anon.diff
import anon.atom as anp
# import anabel.ops as anp
# import elle.iterate
from anon.core import interface

class Dual:
    pass


def template(
    dim:      Union[int,Tuple[Tuple[int,int]]]=None,
    statevar: str   = "state",
    main:     str   = "main",
    jacx:     str   = "jacx",
    form:     str   = "x,y,s=s,p=p,**e->x,y,s",
    params:   str   = "params",
    # paramvar: str   = "params",
    dimvar:   str   = None,
    origin:   tuple = None,
    order:    int   = 0,
    **kwargs
    # inv:Callable  = elle.iterate.inv_no1,
) -> Callable:
    """Decorator that wraps a basic local map generator.

    Attributes
    ----------
    `origin`:
        A structure containing arguments which may act as an 'origin' for the target function.

    `shape`: tuple
        Description of the shape of the dual and primal spaces.

    Generated arguments
    -------------------
    `_expose_closure`: bool
    :   Expose closed-over local variables as an attribute.

    `_jit`: bool
    :   If `True`{.python}, JIT-compiles the target function

    Studies
    -------
    [Structural dynamics (`elle-0008`)](/stdy/elle-0008)

    """


    def _decorator(func: Callable[...,dict]) -> Callable[...,tuple]:

        args:     str   = ",".join(["x","y","state"]+(["params"] if params else []))
        funcname = func.__name__.replace("<","_").replace(">","_") #+ ".domain"
        Arguments = namedtuple(funcname, args)

        @wraps(func)
        def wrapper(
            *args,
            _expose_closure=False,
            _jit=False,
            _reform:Union[str,bool]=False,
            _curry=False,
            _debug=False,
            _form = None,
            _iter=False,
            **kwds
        ) -> Callable:

            loc = func(*args, **kwds)
            if isinstance(order,str):
                _order = loc[order]
            else:
                _order = order

            if origin is None:
                assert dim is not None

                if isinstance(dim,str):
                    _dim = loc[dim]
                else:
                    _dim = dim

                if dim == 0:
                    xshape = yshape = 1

                elif isinstance(_dim,int):
                    xshape,yshape = ((_dim, 1), (_dim,1))
                else:
                    xshape, yshape = _dim

                    if xshape == 1:
                        pass
                    elif isinstance(xshape,int):
                        xshape = (xshape, 1)

                    if yshape == 1:
                        pass
                    elif isinstance(yshape,int):
                        yshape = (yshape, 1)

                shape = (xshape, yshape)
                if dim == 0:
                    x0, y0 = 0.0, 0.0
                else:
                    x0, y0 = anp.zeros(xshape), anp.zeros(yshape)
                    # shape = ((_order, *xshape), yshape)

                # _params = loc[params] if params else {}
                interface = anon.core.new_interface(
                    funcname+"_params",
                    loc[params] if params in loc else {} # get_unspecified_params()
                )

                # _params = interface(**loc[params])
                # _params = interface()
                
                _params = {}

                _origin = Arguments(*(
                    [x0, y0,loc[statevar],_params]
                ))

            else:
                # origin has been passed to generator
                if isinstance(origin,str):
                    _origin = loc[origin]
                else:
                    _origin = origin

                if isinstance(_origin[0], (float,int)):
                    xshape=1
                else:
                    xshape=_origin[0].shape

                if isinstance(_origin[1], (float,int)):
                    yshape=1
                else:
                    yshape=_origin[1].shape

                shape = xshape, yshape
                # TODO: refactor
                try:
                    _params = _origin[3]
                except:
                    _params = {}

                _params = anon.core.new_interface(
                    funcname+"_params",_params
                )

            if isinstance(main,str):
                _main = loc[main]
            else:
                _main = main



            #-transformations------------------
            if _reform is True:
                """
                assume main is of form `f: x->y`; reform to
                `f: (x,y,state,params,**kwds) -> (x,y,state)`
                """
                original_main = _main
                _main = lambda x, y=None,state={},params={},**kwds: (x, original_main(x), {})

            if _form == "<jax>":
                original_main = _main
                def _main(state, x):
                    # print(state, x)
                    out = original_main(x, None, state)
                    return out[2], out[1]


            if _jit:
                _main = jit(_main)
            #----------------------------------
            if _debug:
                def about():
                    print(f"order:  {_order}")
                    print(f"shape:  {shape}")
                    print(f"origin: {_origin}")
                _main.about = about
            if _expose_closure:
                _main.closure = loc
            #----------------------------------

            if "tanx" in loc: _main.tanx = loc["tanx"]
            _main.origin = _origin
            _main.params = _params
            _main.uparams = get_unspecified_parameters(_main)
            _main.shape  = shape
            _main.order  = _order
            if isinstance(jacx,str):
                if jacx in loc:
                    _jacx = loc["jacx"]
                elif jacx == "fwd":
                    _jacx = anon.diff.jacfwd(_main,1,0)
                else:
                    _jacx = anon.diff.jacfwd(_main,1,0)
            else:
                _jacx = jacx

            _main.jacx = jax.jit(_jacx) if _jit else _jacx


            return _main
            # ~ __invert__
        wrapper.part  = lambda **kwds: partial(wrapper, **kwds)
        return wrapper
    return _decorator

def wrap(f, *args, **kwds):
    """Wrap a pre-defined function to act as a local dual map.
    """
    state = kwds["state"] if "state" in kwds else {}
    params = kwds["params"] if "params" in kwds else {}
    return generator(
        *args, **kwds
    )(
        lambda *_, **__: {
            "main": f,
            "state": state,
            "params": params,
        }
    )(**kwds)

# def reform(func:Callable, form:str, newform:str):
#     old_args, old_out = form.split("->")

#     old_out = old_out.replace(" ","").split(",")


#     new_args, new_out = newform.split("->")
#     new_out = new_out.replace(" ","").split(",")

#     indices = ",".join(old_out.index[new] for new in new_out)

#     return exec(f"def reformed({}): return func({})[{indices}]")

def serialize(f):
    """
    >Requires JAX
    """
    return xla_client.XlaComputation(
        jax.xla_computation(f)(
            *f.origin
        ).as_serialized_hlo_module_proto()
    ).as_serialized_hlo_module_proto()


def assemble(f):
    backend = xla_client.get_local_backend()
    F = backend.compile(serialize(f))
    @wraps(f)
    def compiled_f(*args, **kwds):
        return xla_client.execute_with_python_values(
            F,(args, kwds), backend=backend
        )[0]
    return compiled_f



def generator_no2():
    """
    def F():
        name = "my-element"
        def main(x, y, state, **kwds, _name=name):
            pass
    """
    pass


def get_unspecified_parameters(func,recurse=False):
    """
    created 2021-03-31
    """
    signature = inspect.signature(func)
    params = {
        k: v
        for k, v in signature.parameters.items()
        if (v.default is None) or isinstance(v.default,inspect.Parameter) #not inspect.Parameter.empty
    }
    if recurse and "params" in signature.parameters:
        #print(signature.parameters)
        params.update({"params": {
                k: v
                for k,v in signature.parameters["params"].default.items()
                if (v is None) or isinstance(v,inspect.Parameter)
            }
        })
    return params

generator = template

