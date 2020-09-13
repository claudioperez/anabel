"""
A collection of functions for composing graphs of finite elements.
"""

import inspect
from collections import namedtuple, deque
from collections import MutableMapping
from .interfaces import interface

from jax import jit


# from anabel.models import _linear_wire, model

def compose(elements, tree, key=None, _built=False, compose_node='ElemSpace'):
    """Apply function composition over a computational graph."""
    node = compose_node
    if key is None: 
        key = list(tree[node].keys())[0]
        tree = list(tree[node].values())[0]

    if node in tree and (not _built): # build grandchild elements
        tree[node] = {
            hkey: compose( elements, tree[node][hkey], key=hkey) 
            for hkey in [eli for eli in tree[node]] }

        return compose(elements, tree, key, True) # build child element

    else: return elements[key](**tree)

Create_Model = compose

def merge(d1, d2):
    """
    Merge two dicts of dicts recursively.
    If either mapping has leaves that are not instances 
    of a `MutableMapping`, the second's leaf overwrites 
    the first's.
    """
    for k, v in d1.items():
        if k in d2:
            if all(isinstance(e, MutableMapping) for e in (v, d2[k])):
                d2[k] = merge(v, d2[k])
    d3 = d1.copy()
    d3.update(d2)
    return d3



def dict_depth(d):
    queue = deque([(id(d), d, 1)])
    memo = set()
    while queue:
        id_, o, level = queue.popleft()
        if id_ in memo: continue
        memo.add(id_)
        if isinstance(o, dict):
            queue += ((id(v), v, level + 1) for v in o.values())
    return level

Element = namedtuple('Element', 'f, Df, params')

# def state_wrapper(element):
#     return Element(
#         lambda **kwds: (element.f(**kwds), None),
#         element.Df ,
#         element.state
#     )

# Model = {
#     'basic-linear': _linear_wire
# }

###############################################################
# Combinators
###############################################################
def Z(f): 
    """Z-combinator
    $$\lambda f .(\lambda x .f(\lambda v .((x x) v)))(\lambda x . f(\lambda v .((x x) v)))$$"""
    return (lambda x: f(lambda v: x(x)(v)))(lambda x: f(lambda v:x(x)(v)))


def step (f):
    def incr (originalValue, approx):
        improvedApprox = (approx + (originalValue / approx)) / 2;

        # How far off the mark we are.
        discrepancy = abs(originalValue -  improvedApprox * improvedApprox)

        # Termination condition
        if discrepancy < 0.00001:
            return improvedApprox
        return f(originalValue, improvedApprox)
    return incr


