
_BACKEND = {
    'ops': 'jax',
    'numeric': 'jax'
}


def use(backend, **kwds):
    _BACKEND['ops'] = backend
    _BACKEND['numeric'] = backend





