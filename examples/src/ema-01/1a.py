def Model(
    ElemSpace={
        'g1': lambda e, *a: e(a),
        'g2': lambda e, *a: e(a), 
    },
    CON= {
        'a': ['g1', ['n1','n2']]
    }
) -> list: ...
