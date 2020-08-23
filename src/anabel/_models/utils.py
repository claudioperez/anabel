

def number_dof_basic(mesh, bn, debug=False):
    """Basic dof numbering"""
    ndf = max(len(con) for con in mesh.values())
    nr = sum(sum(boun) for boun in bn.values())
    nodes = {node for con in mesh.values() for node in con[1]}
    nn = len(nodes)

    crxns = ndf*nn - nr + 1

    df = 1
    temp = {}

    for node in sorted(nodes):
        DOFs = []
        try:
            for rxn in bn[node]:
                if not rxn:
                    DOFs.append(df)
                    df += 1
                else:
                    DOFs.append(crxns)
                    crxns += 1
        except KeyError:
            df -= 1
            DOFs = [df := df + 1 for _ in range(ndf)]
            df += 1

        temp[node] = DOFs
    # print(locals())
    return temp
