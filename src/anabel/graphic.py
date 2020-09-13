import matplotlib.pyplot as plt


def plot(con,xyz,ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for edge in con.values():
        XY = [[x[0],x[1]] for n,x in xyz.items() if n in edge[1]]
        ax.plot(*zip(*XY))