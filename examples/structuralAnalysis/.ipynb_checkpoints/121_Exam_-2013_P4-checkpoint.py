


######################################################################
# Problem taken from 2019 lecture notes, Lecture 12
# 

dm = em.Domain(2,3)
e = dm.delems
n = dm.dnodes

dm.node('1',  0.0, 0.0)
dm.node('2',  4.0, 0.0)
dm.node('3',  8.0, 0.0)
dm.node('4',  0.0, 3.0)
dm.node('5', -4.0, 0.0)


dm.beam('a', n['1'], n['2'])
dm.beam('b', n['2'], n['3'])
dm.beam('c', n['1'], n['4'])
dm.truss('d', n['4'], n['5'])


dm.fix(n['1'], ['y'])
dm.fix(n['3'], ['x', 'y', 'rz'])
dm.fix(n['5'], ['x', 'y', 'rz'])

n['2'].p['y'] = -60
n['4'].p['x'] = -30
dm.numDOF()

fig, ax = plt.subplots()
em.plot_structure(dm, ax)

# V.set_item('')