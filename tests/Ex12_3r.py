import emme as em

mdl = em.rModel(2,3)

n1 = mdl.node('1',  0.0,  0.0)
n2 = mdl.node('2',  0.0,  5.0)
n3 = mdl.node('3',  4.0,  5.0)
n4 = mdl.node('4',  8.0,  5.0)
n5 = mdl.node('5',  8.0,  0.0)

a = mdl.beam('a', n1,  n2)
b = mdl.beam('b', n2,  n3)
c = mdl.beam('c', n3,  n4)
d = mdl.beam('c', n4,  n5)

mdl.fix(n1, ['x', 'y', 'rz'])
mdl.fix(n5, ['x', 'y', 'rz'])

n2.p['x'] = 30
n3.p['y'] = -50

mdl.DOF = [[6, 7, 8],[1, 7, 2], [1, 3, 4], [1, 9, 5], [10, 9, 11]]

m, s = em.analysis.characterize(mdl)
assert m == 0
assert s == 3


# Define plastic capacity
Qp_c = 150
Qp_g = 120
a.Qp['+']['1'] = a.Qp['-']['1'] = 10000
a.Qp['+']['2'] = a.Qp['-']['2'] = Qp_c
a.Qp['+']['3'] = a.Qp['-']['3'] = Qp_c
d.Qp['+']['1'] = d.Qp['-']['1'] = 10000
d.Qp['+']['2'] = d.Qp['-']['2'] = Qp_c
d.Qp['+']['3'] = d.Qp['-']['3'] = Qp_c

b.Qp['+']['1'] = b.Qp['-']['1'] = 10000
b.Qp['+']['2'] = b.Qp['-']['2'] = Qp_g
b.Qp['+']['3'] = b.Qp['-']['3'] = Qp_g
c.Qp['+']['1'] = c.Qp['-']['1'] = 10000
c.Qp['+']['2'] = c.Qp['-']['2'] = Qp_g
c.Qp['+']['3'] = c.Qp['-']['3'] = Qp_g

lambdac, Q = em.analysis.PlasticAnalysis_wLBT(mdl)
assert abs(lambdac - 2.2286) < 1e-3