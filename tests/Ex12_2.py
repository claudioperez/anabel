import emme as em 

mdl = em.Model(2,3)

n1 = mdl.node('1',  0.0,  0.0)
n2 = mdl.node('2',  0.0,  6.0)
n3 = mdl.node('3',  4.0,  6.0)
n4 = mdl.node('4',  8.0,  6.0)

a = mdl.beam('a', n1,  n2)
b = mdl.beam('b', n2,  n3)
c = mdl.beam('c', n3,  n4)
d = mdl.truss('d', n1,  n4)

mdl.fix(n1, ['x', 'y', 'rz'])
mdl.fix(n4, ['y'])

n2.p['x'] =  20
n3.p['y'] = -20

mdl.numDOF()

# Define plastic capacity
Qp_c = 160
Qp_g = 120
a.Qp['+']['1'] = a.Qp['-']['1'] = 1000
a.Qp['+']['2'] = a.Qp['-']['2'] = Qp_c
a.Qp['+']['3'] = a.Qp['-']['3'] = Qp_c

d.Qp['+']['1'] = d.Qp['-']['1'] = 50

b.Qp['+']['1'] = b.Qp['-']['1'] = 1000
b.Qp['+']['2'] = b.Qp['-']['2'] = Qp_g
b.Qp['+']['3'] = b.Qp['-']['3'] = Qp_g
c.Qp['+']['1'] = c.Qp['-']['1'] = 1000
c.Qp['+']['2'] = c.Qp['-']['2'] = Qp_g
c.Qp['+']['3'] = c.Qp['-']['3'] = Qp_g


lambdac, Q = em.analysis.PlasticAnalysis_wLBT(mdl)

assert abs(lambdac - 3.2) < 1e-4