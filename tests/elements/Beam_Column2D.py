import anabel as em

E = 1000
At = 30
A = 30000000
I = 100

mdl = em.rModel(2,3)
n = mdl.dnodes
e = mdl.delems

mdl.material('default', E)
mdl.xsection('default', A, I)
truss_section = mdl.xsection('truss', At, I)

mdl.node('1',  0.0, 0.0)
mdl.node('2',  8.0, 0.0)
mdl.node('3',  8.0, 6.0)
mdl.node('4', 16.0, 6.0)

#mdl.beam_column2D('a', n['1'], n['2'])
mdl.beam('a', n['1'], n['2'])
mdl.beam('b', n['2'], n['3'])
mdl.beam('c', n['3'], n['4'])
mdl.truss('d', n['2'], n['4'], xsec=truss_section)

mdl.hinge(e['c'], n['4'])

mdl.fix(n['1'], ['x', 'y', 'rz'])
mdl.fix(n['4'], ['y', 'rz'])

mdl.DOF = [[5, 6, 7], [5, 1, 2], [3, 1, 4], [3, 8, 9]]


e['c'].w['y'] = -5
Uf = em.analysis.SolveDispl(mdl)
#Uf = mdl.compose()()

true_Uf = [-0.00855762, -0.00064547, -0.00104513,  0.00037486]

for true, test in zip(true_Uf, Uf):
    assert abs(true-test) < 1e-8


