import ema as em

mdl = em.Model(2,2)

# Create nodes
n1 = mdl.node('1',  0,  0)
n2 = mdl.node('2',  0,  2)
n3 = mdl.node('3',  3,  2)
n4 = mdl.node('4',  3,  4)
n5 = mdl.node('5',  6,  4)
n6 = mdl.node('6',  9,  4)
n7 = mdl.node('7',  9,  2)
n8 = mdl.node('8', 12,  2)
n9 = mdl.node('9', 12,  0)

# Create elems
a = mdl.truss('a', n1, n2)
b = mdl.truss('b', n1, n3)
c = mdl.truss('c', n2, n3)
d = mdl.truss('d', n2, n4)
e = mdl.truss('e', n3, n4)
f = mdl.truss('f', n3, n5)
g = mdl.truss('g', n4, n5)
h = mdl.truss('h', n5, n6)
i = mdl.truss('i', n5, n7)
j = mdl.truss('j', n6, n7)
k = mdl.truss('k', n6, n8)
l = mdl.truss('l', n7, n8)
m = mdl.truss('m', n7, n9)
n = mdl.truss('n', n8, n9)

# Create reactions
mdl.pin(mdl.nodes[ 0])
mdl.pin(mdl.nodes[-1])

# number DOFs
mdl.numDOF()

# Load
n4.p['y']= -12
n8.p['x']= -18