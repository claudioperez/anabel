import ema as em
import matplotlib.pyplot as plt
import numpy as np

mdl = em.rModel(2,3)
n = mdl.dnodes
e = mdl.delems
w = 20
h = 10

mdl.frame((1,w), (2,h))


n['4'].mass = 10
n['6'].mass =  5

mdl.fix(n['1'], ['x', 'y'])
mdl.fix(n['2'], ['x', 'y'])

    
mdl.DOF = [[ 8, 9, 10], [11, 12, 13], 
           [ 1, 9,  4], [ 1, 12,  5], 
           [ 2, 9,  6], [ 2, 12,  7]]