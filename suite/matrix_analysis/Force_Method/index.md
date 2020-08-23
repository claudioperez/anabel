---
title: The Force Method
---

# The Force Method

```python
import ema as em
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
%config InlineBackend.figure_format = 'svg'
```


```python
dm = em.Model(2,2) # create instance of model object
n = dm.dnodes 
e = dm.delems

A1 = 10000
Ac = 20000
I = 1
dm.xsection('default', A1, I)
csec = dm.xsection('section-c', Ac, I)

n1 = dm.node('1', 0.0, 0.0)
n2 = dm.node('2', 16., 0.0)
n3 = dm.node('3', 8.0, 6.0)
n4 = dm.node('4', 0.0, 6.0)

a = dm.truss('a', n3, n4) # add truss element to model object
b = dm.truss('b', n1, n3)
c = dm.truss('c', n2, n3, xsec=csec)

dm.pin(n1)
dm.pin(n4)
dm.pin(n2)

dm.numDOF();
```


```python
prim = em.Model(2,2) # create instance of model object
np = prim.dnodes 
ep = prim.delems

A1 = 10000
Ac = 20000
I = 1
prim.xsection('default', A1, I)
csec = prim.xsection('section-c', Ac, I)

prim.node('1', 0.0, 0.0)
prim.node('2', 16., 0.0)
prim.node('3', 8.0, 6.0)
prim.node('4', 0.0, 6.0)

prim.truss('a', np['3'], np['4']) # add truss element to model object
prim.truss('c', np['2'], np['3'], xsec=csec)

prim.pin(np['1'])
prim.pin(np['4'])
prim.pin(np['2'])

prim.numDOF(); # Automatically number model dofs
```


```python
# Establish redundant member force
dm.redundant(b, '1')
```


```python
fig, ax = plt.subplots()
em.plot_structure(dm, ax)
```


![svg](output_5_0.svg)


## Part 1 : Nodal Loading


```python
A = em.A_matrix(dm)
```


```python
np['3'].p['y'] = 30
np['3'].p['x'] = 50

Up = em.analysis.SolveDispl(prim)
A.f@Up
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$V_{{fffff}}$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$a_1$</th>
      <td>0.0720</td>
    </tr>
    <tr>
      <th>$b_1$</th>
      <td>0.1402</td>
    </tr>
    <tr>
      <th>$c_1$</th>
      <td>0.0250</td>
    </tr>
  </tbody>
</table>



## $U_x$


```python
b.q0['1'] = 1
np['3'].p['y'] = 0.6
np['3'].p['x'] = 0.8
Ux = em.analysis.SolveDispl(prim)
V0 = em.V0_vector(dm)
em.plot_U(dm, Ux, ax, scale=500)
Ux
```

    [0.0, 0.0]
    [0.0, 0.0]
    [0.00128, 0.0025399999999999997]
    [0.0, 0.0]
    




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$U_{{fffffffffffffffff}}$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$1$</th>
      <td>0.00128</td>
    </tr>
    <tr>
      <th>$2$</th>
      <td>0.00254</td>
    </tr>
  </tbody>
</table>




![svg](output_10_2.svg)



```python
A.f@Ux + V0
```

    super
    

    C:\Users\claud\Anaconda3\lib\site-packages\IPython\core\formatters.py:371: FormatterWarning: text/html formatter returned invalid type <class 'ema.matvecs.Deformation_vector'> (expected <class 'str'>) for object: Deformation_vector([0.00128 , 0.003548, 0.0005  ])
      FormatterWarning
    




    Deformation_vector([0.00128 , 0.003548, 0.0005  ])




```python
# Define nodal loading
b.q0['1'] = 0
n['3'].p['y'] = 30
n['3'].p['x'] = 50
UP = em.analysis.SolveDispl(dm)
UP
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$U_{{fffffffff}}$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$1$</th>
      <td>0.021421</td>
    </tr>
    <tr>
      <th>$2$</th>
      <td>0.037298</td>
    </tr>
  </tbody>
</table>




```python
A.f@UP
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$V_{{fffffffff}}$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$a_1$</th>
      <td>0.021421</td>
    </tr>
    <tr>
      <th>$b_1$</th>
      <td>0.039515</td>
    </tr>
    <tr>
      <th>$c_1$</th>
      <td>0.005242</td>
    </tr>
  </tbody>
</table>


