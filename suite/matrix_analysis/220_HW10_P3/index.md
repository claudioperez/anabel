```python
import ema as em
import matplotlib.pyplot as plt
import numpy as np
%config InlineBackend.figure_format = 'svg'
```

# Problem 3


```python
mdl = em.rModel(2,3)
mdl.material('default', 1000)
mdl.xsection('default', 10000000, 50)
tsec = mdl.xsection('truss', 20, 1)

n1 = mdl.node('1', 0.0, 0.0)
n2 = mdl.node('2', 8.0, 0.0)
n3 = mdl.node('3', 16., 0.0)
n4 = mdl.node('4', 0.0, 6.0)

a = mdl.beam('a', n1, n2)
b = mdl.beam('b', n2, n3)
c = mdl.beam('c', n1, n4)

d = mdl.truss('d', n2, n4, xsec=tsec)

n2.p['y'] = -20

mdl.hinge(c, n4)

mdl.roller(n1)
mdl.fix(n3, ['x', 'y', 'rz'])
mdl.fix(n4, ['rz'])

# mdl.numDOF()
mdl.DOF = [[5, 6, 1], [5, 2, 3], [5, 7, 8], [4, 6, 9]] # Manually number DOFs
```


```python
# em.utilities.export.FEDEAS(mdl)
```


```python
mdl.nt
```




    9




```python
fig, ax = plt.subplots()
em.plot_structure(mdl, ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2094a461a88>




![svg](output_5_1.svg)



```python
K = em.K_matrix(mdl)
K.f
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$1$</th>
      <th>$2$</th>
      <th>$3$</th>
      <th>$4$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$P_{1}$</th>
      <td>50000.00000</td>
      <td>-4687.50</td>
      <td>12500.0</td>
      <td>4166.66667</td>
    </tr>
    <tr>
      <th>$P_{2}$</th>
      <td>-4687.50000</td>
      <td>3063.75</td>
      <td>0.0</td>
      <td>960.00000</td>
    </tr>
    <tr>
      <th>$P_{3}$</th>
      <td>12500.00000</td>
      <td>0.00</td>
      <td>50000.0</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>$P_{4}$</th>
      <td>4166.66667</td>
      <td>960.00</td>
      <td>0.0</td>
      <td>1974.44444</td>
    </tr>
  </tbody>
</table>




```python
Uf = em.analysis.SolveDispl(mdl)
Uf
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$U_{{}}$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$1$</th>
      <td>-0.002494</td>
    </tr>
    <tr>
      <th>$2$</th>
      <td>-0.014149</td>
    </tr>
    <tr>
      <th>$3$</th>
      <td>0.000624</td>
    </tr>
    <tr>
      <th>$4$</th>
      <td>0.012143</td>
    </tr>
  </tbody>
</table>



### Compatibility


```python
A = em.A_matrix(mdl)
V = A.f@Uf
V
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$V_{{}}$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$a_1$</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>$a_2$</th>
      <td>-0.000726</td>
    </tr>
    <tr>
      <th>$a_3$</th>
      <td>0.002392</td>
    </tr>
    <tr>
      <th>$b_1$</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>$b_2$</th>
      <td>-0.001145</td>
    </tr>
    <tr>
      <th>$b_3$</th>
      <td>-0.001769</td>
    </tr>
    <tr>
      <th>$c_1$</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>$c_2$</th>
      <td>-0.000470</td>
    </tr>
    <tr>
      <th>$c_3$</th>
      <td>0.002024</td>
    </tr>
    <tr>
      <th>$d_1$</th>
      <td>-0.001225</td>
    </tr>
  </tbody>
</table>




```python
mdl.redundant(b, '3')
mdl.redundant(d, '1')
B = em.B_matrix(mdl)
```


```python
np.around(B.barx,5)
```


    ---------------------------------------------------------------------------

    LinAlgError                               Traceback (most recent call last)

    <ipython-input-10-5ef0eae70dd5> in <module>
    ----> 1 np.around(B.barx,5)
    

    ~\OneDrive\400_box\Python\myPackages\ema\matrices.py in barx(self)
        550         nx = len(self.model.redundants)
        551 
    --> 552         Bbarxi = self.barxi
        553 
        554         Bbarx = Structural_Matrix(np.zeros((nQ,nx)))
    

    ~\OneDrive\400_box\Python\myPackages\ema\matrices.py in barxi(self)
        541     def barxi(self):
        542         Bx = self.f.x
    --> 543         Bbarxi = self.bari @ -Bx
        544         Bbarxi.column_data = Bx.column_data
        545         return Bbarxi
    

    ~\OneDrive\400_box\Python\myPackages\ema\matrices.py in bari(self)
        568     @property
        569     def bari(self):
    --> 570         return self.i.del_zeros().inv
        571 
        572     @property
    

    ~\OneDrive\400_box\Python\myPackages\ema\matrices.py in inv(self)
        212     @property
        213     def inv(self):
    --> 214         mat = np.linalg.inv(self)
        215         transfer_vars(self, mat)
        216         mat.row_data = self.column_data
    

    <__array_function__ internals> in inv(*args, **kwargs)
    

    ~\Anaconda3\lib\site-packages\numpy\linalg\linalg.py in inv(a)
        544     a, wrap = _makearray(a)
        545     _assertRankAtLeast2(a)
    --> 546     _assertNdSquareness(a)
        547     t, result_t = _commonType(a)
        548 
    

    ~\Anaconda3\lib\site-packages\numpy\linalg\linalg.py in _assertNdSquareness(*arrays)
        211         m, n = a.shape[-2:]
        212         if m != n:
    --> 213             raise LinAlgError('Last 2 dimensions of the array must be square')
        214 
        215 def _assertFinite(*arrays):
    

    LinAlgError: Last 2 dimensions of the array must be square



```python
B.barx
```


```python
ker = B.f.c.ker
np.around(ker,4)
```


```python
B.barx.T@V
```


```python
K.f@Uf
```

## Find element forces


```python
Q = K.s@V

Q
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-11-b08c858beddb> in <module>
    ----> 1 Q = K.s@V
          2 
          3 Q
    

    AttributeError: 'Stiffness_matrix' object has no attribute 's'



```python
em.plot_U(mdl, Uf, ax, scale=100, chords=False)
```


![svg](output_18_0.svg)



```python
x = np.linspace(0, c.L, 100)
v_tags = [c.tag+'_2', c.tag+'_3']
v = [V.get(v_tags[0]),V.get(v_tags[1])]
y = c.Elastic_curve(x, v, scale=1000, global_coord=True)
plt.plot(y[0], y[1])

```




    [<matplotlib.lines.Line2D at 0x2094b8c7d08>]




![svg](output_19_1.svg)



```python
Uf.row_data
```


```python

```
