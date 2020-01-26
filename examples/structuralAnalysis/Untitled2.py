"""
AIN_MATRIX
==========

Syntax
------

-  AUB = AUB_MATRIX (MODEL,ELEMDATA)
-  AUB = AUB_MATRIX (MODEL,ELEMDATA,ALPH)

the function sets up the block diagonal matrix of element limit-surface
equations, :math:`A_{in}`, for the structural model specified in data
structure MODEL with element property information in cell array
ELEMDATA. Options for parameter ALPH are specified below.

Parameters
~~~~~~~~~~

--------------

-  | ``Model``: struct
   | Contains structure data.

-  | ``ElemData``: cell array
   | Contains element property data.

-  ``alph``: (Optional) Char array or float array.

   **Array.** If an :math:`n \times 2` array is passed, :math:`n`
   piecewise linear axial-moment interaction equations of the following
   form will be generated for each element:

   .. math:: a_n \frac{|N|}{N_{pl}} + b_n\left( \frac{|M_z|}{M_{p,z}} + \frac{|M_y|}{M_{p,y}}\right)  \leq 1.0

   If an :math:`n \times 3` array is passed, :math:`n` piecewise linear
   axial-moment-shear interaction equations of the following form will
   be applied:

   .. math:: a_n \frac{|N|}{N_{p}} + b_n \frac{|M|}{M_{p}} + c_n \frac{|V|}{V_{p}} \leq 1.0

   **String.** Alternatively, a ``char array`` may be passed indicating
   one of the following options:

   -  ``'AISC-H2'`` (Default)
   -  ``'AISC-H1'``

   **Empty.** If no parameter is passed in the third position, the
   function will go to the ``NMOpt`` field of the cell ``ElemData`` for
   each element, which may also contain a string or an array. Elements
   with no such field will default to the ``AISC-H2`` option.

Formulation
-----------

The matrix :math:`A_{in}` forms the upper bound plastic conditions in
the form

.. math:: \mathbf{A}_{in} \mathbf{Q} \leq \mathbf{1}

for the lower bound linear programming problem expressed as follows:

.. math::

   \begin{aligned}
   \lambda_{c}=\max \lambda & \\
   \text { with } &\left\{\begin{aligned}
   \lambda P_{r e f}+P_{c f} &=\mathbf{B}_{f} Q \\
   \mathbf{A}_{in} \mathbf{Q} & \leq \mathbf{1} \\
   \end{aligned}\right.
   \end{aligned}

### 2D Shear-moment-axial interaction
-------------------------------------

Interaction between shear, moment, and axial forces is implemented with
:math:`n` piecewise linear equations of the following form:

.. math:: a_n\dfrac{|N|}{N_p} + b_n\dfrac{|M|}{M_p} + c_n\dfrac{|V|}{V_p} \leq1.0

:math:`N`, :math:`M`, and :math:`V` are substituted by :math:`Q_1`,
:math:`Q_2`, and :math:`(Q_2 +Q_3)/L` respectively, and the equations
are rearranged and implemented follows:

.. math:: \dfrac{a}{N_p}Q_1 + \dfrac{b}{M_p}Q_2 + \dfrac{ c}{LV_p}(Q_2+Q_3) \leq1.0

.. math:: \dfrac{a}{N_p}Q_1 + \left(\dfrac{b}{M_p}+\dfrac{c}{LV_p}\right)Q_2 + \dfrac{c}{LV_p}Q_3 \leq1.0

"""

aii =  [ A/Qpl(1,1)   B/Qpl(2,2)+C/(L*Qpl(4,1))   C/(L*Qpl(4,1));
        -A/Qpl(1,2)   B/Qpl(2,2)+C/(L*Qpl(4,1))   C/(L*Qpl(4,1));
         A/Qpl(1,1)  -B/Qpl(2,1)+C/(L*Qpl(4,1))   C/(L*Qpl(4,1));
        -A/Qpl(1,2)  -B/Qpl(2,1)+C/(L*Qpl(4,1))   C/(L*Qpl(4,1));
         A/Qpl(1,1)   B/Qpl(2,2)-C/(L*Qpl(4,1))  -C/(L*Qpl(4,1));
        -A/Qpl(1,2)   B/Qpl(2,2)-C/(L*Qpl(4,1))  -C/(L*Qpl(4,1));
         A/Qpl(1,1)  -B/Qpl(2,1)-C/(L*Qpl(4,1))  -C/(L*Qpl(4,1));
        -A/Qpl(1,2)  -B/Qpl(2,1)-C/(L*Qpl(4,1))  -C/(L*Qpl(4,1))];     

