"""
Element Library. (:mod:`elemlib`)
=====================================================

.. currentmodule:: matrices

This module provides a collection of various elements.

"""

class LE2dFrm:
    """LE2dFRM 2d LE frame element under linear or nonlinear geometry
    response of 2d linear elastic frame element;
    the element accounts for linear and nonlinear geometry for the nodal dof transformations; 
    depending on the value of the character variable ACTION the function returns information
    in data structure ELEMRESP for the element with number EL_NO, end node coordinates XYZ,
    and material and loading properties in the data structure ELEMDATA."""
    def __init__(self):
        pass 

    def ke_matrix(self,u):
        """Stiffness matrix"""
        pass
    
    def kg_matrix(self,u):
        """Stiffness matrix"""
        pass

    def ag_matrix(self):
        pass