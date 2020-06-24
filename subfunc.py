# This file contains some basic functions for the solver.py to work

# 1.1: Import useful libraries.

import numpy as np
import sys

# 2: Handmade functions.

# 2.1: Calculate the members' length.

def mem_length(L, h):
    '''
    
    Parameters
    ----------
    L : (m x 1) ARRAY
        The length of the truss (m).
    h : (m x 1) ARRAY
        The heigth of the truss (m).
    
    Returns
    -------
    S : (m x 11) ARRAY
        The legth of the truss' members (m).
    
    '''
    
    m, n =np.shape(L)
    
    temp1 = np.sqrt(h**2 + (L/3/2)**2)
    temp2 = L/3
    
    temp3 = np.zeros((m, 11), dtype = int)
    temp3[:, ::2] = 1
    
    temp4 = np.ones((m, 11), dtype = int) 
    temp4 -= temp3
    
    S = temp1 * temp3 + temp2 * temp4
    
    return S

# 2.2: Calculate the members' internal forces.

def axial_f(P, S, L, h):
    '''
    
    Parameters
    ----------
    P : (m x 3) ARRAY
        The forces that the truss bears (kN).
    S : (m x 11) ARRAY
        The legth of the truss' members (m).
    L : (m x 1) ARRAY
        The length of the truss (m).
    h : (m x 1) ARRAY
        The heigth of the truss (m).

    Returns
    -------
    F : (m x 11) ARRAY
        The axial forces of the truss' members (kN).
    
    '''
    
    m, n = np.shape(S)
    SP = np.sum(P, axis = 1, keepdims = True)
#     print(SP)
#     print(S[:, 0])
    
    temp = np.zeros((m, 6))
    
    temp[:, 0] = -SP.flatten() / 2
#     print(temp[:, 0])
    temp[:, 0] /= (h.flatten() / S[:, 0])
#     print(temp[:, 0])
    
    temp[:, 1] = np.abs(temp[:, 0]) 
    temp[:, 1] *= S[:, 1] / 2
    temp[:, 1] /= S[:, 0]
#     print(temp[:, 1])
    
    temp[:, 2] = np.abs(temp[:, 0]) * (h.flatten() / S[:, 0])
    temp[:, 2] -= P[:, 0]
    temp[:, 2] /= (h.flatten() / S[:, 2])
#     print(temp[:, 2])
    
    temp[:, 3] = np.abs(temp[:, 0]) + np.abs(temp[:, 2])
    temp[:, 3] *= -(S[:, 1] / S[:, 2]) / 2
#     print(temp[:, 3])
    
    temp[:, 4] = -temp[:, 2]
#     print(temp[:, 4])
    
    temp[:, 5] = (np.abs(temp[:, 2]) + np.abs(temp[:, 4]))
    temp[:, 5] *= (S[:, 1] / S[:, 2]) / 2
    temp[:, 5] += np.abs(temp[:, 1])
#     print(temp[:, 5])
    
#     print(temp)
    
    F = np.zeros((m, n), dtype = float)
    F[:, :6] = temp
    temp = temp[:, ::-1]
    temp = np.delete(temp, 0, 1)

    F[:, 6:] = temp
    
    return F

# 2.3: Calculate the members' resistance.

def mem_res(S, b, t, fy, E, k):
    '''
    

    Parameters
    ----------
    S           : (m x 11) ARRAY
                    The legth of the truss' members (m).
    b           : (m x 11) ARRAY
                    The width of the truss' members (mm).
    t           : (m x 11) ARRAY
                    The thickness of the truss' members (mm).
    fy          : INTEGER
                    The resistance of the steel (MPa).
    E           : INTEGER
                    The elasticity modulous of the steel (GPa).
    k           : FLOAT
                    The hardening parameter of the steel (-).

    Returns
    -------
    T_for_cr   : (m x 11) ARRAY
                    The tensile resistance of the truss' members (kN).
    C_for_cr   : (m x 11) ARRAY
                    The resistance of the truss' members to compression (kN).

    '''

    m, n = np.shape(b)
#     b = b.reshape((1, len(b)))
#     t = t.reshape((1, len(t)))
    
    b = b / 10 ** 3
    t = t / 10 ** 3
    
    A = 4 * b * t
    Vol = A * S
    
    fy *= 10**3
    E *= 10**6
    
    T_for = fy * A
    C_for = - k * T_for
    
    Buck_for = np.pi**2 * E * ((b + t)**4 / 12 - (b - t)**4 / 12)
    Buck_for /= -S**2
    
    T_for_cr = T_for
    C_for_cr = np.maximum(C_for, Buck_for)
    
    return T_for_cr, C_for_cr

# 2.4: Evaluate whether the member fails or not.

def evaluation(F, T_for_cr, C_for_cr):
    '''
    

    Parameters
    ----------
    F          : (m x 11) ARRAY
                    The axial forces of the truss' members (kN).
    T_for_cr   : (m x 11) ARRAY
                    The tensile resistance of the truss' members (kN).
    C_for_cr   : (m x 11) ARRAY
                    The resistance of the truss' members to compression (kN).

    Returns
    -------
    Failure    : (m x 1) ARRAY WITH ELEMENT VALUES 0 OR 1
                    1 if the truss fails, 0 otherwise.

    '''
    
    m, n = np.shape(F)
    
    Failure = (C_for_cr > F) | (T_for_cr < F)
    temp = Failure.astype(int)
    Failure = np.amax(temp, axis = 1, keepdims = True)
    
#     Failure = np.asscalar(Failure)
    
    return Failure

# 3: Check that width and thickness are not out of bounds.

# 3.1: Create custom input error class:

class InputError(Exception):
    """Base class for input exceptions in this module."""
    
    pass

# 3.1.1: Create number of training examples error subclass.

class ExamplesNumber(InputError):
    """Exception raised for errors in the input of the number of training 
        examples.
    
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload
        
    def __str__(self):
        return str(self.message)

# 3.1.2: Create width error subclass.

class WidthInput(InputError):
    """Exception raised for errors in the input of the width.
    
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload
        
    def __str__(self):
        return str(self.message)
    
# 3.1.3: Create thickness error subclass.
    
class ThickInput(InputError):
    """Exception raised for errors in the input of the thickness.
    
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload
        
    def __str__(self):
        return str(self.message)

# 3.2.1: Test the variable m for input error.

def ex_num(m):
    '''
    

    Parameters
    ----------
    m : INTEGER(?)
        Number of training examples.

    Returns
    -------
    None.

    '''
    
    try:
        
        m = int(m)
        
    except:
        
        try:
            raise ExamplesNumber("INVALID INPUT",
                                 "The number of training examples " +
                                 "is not an integer.")
        except ExamplesNumber as error:
            print(str(error))
            print("Detail: {}".format(error.payload))
            sys.exit()
    
    return m
    
# 3.2.2: Test the variables b and t for input error.

def bounds(b, t):
    '''
    

    Parameters
    ----------
    b : (m x 11) ARRAY
        The width of the truss' members (mm).
    t : (m x 11) ARRAY
        The thickness of the truss' members (mm).

    Returns
    -------
    None.

    '''
    
#     b = b.reshape((1, len(b)))
#     t = t.reshape((1, len(t)))
    
    if np.any(b < 80) | np.any(b > 300):
        try:
            raise WidthInput("INVALID INPUT", "The width is out of bounds.")
        except WidthInput as error:
            print(str(error))
            print("Detail: {}".format(error.payload))
            sys.exit()
    
    if np.any(t < 3) | np.any(t > 20):
        try:
            raise ThickInput("INVALID INPUT", "The thickness is out of bounds.")
        except ThickInput as error:
            print(str(error))
            print("Detail: {}".format(error.payload))
            sys.exit()
            