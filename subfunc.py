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
    L : FLOAT
        The length of the truss (m).
    h : FLOAT
        The heigth of the truss (m).
    
    Returns
    -------
    S : FLOAT VECTOR
        The legth of the truss' members (m).
    
    '''
    
    temp1 = np.sqrt(h**2 + (L/3/2)**2)
    temp2 = L/3
    
    temp3 = np.zeros((1, 11), dtype = int)
    temp3[0, ::2] = 1
    
    temp4 = np.ones((1, 11), dtype = int) 
    temp4 -= temp3
    
    S = temp1 * temp3 + temp2 * temp4
    
    return S

# 2.2: Calculate the members' internal forces.

def axial_f(P, S, L, h):
    '''
    
    Parameters
    ----------
    P : FLOAT VECTOR
        The forces that the truss bears (kN).
    S : FLOAT VECTOR
        The legth of the truss' members (m).
    L : FLOAT
        The length of the truss (m).
    h : FLOAT
        The heigth of the truss (m).

    Returns
    -------
    F : FLOAT VECTROR
        The axial forces of the truss' members (kN).
    
    '''
    
    SP = np.sum(P, axis = 1, keepdims = True)
    
    
    temp1 = -SP / 2
    temp1 /= (h / S[0, 0])
    
    temp2 = np.abs(temp1) 
    temp2 *= (S[0, 1] / S[0, 0]) / 2
    
    temp3 = (np.abs(temp1) * (h / S[0, 0]) - P[0, 0])
    temp3 /= (h / S[0, 2])
    
    temp4 = np.abs(temp1) + np.abs(temp3)
    temp4 *= -(S[0, 1] / S[0, 2]) / 2
    
    temp5 = -temp3
    
    temp6 = (np.abs(temp3) + np.abs(temp5))
    temp6 *= (S[0, 1] / S[0, 2]) / 2
    temp6 += np.abs(temp2)
    
    temp = np.array([temp1, temp2, temp3, temp4, temp5, temp6])
    temp = temp.flatten()
    temp = np.reshape(temp, (1, 6))
    
    F = np.zeros((1, 11), dtype = float)
    F[0, :6] = temp
    temp = temp[0, ::-1]
    temp = np.delete(temp, 0)

    F[0, 6:] = temp
    
    return F

# 2.3: Calculate the members' resistance.

def mem_res(S, b, t, fy, E, k):
    '''
    

    Parameters
    ----------
    S           : FLOAT VECTOR
                    The legth of the truss' members (m).
    b           : FLOAT VECTOR
                    The width of the truss' members (mm).
    t           : FLOAT VECTOR
                    The thickness of the truss' members (mm).
    fy          : INTEGER
                    The resistance of the steel (MPa).
    E           : INTEGER
                    The elasticity modulous of the steel (GPa).
    k           : FLOAT
                    The hardening parameter of the steel (-).

    Returns
    -------
    T_for_cr   : FLOAT VECTOR
                    The tensile resistance of the truss' members (kN).
    C_for_cr   : FLOAT VECTOR
                    The resistance of the truss' members to compression (kN).

    '''

    b = b.reshape((1, len(b)))
    t = t.reshape((1, len(t)))
    
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
    F          : FLOAT VECTROR
                    The axial forces of the truss' members (kN).
    T_for_cr   : FLOAT VECTOR
                    The tensile resistance of the truss' members (kN).
    C_for_cr   : FLOAT VECTOR
                    The resistance of the truss' members to compression (kN).

    Returns
    -------
    Failure    : INTEGER (0 OR 1)
                    1 if the truss fails, 0 otherwise.

    '''
    
    Failure = (C_for_cr > F) | (T_for_cr < F)
    temp = Failure.astype(int)
    Failure = np.amax(temp, axis = 1)
    
    Failure = np.asscalar(Failure)
    
    return Failure

# 3: Check that width and thickness are not out of bounds.

# 3.1: Create custom input error class:

class InputError(Exception):
    """Base class for input exceptions in this module."""
    
    pass

# 3.1.1: Create width error subclass.

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
    
# 3.1.2: Create thickness error subclass.
    
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
    
# 3.2: Test the variables b and t for input error.

def bounds(b, t):
    '''
    

    Parameters
    ----------
    b : FLOAT VECTOR
        The width of the truss' members (mm).
    t : FLOAT VECTOR
        The thickness of the truss' members (mm).

    Returns
    -------
    None.

    '''
    
    b = b.reshape((1, len(b)))
    t = t.reshape((1, len(t)))
    
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
            