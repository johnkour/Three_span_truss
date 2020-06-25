# This program solves a standard 2D truss with three spans pinned to the ground

'''
    In lines 11 - 79 you will find my custom classes, craeted to raise some 
error messages.
    In lines 80 - 482 you will find my custom functions, used to speed up the 
program and free some memory.
    From line 483 on you will find the main program for the analysis.
'''

# 1.1: Import useful libraries.

import numpy as np
from numpy import  random as rnd
import pandas as pd
import sys

# ====================Define custom classes===================================

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


# ====================Define custom functions=================================

# 2: Handmade functions.

# 2.1: Calculate the members' length.

def mem_length(L, h):
    '''
    
    Summary
    ----------
    This function is used to calculate the length of the truss' members.
    It also works if the length and the heigth of the truss change in every 
    training example. Note that with minor changes to the indexing it will 
    produce results for a truss more than 11 sections.
    
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

    temp2 = L/3    
    temp1 = np.hypot(h, temp2 /2)
    
    temp3 = np.zeros((m, 11), dtype = int)
    temp3[:, ::2] = 1
    
    temp4 = np.ones((m, 11), dtype = int) 
    temp4 -= temp3
    
    S = temp1 * temp3 + temp2 * temp4
    
    return S

# 2.2: Calculate the members' internal forces.

def axial_f(P, S, L, h):
    '''
    
    Summary
    ----------
    This function is used to calculate the axial force of the truss' members.
    We uttilize the symmetry of the problem to write less lines of smarter 
    code using an auxiliary array temp to compute our final product F. The 
    nodal loads(P[i,j]) are differ from node to node and from analysis to
    analysis. Here as before the function runs many analyses simultaneously.
    
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
    
    Summary
    ----------
    This function is used to define the resistance of each member of the truss
    to tension and compression. Once again we chose to run all of our analyses
    at the same time. For the purposes of this assignment fy and E are the 
    same for every analysis. Note that it would make no sense to change fy for
    different members in the same analysis and that E is pretty much standard
    regardless of member or analysis.
    
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
    
    Summary
    ----------
    This function is used to estimate whether a member (which means the whole
    truss) fails or not. The output of this function are the true values of Y
    that will be used later to train machine learning algorithms to simulate
    the structural analysis programmed here.
    
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

# 3.2.1: Test the variable m for input error.

def ex_num(m):
    '''
    
    Summary
    ----------
    This function is used to evaluate if the number of training examples typed
    by the user via keyboard is an integer number. If not an error message is
    displayed so as to inform the user. It also converts m from string to
    integer.

    Parameters
    ----------
    m : STRING
        Number of training examples.

    Returns
    -------
    m : INTEGER(?)
        Number of training examples.

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
    
    Summary
    ----------
    This function is used to evaluate if the section properties (thickness and
    width) are out of bounds. If they are, an error message is
    displayed so as to inform the user and the analysis is stopped(the progam
    closes).
    
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
            

# Define function to generate the data:

def generator(m):
    '''
    
    Summary
    ----------
    This function is used to produce m number of training examples(or analyses
    ). Each training example consists of a vector of 25 elements, the first 11
    are the width of the sections, then the next 11 are the thickness of the 
    sections and the last 3 are the nodal loads. Note that if the code above is
    slightly modified there could be another 3 variables: the length and heigth
    of the truss and the quality of the steel(L, h, fy). We would then have
    to give them random values for each analysis below.
    
    Parameters
    ----------
    m : INTEGER
        Number of training examples.

    Returns
    -------
    b : (m x 11) ARRAY
        The width of the truss' members (mm).
    t : (m x 11) ARRAY
        The thickness of the truss' members (mm).
    P : (m x 3) ARRAY
        The load of the truss in each of the 3 nodes (kN).

    '''
    
    # b starts at 80mm, stops at 300mm and can be raised by 5mm.
    
    b = rnd.randint(0, 44, size = (m, 11)) * 5
    b += 80
    
    # t starts at 3mm, stops at 20mm and can be raised by 1mm.
    
    t = rnd.randint(3, 20, size = (m, 11))
    
    # P starts at 0kN and stops at 250kN.
    
    P = rnd.random(size = (m, 3))
    P *= 250
    P = np.round(P, decimals = 2)
    
    return b, t, P

# Define function to export the data to CSV:

def exporter(path, name, tup, fl_form):
    '''
    
    Summary
    ----------
    This function is used to export the data((b, t, P) or (Failure,)) to CSV 
    format in order to use them in the next programs. The Tupple (tup) should
    contain the parameters(b, t, P) or the true values(Failure,), so as to have
    the input and the output of the Machine Learning algorithms seperated to
    different files and avoid further confussion. We recommend using fl_form =
    %d, meaning floating format is integer for the true values(Failure,) and
    fl_form = %.2f, meaning floating format with 2 decimals for the parameters.
    The name of the file in which the data will be stored should not have the
    suffix '.csv'. We use the pandas library to store the data to the CSV
    because pandas are faster than the standard methods when dealing with big
    data.
    
    Parameters
    ----------
    path : STRING
        The path, where the file containing the data will be stored.
    name : STRING
        The name of the CSV file containing the data.
    tup : TUPLE OF ARRAYS
        The tuple containing the arrays with the data to be stored.
    fl_form : STRING
        The format of the numbers to be stored in the CSV.

    Returns
    -------
    None.

    '''

    name += '.csv'
    data = np.concatenate(tup, axis = 1)
    df = pd.DataFrame(data)
    
    path += '/' + name
    
    df.to_csv(path, float_format = fl_form, header = False, index = False)



# ============================Main Program====================================

# 1.2: Define the geometry of the truss.

m = input(
    'Please enter the number of training examples ' +
    '(or press enter fordefault value: 500k)\n')

if len(m) < 1:  m = 500 * 10**3

m = ex_num(m)

L = np.ones((m, 1))
L *= 20
h = np.ones((m, 1))
h *= 1.5
# print(L, h)

# 2.1: Calculate the members' length.

S = mem_length(L, h)
# print(S)

# print(S)

# 1.3 & 1.5: Define the load of the truss and the dimensions of it's sections.

# load = np.ones((m, 3))
# load *= 100
# P = np.array(load)

b, t, P = generator(m)

# 3.2: Test the variables b and t for input error.

bounds(b, t)

# 2.2: Calculate the members' internal forces.

F = axial_f(P, S, L, h)
# print(F)

# 1.4: Define the properties of the steel used.

fy = 275            # MPa
k = 0.80
E = 210             # GPa

# print(F)

# 2.3: Calculate the members' resistance.

T_for_cr, C_for_cr = mem_res(S, b, t, fy, E, k)
# print(T_for_cr)
# print(C_for_cr)

# 2.4: Evaluate whether the member fails or not.
Failure = evaluation(F, T_for_cr, C_for_cr)
# print(Failure)

n_fail = np.sum(Failure, axis = 0)
# print(n_fail, m-n_fail)

# 4: Export results to csvs.

message = 'Please enter the path where the file will be stored'
message += ' (Press enter to set path: '
message += 'D:/jkour/Documents/Σχολή/2ο έτος/Εαρινό Εξάμηνο/Προγραμματισμός Η.Υ'
message += '/Εργασία εξαμήνου/MyCode/Truss_analysis)\n'

pth = input(message)

if len(pth) < 1:
    
    pth = 'D:/jkour/Documents/Σχολή/2ο έτος/Εαρινό Εξάμηνο/Προγραμματισμός Η.Υ'
    pth += '/Εργασία εξαμήνου/MyCode/Truss_analysis'

tup = (b, t, P)

exporter(pth, 'variables', tup, '%.2f')

tup = (Failure, )

exporter(pth, 'results', tup, '%d')
