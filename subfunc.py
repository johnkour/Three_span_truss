# This file contains some basic functions for the solver.py to work

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

# 3.1.4: Create data import error subclass.

class DataInput(InputError):
    """Exception raised for errors in the input of the data from CSV.
    
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload
        
    def __str__(self):
        return str(self.message)

# ============================================================================

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
    
    b = rnd.random_integers(0, 44, size = (m, 11)) * 5
    b += 80
    
    # t starts at 3mm, stops at 20mm and can be raised by 1mm.
    
    t = rnd.random_integers(3, 20, size = (m, 11))
    
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

# Define function to import the data from CSV:

def importer(name, data_t, ch_size = 100 * 10**5):
    '''
    
    Summary
    ----------
    This function is used to import the data created by the solver from the
    CSV file, to the progamm that fits the Machine Learning algorithm. It is
    used 2 times in each program, one to import the parameters X and one to
    import the true values Y. The different training examples(analyses) are
    assigned to each column of the array X(or Y). The rows store the different
    parameters (b[i], t[i], P[i] for array X) or classes (1 class = 1 row for 
    array Y) of the model. The user should provide an auxiliary variable,
    data_t ('inp' for input and 'out' for output data). Here we use list
    comprehension to make our for loops run faster and to create the temporary
    list lst. lst is used to ensure that no data are lost when extracting them
    from the CSV. Again, pandas are faster when dealing with big data,
    especially when we seperate the data to chunks. Note that if the user does
    not assign 'inp' or 'out' to data_t an error appears on screen and the
    program is terminated.
    
    Parameters
    ----------
    name    : STRING
                The name of the CSV file with the data.
    data_t  : STRING
                'inp': if the file contains the variables.
                'out': if the file contains the results.
    ch_size : INTEGER
                The size of the chunks of data to be parsed simultaneously.

    Returns
    -------
    data    : (25 x m) ARRAY or (1 x m) ARRAY
                The array which contains the variables of the problem (rows)
                for all the training examples in the CSV file.

    '''
    
    if data_t == 'inp':
        
        lst = [ 'b' + str(i) for i in range(1, 12)]
        lst += [ 't' + str(i) for i in range(1, 12)]
        lst += [ 'P' + str(i) for i in range(1, 4)]
        
    elif data_t == 'out':
        
        lst = ['Failure']
        
    else:
        
        try:
            raise DataInput("INVALID INPUT", "The variable data must have value 'inp' or 'out'.")
        except DataInput as error:
            print(str(error))
            print("Detail: {}".format(error.payload))
            sys.exit()
    
    name += '.csv'
    chunks = pd.read_csv(name, names = lst, chunksize = ch_size)
    
    data = pd.concat(chunks)
    data = data.values
    data = data.T
    
    return data
