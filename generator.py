# This function will be used to generate random variables to be used as data.

# Import usefull libraries:

import numpy as np
from numpy import  random as rnd

# Define function:

def gen(m):
    '''
    

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