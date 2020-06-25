# This progam divideds the dataset stored in two CSV files to 3 smaller 
# datasets and stores them in 6 new CSVs: X_train, Y_train, X_dev, Y_dev,
# X_test & Y_test.

'''
    This program uses the build-in functions to seperate the initial dataset 
randomly to 3 sub-datasets: Train(X_train, Y_train), Development(X_dev, Y_dev)
and Test(X_test & Y_test). The sub-datasets are then stored to CSV files(2 for
every sub-dataset, 6 in total). The sub-datasets need to be fixxed for reasons
of model selection and final evaluation of the models.
'''

# Import usefull libraries:

import numpy as np
import pandas as pd
import sys

# ==========================Define custom classes=============================

# 3.1: Create custom input error class:

class InputError(Exception):
    """Base class for input exceptions in this module."""
    
    pass

# 3.1.1: Create data import error subclass.

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

# =========================Define custom functions============================

# IMPORT THE DATA:

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

# DIVIDE THE DATA:

def data_division(X, Y):
    '''
    
    Summary
    ----------
    This function divideds the entirety of the dataset to three parts sampled 
    with the training examples in random order. The percentage of the split 
    depends on the size of the initial dataset.
    
    Parameters
    ----------
    X : 2-D Array of size: n_x, m
        where n_x is the number of input variables and m are the training examples.
        It will be divided to two 2-D arrays: X_train and X_test.
        
    Y : 2-D Array of size: 1, m
        where m are the training examples.
        It will be divided to two 2-D arrays: Y_train and Y_test.
    Returns
    -------
    Three dictionaries: Train, Dev, Test, where:
        Train contains the matrices: X_train and Y_train (keys: X_train, Y_train)
        Train contains the matrices: X_dev and Y_dev (keys: X_dev, Y_dev)
        Test contains the matrices: X_test and Y_test (keys: X_test, Y_test)
    '''
    
    Train = dict()                # Initialize Train dictionary.
    Dev = dict()                  # Initialize Development dictionary.
    Test = dict()                 # Initialize Test dictionary.
    
    n_x, m = X.shape              # Get the geometry of the problem.
    
    if (m < 500 * 10**3):         # Split the dataset.
        per_test = 20 / 10**2     # For small datasets 20% goes to testing.
        per_dev = 20 / 10**2      # For small datasets 20% goes to development.
    elif (m < 1.5 * 10**6):
        per_test = 1 / 10**2      # For big datasets 1% goes to testing.
        per_dev = 1 / 10**2      # For big datasets 1% goes to development.
    else:
        per_test = 1 / 10**3      # For very big datasets 0.1% goes to testing.
        per_dev = 4 / 10**3      # For very big datasets 0.4% goes to development.
    
    m_test = np.floor(per_test * m)
    m_test = int(m_test)          # Slice indices must be integers.
    m_dev = np.floor(per_dev * m)
    m_dev = int(m_dev)          # Slice indices must be integers.
    m_train = m - (m_dev + m_test)
    
    X_train = X[:, :m_train]
    Y_train = Y[:, :m_train]
    
    X_dev = X[:, m_train:(m_train + m_dev)]
    Y_dev = Y[:, m_train:(m_train + m_dev)]
    
    X_test = X[:, (m_train + m_dev):]
    Y_test = Y[:, (m_train + m_dev):]
    
    Train = {'X': X_train, 'Y': Y_train}
    Dev = {'X': X_dev, 'Y': Y_dev}
    Test = {'X': X_test, 'Y': Y_test}
    
    return Train, Dev, Test

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

# IMPORT DATA:

X = importer('variables', 'inp')

Y = importer('results', 'out')

n_x, m = X.shape

Y = np.reshape(Y, (1, m))

# DIVIDE THE DATA INTO TRAIN AND TEST SETS:

Train, Dev, Test = data_division(X, Y)

X_train = Train['X']
Y_train = Train['Y']

X_dev = Dev['X']
Y_dev = Dev['Y']

X_test = Test['X']
Y_test = Test['Y']

# 4: Export results to csvs.

message = 'Please enter the path where the file will be stored'
message += ' (Press enter to set path: '
message += 'D:/jkour/Documents/Σχολή/2ο έτος/Εαρινό Εξάμηνο/Προγραμματισμός Η.Υ'
message += '/Εργασία εξαμήνου/MyCode/DNN (3_layer))\n'

pth = input(message)

if len(pth) < 1:
    
    pth = 'D:/jkour/Documents/Σχολή/2ο έτος/Εαρινό Εξάμηνο/Προγραμματισμός Η.Υ'
    pth += '/Εργασία εξαμήνου/MyCode/DNN (3_layer)'

tup = (X_train.T, )

exporter(pth, 'X_train', tup, '%.2f')

tup = (Y_train.T, )

exporter(pth, 'Y_train', tup, '%d')

tup = (X_dev.T, )

exporter(pth, 'X_dev', tup, '%.2f')

tup = (Y_dev.T, )

exporter(pth, 'Y_dev', tup, '%d')

tup = (X_test.T, )

exporter(pth, 'X_test', tup, '%.2f')

tup = (Y_test.T, )

exporter(pth, 'Y_test', tup, '%d')
