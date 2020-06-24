# This program applies simple logistic regression to the data produced by the 
# solver.

# IMPORT USEFULL LIBRARIES:

import numpy as np
import subfunc as sf
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.externals import joblib

# DEFINE FUNCTION TO SPLIT THE DATA:

def data_division(X, Y):
    """
    
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
    Two dictionaries: Train, Test, where:
        Train contains the matrices: X_train and Y_train (keys: X_train, Y_train)
        Test contains the matrices: X_test and Y_test (keys: X_test, Y_test)
    """
    
    Train = dict()                # Initialize Train dictionary.
    Test = dict()                 # Initialize Test dictionary.
    
    n_x, m = X.shape              # Get the geometry of the problem.
    
    if (m < 500 * 10**3):         # Split the dataset.
        per_test = 20 / 10**2     # For small datasets 20% goes to testing.
    elif (m < 1.5 * 10**6):
        per_test = 2 / 10**2      # For big datasets 2% goes to testing.
    else:
        per_test = 5 / 10**3      # For very big datasets 20% goes to testing.
    
    m_test = np.floor(per_test * m)
    m_test = int(m_test)          # Slice indices must be integers.
    m_train = m - m_test
    
    X_train = X[:, :m_train]
    Y_train = Y[:, :m_train]
    
    X_test = X[:, m_train:]
    Y_test = Y[:, m_train:]
    
    Train = {'X': X_train, 'Y': Y_train}
    Test = {'X': X_test, 'Y': Y_test}
    
    return Train, Test

# IMPORT DATA:

X = sf.importer('variables', 'inp')

Y = sf.importer('results', 'out')

# DIVIDE THE DATA INTO TRAIN AND TEST SETS:

Train, Test = data_division(X, Y)

X_train = Train['X']
Y_train = Train['Y']

X_test = Test['X']
Y_test = Test['Y']

# TRAINING SET SIZE:

m = X_train.shape[1]                # number of training examples
n_x = X_train.shape[0]              # number of input variables
# print(n_x, m)

# TRAIN THE LOGISTIC REGRESSION CLASSIFIER:

clf = sklearn.linear_model.LogisticRegressionCV(max_iter = 2 * 10**2)
clf.fit(X_train.T, np.ravel(Y_train))

# SAVE MODEL TO FILE:

message = 'Please enter the name of the file to save the model:'
message += '(For joblib_model press enter)\n'

joblib_file = input(message)

if len(joblib_file) < 1:    joblib_file = 'joblib_model'

joblib_file += '.pkl'
joblib.dump(clf, joblib_file)

clf = joblib.load(joblib_file)

# MAKE PREDICTIONS FOR TRAINING:

Accur = dict()

Y_hat = clf.predict(X_train.T)
Accur['Train'] = (np.dot(Y_train,Y_hat) + np.dot(1-Y_train,1-Y_hat))/(Y_train.size)

print ('Accuracy of logistic regression in training: %d ' % float(Accur['Train']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# MAKE PREDICTIONS FOR TESTING:

Y_hat = clf.predict(X_test.T)
Accur['Test'] = (np.dot(Y_test,Y_hat) + np.dot(1-Y_test,1-Y_hat))/(Y_test.size)

print ('Accuracy of logistic regression in testing: %d ' % float(Accur['Test']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

