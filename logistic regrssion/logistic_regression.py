# This program applies simple logistic regression to the data produced by the 
# solver.

'''
    In lines 26 - 51 you will find my custom classes, craeted to raise some 
error messages.
    In lines 52 - 119 you will find my custom functions, used to speed up the 
program and free some memory.
    From line 120 on you will find the main program for the logistic regression.
'''

# IMPORT USEFULL LIBRARIES:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.datasets
import sklearn.linear_model
import joblib
import sklearn.metrics as skm
from sklearn.metrics import plot_confusion_matrix

# ====================Define custom classes===================================

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

# =====================Define custom function=================================

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

# ============================Main Program====================================

# IMPORT DATA:

X = importer('variables', 'inp')

Y = importer('results', 'out')

# DATASET SIZE:

m = X.shape[1]                # number of training examples
n_x = X.shape[0]              # number of input variables
# print(n_x, m)

# SELECT HOW TO SPLIT THE DATASET:

if (m < 500 * 10**3):         # Split the dataset.
    per_test = 20 / 10**2     # For small datasets 20% goes to testing.
elif (m < 1.5 * 10**6):
    per_test = 2 / 10**2      # For big datasets 2% goes to testing.
else:
    per_test = 5 / 10**3      # For very big datasets 20% goes to testing.

# DIVIDE THE DATA INTO TRAIN AND TEST SETS:

X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size = per_test)

# TRAIN THE LOGISTIC REGRESSION CLASSIFIER:

clf = sklearn.linear_model.LogisticRegressionCV(max_iter = 2.5 * 10**2)
clf.fit(X_train, np.ravel(Y_train))

# SAVE MODEL TO FILE:

message = 'Please enter the name of the file to save the model:'
message += '(For joblib_model press enter)\n'

joblib_file = input(message)

if len(joblib_file) < 1:    joblib_file = 'joblib_model'

joblib_file += '.pkl'
joblib.dump(clf, joblib_file)

clf = joblib.load(joblib_file)

# MAKE PREDICTIONS FOR TRAINING:

Y_hat = clf.predict(X_train)

# PRODUCE THE MEASURES TO EVALUATE THE LEARNING ALGORITHM FOR THE TRAINING SET:

Con_mat = dict()

Prec = dict()
Rec = dict()
Accur = dict()
F1_Score = dict()

Con_mat['Train'] = skm.confusion_matrix(Y_train, Y_hat)
Prec['Train'] = skm.precision_score(Y_train, Y_hat)
Rec['Train'] = skm.recall_score(Y_train, Y_hat)
Accur['Train'] = skm.accuracy_score(Y_train, Y_hat)
F1_Score['Train'] = skm.f1_score(Y_train, Y_hat)

# PLOT THE CONFUSION MATRIX FOR THE TRAINING SET:

title = 'Confusion Matrix for the training set'
disp = plot_confusion_matrix(clf, X_train, Y_train)
disp.ax_.set_title(title)
plt.show()

# PRINT THE MEASURES TO EVALUATE THE LEARNING ALGORITHM FOR THE TRAINING SET:

# Accuracy

print ('Accuracy of logistic regression in training: %.2f ' % float(Accur['Train']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# Precision

print ('Precision of logistic regression in training: %.2f ' % float(Prec['Train']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# Recall

print ('Recall of logistic regression in training: %.2f ' % float(Rec['Train']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# F1-Score

print ('F1-Score of logistic regression in training: %.2f ' % float(F1_Score['Train']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# MAKE PREDICTIONS FOR TESTING:

Y_hat = clf.predict(X_test)


# PRODUCE THE MEASURES TO EVALUATE THE LEARNING ALGORITHM FOR THE TRAINING SET:

Con_mat['Test'] = skm.confusion_matrix(Y_test, Y_hat)
Prec['Test'] = skm.precision_score(Y_test, Y_hat)
Rec['Test'] = skm.recall_score(Y_test, Y_hat)
Accur['Test'] = skm.accuracy_score(Y_test, Y_hat)
F1_Score['Test'] = skm.f1_score(Y_test, Y_hat)


# PLOT THE CONFUSION MATRIX FOR THE TEST SET:

title = 'Confusion Matrix for the test set'
disp = plot_confusion_matrix(clf, X_test, Y_test)
disp.ax_.set_title(title)
plt.show()

# PRINT THE MEASURES TO EVALUATE THE LEARNING ALGORITHM FOR THE TEST SET:

# Accuracy

print ('Accuracy of logistic regression in testing: %.2f ' % float(Accur['Test']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# Precision

print ('Precision of logistic regression in testing: %.2f ' % float(Prec['Test']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# Recall

print ('Recall of logistic regression in testing: %.2f ' % float(Rec['Test']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# F1-Score

print ('F1-Score of logistic regression in testing: %.2f ' % float(F1_Score['Test']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

