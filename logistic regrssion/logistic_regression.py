# This program applies simple logistic regression to the data produced by the 
# solver.

# IMPORT USEFULL LIBRARIES:

import numpy as np
import matplotlib.pyplot as plt
import subfunc as sf
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.datasets
import sklearn.linear_model
from sklearn.externals import joblib
import sklearn.metrics as skm
from sklearn.metrics import plot_confusion_matrix

# IMPORT DATA:

X = sf.importer('variables', 'inp')

Y = sf.importer('results', 'out')

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

# PLOT THE CONFUSION MATRIX FOR THE TRAINING SET:

# PLOT THE CONFUSION MATRIX FOR THE TRAINING SET:

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

