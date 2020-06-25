# This program fits a 3 layer Deep Neural Network to the data produced by the 
# solver.

'''
    In lines 21 - 46 you will find my custom classes, craeted to raise some 
error messages.
    In lines 47 - 474 you will find my custom functions, used to speed up the 
program and free some memory.
    From line 475 on you will find the main program for the logistic regression.
'''

# IMPORT USEFUL LIBRARIES:

import numpy as np
import pandas as pd
import sys
import math
import matplotlib.pyplot as plt
import tensorflow as tf

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

# =====================Define custom functions================================

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

# RANDOM MINI BATCHES:

def random_mini_batches(X, Y, mini_batch_size = 64):
    '''
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for failure / 0 otherwise), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    '''
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        k = num_complete_minibatches
        
        mini_batch_X = shuffled_X[:, k * mini_batch_size :]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# FEEDFORWARD PROPAGATION:

def feedfor(parameters, X):
    '''
    

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    X         : TYPE
        DESCRIPTION.

    Returns
    -------
    Y_hat:

    '''
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = W1 @ X + b1
    A1 = np.maximum(Z1, 0)
    Z2 = W2 @ A1 + b2
    A2 = np.maximum(Z2, 0)
    Z3 = W3 @ A2 + b3
    A3 = 1 / (1 + np.exp(-Z3))
    
    Y_hat = (A3 >= 0.5).astype(int)
    
    return Y_hat
    
def Conf_matrix(Y, Y_hat):
    '''
    

    Parameters
    ----------
    Y : TYPE
        DESCRIPTION.
    Y_hat : TYPE
        DESCRIPTION.

    Returns
    -------
    Conf_matrix : TYPE
        DESCRIPTION.
    Acc : TYPE
        DESCRIPTION.
    Prec : TYPE
        DESCRIPTION.
    Rec : TYPE
        DESCRIPTION.
    F1_Score : TYPE
        DESCRIPTION.

    '''
    
    temp1 = Y == 1
    temp2 = Y_hat == 1
    temp3 = (temp1 & temp2).astype(int)
    
    Tr_pos = np.sum(temp3)
    
    temp1 = Y == 1
    temp2 = Y_hat == 0
    temp3 = (temp1 & temp2).astype(int)
    
    Tr_neg = np.sum(temp3)
    
    temp1 = Y == 0
    temp2 = Y_hat == 1
    temp3 = (temp1 & temp2).astype(int)
    
    Fal_pos = np.sum(temp3)
    
    temp1 = Y == 0
    temp2 = Y_hat == 0
    temp3 = (temp1 & temp2).astype(int)
    
    Fal_neg = np.sum(temp3)
    
    Conf_matrix = np.array([[Tr_pos, Tr_neg], [Fal_pos, Fal_neg]])
    
    Acc = (Tr_pos + Fal_neg) / (Tr_pos + Tr_neg + Fal_pos + Fal_neg)
    
    Prec = Tr_pos / (Tr_pos + Fal_pos)
    Rec = Tr_pos / (Tr_pos + Fal_neg)

    F1_Score = 2 * Prec * Rec / (Prec + Rec)
    
    return Conf_matrix, Acc, Prec, Rec, F1_Score

# IMPORT X,Y AS PLACEHOLDERS:

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32, [n_x, None], name = 'X')
    Y = tf.placeholder(tf.float32, [n_y, None], name = 'Y')
    
    return X, Y

# INITIALIZATION:

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [50, 25]
                        b1 : [50, 1]
                        W2 : [50, 50]
                        b2 : [50, 1]
                        W3 : [1, 50]
                        b3 : [1, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
        
    W1 = tf.get_variable("W1", [50, 25], initializer = tf.initializers.he_normal())
    b1 = tf.get_variable("b1", [50, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [50, 50], initializer = tf.initializers.he_normal())
    b2 = tf.get_variable("b2", [50, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1, 50], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [1, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

# FORWARD PROPAGATION:

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
                                                           # Numpy Equivalents:
    Z1 = tf.matmul(W1, X) + b1                             # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2                            # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3                            # Z3 = np.dot(W3, A2) + b3
    
    return Z3

# COST FUNCTION:

def compute_cost(Z3, Y, parameters, lambd):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (1, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    lambd -- L2 regularization parameter
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
    cost = tf.reduce_mean(cost)
    cost += tf.add_n([ tf.nn.l2_loss(W) for W in [W1, W2, W3] ]) * lambd
    
    return cost

# MODEL:

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, lambd = 0.01,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 2, number of training examples = )
    Y_train -- training set, of shape (output size = 1, number of training examples = )
    X_test -- test set, of shape (input size = 1, number of training examples = )
    Y_test -- test set, of shape (output size = 1, number of test examples = )
    learning_rate -- learning rate of the optimization
    lambd -- L2 regularization parameter
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
#     ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = list()                                    # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y, parameters, lambd)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feed_dict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate) + "\n" + "Lambda =" + str(lambd))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        Y_hat = tf.sigmoid(Z3)
        Y_hat = tf.round(Y_hat)                    # banker's rounding..
        correct_prediction = tf.equal(Y_hat, Y)

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Validation Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
    
# ============================Main Program====================================

# IMPORT DATA:

# Data for training:

X_train = importer('X_train', 'inp')
Y_train = importer('Y_train', 'out')

# Data for evaluation:

X_dev = importer('X_dev', 'inp')
Y_dev = importer('Y_dev', 'out')

# TRAINING SET SIZE:

m = X_train.shape[1]                                                # number of training examples
n_x = X_train.shape[0]                                              # number of input variables
# print(n_x, m)

# TRAIN THE DEEP NEURAL NETWORK:

plt.figure(0)

parameters = model(X_train, Y_train, X_dev, Y_dev, lambd = 0.01, num_epochs = 1500, minibatch_size = 128)

# FEEDFARWARD PROPAGATION FOR TRAINING:

Y_hat = feedfor(parameters, X_train)

# MEASURES TO EVALUATE LEARNING ALGORITHM:

Con_matrix = dict()

Prec = dict()
Rec = dict()
Acc = dict()
F1_Score = dict()

Con_matrix['Train'], Acc['Train'], Prec['Train'], Rec['Train'], F1_Score['Train'] = Conf_matrix(Y_train, Y_hat)

# PRINT THE MEASURES TO EVALUATE THE LEARNING ALGORITHM FOR THE TRAINING SET:

# Accuracy

print ('Accuracy of 3 layer NN in training: %.2f ' % float(Acc['Train']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# Precision

print ('Precision of 3 layer NN in training: %.2f ' % float(Prec['Train']*100) +
       '% ')

# Recall

print ('Recall of 3 layer NN in training: %.2f ' % float(Rec['Train']*100) +
       '% ')

# F1-Score

print ('F1-Score of 3 layer NN in training: %.2f ' % float(F1_Score['Train']*100) +
       '% ')

# FEEDFARWARD PROPAGATION FOR EVALUATION:

Y_hat = feedfor(parameters, X_dev)

Con_matrix['Dev'], Acc['Dev'], Prec['Dev'], Rec['Dev'], F1_Score['Dev'] = Conf_matrix(Y_dev, Y_hat)

# PRINT THE MEASURES TO EVALUATE THE LEARNING ALGORITHM FOR THE EVALUATION SET:

# Accuracy

print ('Accuracy of 3 layer NN in evaluation: %.2f ' % float(Acc['Dev']*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# Precision

print ('Precision of 3 layer NN in evaluation: %.2f ' % float(Prec['Dev']*100) +
       '% ')

# Recall

print ('Recall of 3 layer NN in evaluation: %.2f ' % float(Rec['Dev']*100) +
       '% ')

# F1-Score

print ('F1-Score of 3 layer NN in evaluation: %.2f ' % float(F1_Score['Dev']*100) +
       '% ')
