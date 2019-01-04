import pandas as pd
import logging
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from numpy import linalg as LA

#######################################
#### Normalization


def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    tr = train.T
    ts = test.T
    for i in range(train.shape[1]):
        ft = tr[i]
        tst = ts[i]
        mx = np.amax(ft)
        mn = np.amin(ft)
        if mn != mx:
            ft = (ft - mn) / (mx - mn)
            tst = (tst - mn) / (mx - mn)
        train.T[i] = ft
        test.T[i] = tst
    return train, test


########################################
#### The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """
    #  loss = 0 #initialize the square_loss
    loss = np.sum(np.square(np.dot(X,theta) - y))/(y.shape[0])
    return loss


########################################
### compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    loss = np.dot(X.T, np.dot(X, theta) - y) * 2 /(y.shape[0])
    return loss



###########################################
### Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1)

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO

#################################################
### Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO


####################################
#### Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=10000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeros(num_features) #initialize theta

    loss = compute_square_loss(X, y, theta)
    loss_hist[0] = loss    
    i = 1
    grad = compute_square_loss_gradient(X, y, theta)
    norm = LA.norm(grad, 2)
    while (i <= num_iter) and (norm >= 1.e-6):
        theta = theta - grad * alpha / norm
        theta_hist[i] = theta
        loss = compute_square_loss(X, y, theta)
        loss_hist[i] = loss
        grad = compute_square_loss_gradient(X, y, theta)
        norm = LA.norm(grad, 2)
        i = i + 1

    return theta_hist, loss_hist
    

####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO



###################################################
### Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    loss = np.dot(X.T, np.dot(X, theta) - y) * 2 / (y.shape[0]) +\
           2 * lambda_reg * theta
    return loss
    

###################################################
### Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of loss function without the regularization term, 1D numpy array.
    """
    (num_instances, num_features) = X.shape
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist

    loss = compute_square_loss(X, y, theta)
    loss_hist[0] = loss    
    i = 1
    grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
    norm = LA.norm(grad, 2)
    while (i <= num_iter) and (norm >= 1.e-6):
        theta = theta - grad * alpha / norm
        theta_hist[i] = theta
        loss = compute_square_loss(X, y, theta)
        loss_hist[i] = loss
        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        norm = LA.norm(grad, 2)
        i = i + 1

    return theta_hist, loss_hist


#############################################
## Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss

###helper function
def compute_square_loss_reg(X, y, theta, lambda_reg):
    
    loss = np.sum(np.square(np.dot(X,theta) - y))/(y.shape[0]) + lambda_reg * np.sum(np.square(theta))
    return loss

#############################################
### Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.05, lambda_reg=1.e-5, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == 100, alpha = 1/sqrt(t)
                if alpha == 1000, alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta

    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist

    test = alpha
    i = 0
    while (i <= num_iter):
        
        if test == 100 and i >= 10:
            alpha = 1/math.sqrt(i)
        elif test == 1000 and i >= 10:
            alpha = 1/i
        elif test == 1000 or test == 100:
            alpha = 0.05

            
        #Epoch: using same ordering (ascending order)
        for j in range(num_instances):
            grad = np.dot((theta.T, X[j]) - y[j]) * X[j]
            grad[j] += lambda_reg * theta[j]
            grad = 2 * grad
            norm = LA.norm(grad, 2)
            theta = theta - grad * alpha / norm
            #Initial zero vector not included in theta_hist
            theta_hist[i][j] = theta
            loss = compute_square_loss_reg(X, y, theta, lambda_reg)
            loss_hist[i][j] = loss       

        i = i + 1

    grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
    norm = LA.norm(grad, 2)
    print('grad norm: ', norm)
    return theta_hist, loss_hist
    

################################################
### Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value) and/or objective_function_value

def main():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Add bias term

    # TODO
    

if __name__ == "__main__":
    main()
