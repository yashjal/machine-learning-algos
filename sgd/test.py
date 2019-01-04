import sys
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
# from sklearn.linear_model import SGDClassifier

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
    loss = np.dot(X.T, np.dot(X, theta) - y) * 2 / (y.shape[0])
    return loss


def batch_grad_descent(X, y, alpha=0.1, num_iter=100, check_gradient=False):
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


#X = np.array([[0., 0.], [1., 1.]])
#y = np.array([0, 1])

z1 = batch_grad_descent(X_train,y_train)[1]
z2 = batch_grad_descent(X_train,y_train,0.5)[1]
z3 = batch_grad_descent(X_train,y_train,0.01)[1]
z4 = batch_grad_descent(X_train,y_train,0.05)[1]


t1 = np.arange(0.0, 101, 1.0)
x1, = plt.plot(t1, z1, label='step length = 0.1')
x2, = plt.plot(t1, z2, label='step length = 0.5')
x3, = plt.plot(t1, z3, label='step length = 0.01')
x4, = plt.plot(t1, z4, label='step length = 0.05')

plt.ylabel('J(\{theta})')
plt.xlabel('# of steps')
plt.title('Batch Gradient Descent with different step sizes')
plt.legend([x1,x2,x3,x4], ['step length = 0.1','step length = 0.5','step length = 0.01','step length = 0.05'])
plt.show()

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
    loss = np.dot(X.T, np.dot(X, theta) - y) * 2 / (y.shape[0])
    return loss


x = np.array([[0.5, 0.25],[0.25, 0.5],[0.5,0.75]])
y = np.array([1, 2, 1])
the = np.array([1,2])

print(compute_square_loss(x, y, the))
print(compute_square_loss_gradient(x, y, the))

z = x[0]
z = (z - 1)
x[0] = z
print(z)
print(x)
print(np.amax(z))
